#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# ray-v2-executor-mod/run.sh
#
# Installs RayV2Executor into the vLLM package at container runtime.
# Applied via: --apply-mod ray-v2-executor-mod
#
# What this does:
#   1. Copies ray_v2_executor.py into vllm/v1/executor/
#   2. Patches abstract.py so --distributed-executor-backend ray → RayV2Executor
#   3. Patches parallel_state.py for Gloo/NCCL interface lock + mesh stagger
#      (reads GLOO_SOCKET_IFNAME, NCCL_SOCKET_IFNAME, NCCL_MESH_STAGGER_SEC
#       from container environment — no hardcoded interface names)
#
# After applying, launch with:
#   --distributed-executor-backend ray
# ─────────────────────────────────────────────────────────────────────
set -e

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           RayV2 Executor Mod — Installing                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ── Step 1: Locate vLLM install directory ─────────────────────────────
VLLM_DIR=$(python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))")
EXECUTOR_DIR="$VLLM_DIR/v1/executor"

echo "  vLLM package:  $VLLM_DIR"
echo "  Executor dir:  $EXECUTOR_DIR"

if [ ! -d "$EXECUTOR_DIR" ]; then
    echo "ERROR: Executor directory not found at $EXECUTOR_DIR"
    exit 1
fi

# ── Step 2: Install ray_v2_executor.py ────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$SCRIPT_DIR/ray_v2_executor.py" "$EXECUTOR_DIR/ray_v2_executor.py"
echo "  ✓ Installed ray_v2_executor.py → $EXECUTOR_DIR/"

# Verify import
python3 -c "
from vllm.v1.executor.ray_v2_executor import RayV2Executor
print(f'  ✓ Import verified: {RayV2Executor.__name__}')
print(f'    uses_ray={RayV2Executor.uses_ray}, supports_pp={RayV2Executor.supports_pp}')
"

# ── Step 3: Patch abstract.py so "ray" backend → RayV2Executor ───────
#    vLLM's arg_utils._check_feature_supported() only allows PP with
#    "ray", "mp", or "external_launcher" backends. A custom qualname
#    string fails validation before the executor even loads.
#    Fix: keep --distributed-executor-backend ray but rewire the import.
echo "  Patching abstract.py (executor registry)..."
python3 -c "
import os

file_path = os.path.join('$EXECUTOR_DIR', 'abstract.py')
with open(file_path, 'r') as f:
    content = f.read()

old = '''elif distributed_executor_backend == \"ray\":
            from vllm.v1.executor.ray_executor import RayDistributedExecutor

            executor_class = RayDistributedExecutor'''

new = '''elif distributed_executor_backend == \"ray\":
            from vllm.v1.executor.ray_v2_executor import RayV2Executor

            executor_class = RayV2Executor'''

if 'ray_v2_executor' not in content:
    if old in content:
        content = content.replace(old, new)
        with open(file_path, 'w') as f:
            f.write(content)
        print('    ✓ abstract.py patched: ray → RayV2Executor')
    else:
        print('    ⚠ Could not find expected ray executor import in abstract.py')
        print('      Trying flexible match...')
        # Try a more flexible match
        import re
        pattern = r'elif distributed_executor_backend == .ray.:\s+from vllm\.v1\.executor\.ray_executor import RayDistributedExecutor\s+executor_class = RayDistributedExecutor'
        replacement = '''elif distributed_executor_backend == \"ray\":
            from vllm.v1.executor.ray_v2_executor import RayV2Executor

            executor_class = RayV2Executor'''
        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print('    ✓ abstract.py patched (flexible match): ray → RayV2Executor')
        else:
            print('    ✗ FAILED: Could not patch abstract.py — executor registry unchanged')
            exit(1)
else:
    print('    · abstract.py already patched, skipping')
"

# ── Step 4: Patch parallel_state.py (Gloo lock + mesh stagger) ───────
echo "  Patching parallel_state.py..."
python3 -c "
import os

file_path = os.path.join('$VLLM_DIR', 'distributed', 'parallel_state.py')
with open(file_path, 'r') as f:
    content = f.read()

patched = False

# 4a. Lock Gloo + NCCL to the management interface
#     Required for switchless direct-connect mesh topologies where
#     the RDMA interfaces don't carry IP-routable management traffic.
#     Reads from GLOO_SOCKET_IFNAME / NCCL_SOCKET_IFNAME env vars
#     which the user sets via docker -e flags. These get overwritten
#     by autodiscover.sh → run-cluster-node.sh, so we re-assert them
#     at Python import time (which runs later, during vLLM init).
#     If neither env var is set at patch time, this step is skipped
#     (user is probably on a switched fabric and doesn't need it).

import os as _patch_os
_gloo_if = _patch_os.environ.get('GLOO_SOCKET_IFNAME', '')
_nccl_if = _patch_os.environ.get('NCCL_SOCKET_IFNAME', _gloo_if)

if _gloo_if:
    gloo_lock = '''
import os as _os
import time as _time
# Re-assert management interface for Gloo/NCCL socket traffic.
# autodiscover.sh + run-cluster-node.sh may overwrite these with a
# RoCE net device that carries fabric IPs (not routable for rendezvous).
# Values come from docker -e GLOO_SOCKET_IFNAME=... / NCCL_SOCKET_IFNAME=...
_os.environ['GLOO_SOCKET_IFNAME'] = '{gloo}'
_os.environ['NCCL_SOCKET_IFNAME'] = '{nccl}'
'''.format(gloo=_gloo_if, nccl=_nccl_if)
    if 'GLOO_SOCKET_IFNAME' not in content:
        content = gloo_lock + content
        patched = True
        print(f'    ✓ Gloo/NCCL interface lock applied (GLOO={_gloo_if}, NCCL={_nccl_if})')
    else:
        print('    · Gloo/NCCL interface lock already present, skipping')
else:
    print('    · GLOO_SOCKET_IFNAME not set in environment, skipping interface lock')
    print('      (Set -e GLOO_SOCKET_IFNAME=<mgmt_if> in docker args if needed)')

# 4b. Rank-based stagger for NCCL mesh plugin handshake
#     The custom mesh plugin does bidirectional QP exchange + background
#     handshake threads. Staggering rank init prevents connection storms
#     on clusters without a switch.
#     NCCL_MESH_STAGGER_SEC env var controls seconds per rank (default: 3)
anchor = '    # set the local rank'

_stagger_sec = _patch_os.environ.get('NCCL_MESH_STAGGER_SEC', '3')
injection = '''
    # --- NCCL MESH PLUGIN STAGGER (RayV2 Mod) ---
    try:
        _rank = torch.distributed.get_rank()
        _stagger = _rank * {stagger_sec}
        if _stagger > 0:
            print(f'[RayV2] Rank {{_rank}} sleeping {{_stagger:.1f}}s for NCCL mesh stagger', flush=True)
            _time.sleep(_stagger)
        print(f'[RayV2] Rank {{_rank}} entering NCCL data plane.', flush=True)
    except Exception as _e:
        print(f'[RayV2] Mesh stagger failed: {{_e}}', flush=True)
    # --- END NCCL MESH PLUGIN STAGGER ---

    # set the local rank'''.format(stagger_sec=_stagger_sec)

if anchor in content and 'NCCL MESH PLUGIN STAGGER' not in content:
    # Also skip if the older vllm-patch-mod stagger is present
    if 'MESH PLUGIN INJECTION START' not in content:
        content = content.replace(anchor, injection)
        patched = True
        print('    ✓ Mesh plugin rank stagger applied (3s per rank)')
    else:
        print('    · Older mesh stagger already present, skipping')
else:
    if 'NCCL MESH PLUGIN STAGGER' in content:
        print('    · Mesh plugin rank stagger already present, skipping')
    else:
        print('    ⚠ Could not find anchor for mesh stagger patch')

if patched:
    with open(file_path, 'w') as f:
        f.write(content)
    print('    ✓ parallel_state.py written')
else:
    print('    · parallel_state.py unchanged (all patches already applied)')
"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           RayV2 Executor Mod — Complete                   ║"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  Use with:                                                ║"
echo "║  --distributed-executor-backend ray                       ║"
echo "║  (abstract.py patched: 'ray' now loads RayV2Executor)     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
