#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# fix-qwen3.5-linear-attn-mod/run.sh  (v9)
#
# Fixes Qwen-3.5 MoE hybrid attention + Pipeline Parallelism crashes.
#
# Root cause 1: get_attn_backends_for_group() iterates all layer names
# from KV cache group spec, but get_layers_from_vllm_config() only
# returns local AttentionLayerBase instances.  Non-local layers
# (from other PP stages) cause KeyError.
#
# Root cause 2: _cleanup_profiling_kv_cache() sets layer.kv_cache=[]
# for ALL layers after profiling.  If the subsequent
# initialize_kv_cache_tensors returns empty kv_caches (due to the
# same non-local layer issue in attn_groups), bind_kv_cache({}) is
# a no-op and layers keep kv_cache=[], causing IndexError in forward.
#
# v9 approach:
#   - Guard in get_attn_backends_for_group (same as v8)
#   - ALSO guard init_attn_backend() in attn_utils.py (the secondary
#     code path) for the same non-local layer issue
#   - Enhanced diagnostics at EVERY stage of initialize_kv_cache
#   - SAFETY: After bind_kv_cache, restore PP-sized placeholder for
#     any layer that still has kv_cache=[] so IndexError is prevented
# ─────────────────────────────────────────────────────────────────────
set -e

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Fix Qwen3.5 Linear Attention + PP — Installing (v9)     ║"
echo "╚═══════════════════════════════════════════════════════════╝"

VLLM_DIR=$(python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))")
echo "  vLLM dir: $VLLM_DIR"

# ═══════════════════════════════════════════════════════════════════════
# PATCH 1: gpu_model_runner.py — guard + deep diagnostics + safety net
# ═══════════════════════════════════════════════════════════════════════
python3 << 'PYEOF'
import re, sys, os, shutil, py_compile, vllm

vllm_dir = os.path.dirname(vllm.__file__)
target = os.path.join(vllm_dir, "v1", "worker", "gpu_model_runner.py")

with open(target, "r") as f:
    source = f.read()

if "QWEN35_FIX" in source:
    print("  · gpu_model_runner.py already patched, skipping")
    sys.exit(0)

backup = target + ".qwen35fix.bak"
shutil.copy2(target, backup)

lines = source.split('\n')

# ── Step 1: Build a map of which function each line belongs to ──
func_of = [None] * len(lines)
func_stack = []

for i, line in enumerate(lines):
    stripped = line.lstrip()
    if not stripped or stripped.startswith('#'):
        func_of[i] = func_stack[-1][0] if func_stack else None
        continue
    indent = len(line) - len(stripped)
    while func_stack and indent <= func_stack[-1][1]:
        func_stack.pop()
    m = re.match(r'(async\s+)?def\s+(\w+)\s*\(', stripped)
    if m:
        func_stack.append((m.group(2), indent))
    func_of[i] = func_stack[-1][0] if func_stack else None

# ── Step 2: Find for-loops over layer_name in target functions ──
TARGET_KEYWORDS = ['kv_cache', 'attn_backend']

def should_patch(func_name):
    if not func_name:
        return False
    low = func_name.lower()
    return any(kw in low for kw in TARGET_KEYWORDS)

loops_to_patch = []
for i, line in enumerate(lines):
    if not should_patch(func_of[i]):
        continue
    m = re.match(r'^(\s*)for\s+layer_name\s+in\s+', line)
    if m:
        loops_to_patch.append((i, len(m.group(1)), func_of[i]))

print(f"  Found {len(loops_to_patch)} for-loop(s) in target functions:")
for idx, indent, fn in loops_to_patch:
    print(f"    L{idx+1} in {fn}(): {lines[idx].strip()[:60]}")

if not loops_to_patch:
    print("  ⚠ No loops matched!")
    os.remove(backup)
    sys.exit(1)

# ── Step 3: Insert guard ──
total_patched = 0
for loop_idx, loop_indent, func_name in reversed(loops_to_patch):
    body_indent_str = ' ' * (loop_indent + 4)

    dict_expr = None
    for k in range(loop_idx + 1, min(loop_idx + 40, len(lines))):
        kline = lines[k]
        kstripped = kline.lstrip()
        if not kstripped or kstripped.startswith('#'):
            continue
        k_indent = len(kline) - len(kstripped)
        if k_indent <= loop_indent:
            break
        dm = re.search(r'([\w][\w.]*)\[layer_name\]', kstripped)
        if dm:
            dict_expr = dm.group(1)
            break

    if not dict_expr:
        print(f"  ⚠ L{loop_idx+1}: no dict[layer_name] found, skipping")
        continue

    guard_lines = [
        f"{body_indent_str}if layer_name not in {dict_expr}:  # QWEN35_FIX v9",
        f"{body_indent_str}    continue",
    ]
    for gi, gl in enumerate(guard_lines):
        lines.insert(loop_idx + 1 + gi, gl)
    total_patched += 1
    print(f"  ✓ L{loop_idx+1} in {func_name}(): guard inserted")

if total_patched == 0:
    print("  ✗ No guards inserted!")
    os.remove(backup)
    sys.exit(1)

# ── Step 4: Add diagnostics at bind_kv_cache + safety net ──
source_after_guard = '\n'.join(lines)

diag_pattern = r'(\n)([ \t]+)(bind_kv_cache\()'
diag_match = re.search(diag_pattern, source_after_guard)

if diag_match:
    call_offset = diag_match.start(3)
    paren_depth = 0
    end_pos = call_offset
    for ci in range(call_offset, len(source_after_guard)):
        ch = source_after_guard[ci]
        if ch == '(':
            paren_depth += 1
        elif ch == ')':
            paren_depth -= 1
            if paren_depth == 0:
                nl = source_after_guard.find('\n', ci)
                if nl == -1:
                    nl = len(source_after_guard)
                end_pos = nl
                break

    indent = diag_match.group(2)
    # Also inject diagnostics BEFORE bind_kv_cache
    pre_diag = (
        f"\n{indent}# QWEN35_FIX v9: pre-bind diagnostics\n"
        f"{indent}import logging as _q35log\n"
        f"{indent}_q35logger = _q35log.getLogger('qwen35_fix')\n"
        f"{indent}_q35logger.setLevel(_q35log.INFO)\n"
        f"{indent}if not _q35logger.handlers:\n"
        f"{indent}    _q35logger.addHandler(_q35log.StreamHandler())\n"
        f"{indent}_q35logger.info('=== QWEN35_FIX v9 PRE-BIND DIAGNOSTIC ===')\n"
        f"{indent}_q35_grps = getattr(self, 'kv_cache_config', None)\n"
        f"{indent}if _q35_grps is not None:\n"
        f"{indent}    _q35logger.info('kv_cache_config.kv_cache_groups: %d groups, layer_counts=%s',\n"
        f"{indent}        len(_q35_grps.kv_cache_groups),\n"
        f"{indent}        [len(g.layer_names) for g in _q35_grps.kv_cache_groups])\n"
        f"{indent}    _q35logger.info('kv_cache_config.kv_cache_tensors: %d tensors',\n"
        f"{indent}        len(_q35_grps.kv_cache_tensors))\n"
        f"{indent}else:\n"
        f"{indent}    _q35logger.info('kv_cache_config: NOT SET')\n"
        f"{indent}_q35logger.info('self.attn_groups: %d outer, inner_counts=%s',\n"
        f"{indent}    len(self.attn_groups),\n"
        f"{indent}    [[len(g.layer_names) for g in grp] for grp in self.attn_groups])\n"
        f"{indent}_q35logger.info('kv_caches dict: %d entries', len(kv_caches))\n"
        f"{indent}if len(kv_caches) == 0 and hasattr(self, 'kv_cache_config') and self.kv_cache_config is not None:\n"
        f"{indent}    _q35_total = sum(len(g.layer_names) for g in self.kv_cache_config.kv_cache_groups)\n"
        f"{indent}    if _q35_total > 0:\n"
        f"{indent}        _q35logger.warning('BUG: kv_caches is empty but config has %d layers in %d groups!',\n"
        f"{indent}            _q35_total, len(self.kv_cache_config.kv_cache_groups))\n"
        f"{indent}        for _gi, _grp in enumerate(self.kv_cache_config.kv_cache_groups):\n"
        f"{indent}            _q35logger.warning('  group[%d]: %d layers, spec=%s, names=%s',\n"
        f"{indent}                _gi, len(_grp.layer_names), type(_grp.kv_cache_spec).__name__,\n"
        f"{indent}                _grp.layer_names[:3])\n"
    )

    # Post-bind diagnostics + safety net
    post_diag = (
        f"\n{indent}# QWEN35_FIX v9: post-bind safety net\n"
        f"{indent}import torch as _q35torch\n"
        f"{indent}_q35_pp = self.parallel_config.pipeline_parallel_size\n"
        f"{indent}_q35_fctx = self.compilation_config.static_forward_context\n"
        f"{indent}_q35_fixed = 0\n"
        f"{indent}for _ln, _layer in _q35_fctx.items():\n"
        f"{indent}    _kc = getattr(_layer, 'kv_cache', None)\n"
        f"{indent}    if isinstance(_kc, list) and len(_kc) == 0:\n"
        f"{indent}        _layer.kv_cache = [_q35torch.tensor([]) for _ in range(_q35_pp)]\n"
        f"{indent}        _q35_fixed += 1\n"
        f"{indent}if _q35_fixed > 0:\n"
        f"{indent}    _q35logger.warning('SAFETY: Restored PP-sized placeholder for %d layers with empty kv_cache', _q35_fixed)\n"
        f"{indent}_q35logger.info('=== END QWEN35_FIX v9 DIAGNOSTIC ===')\n"
    )

    # Insert pre-diag BEFORE bind_kv_cache, post-diag AFTER
    bind_start = diag_match.start()
    source_after_guard = (
        source_after_guard[:bind_start] +
        pre_diag +
        source_after_guard[bind_start:end_pos] +
        post_diag +
        source_after_guard[end_pos:]
    )
    print("  ✓ Pre-bind diagnostics + post-bind safety net added")
else:
    print("  ⚠ Could not find bind_kv_cache call")

with open(target, "w") as f:
    f.write(source_after_guard)

try:
    py_compile.compile(target, doraise=True)
    print(f"  ✓ gpu_model_runner.py: {total_patched} guard(s) + diagnostics + safety, syntax OK")
    os.remove(backup)
except py_compile.PyCompileError as e:
    print(f"  ✗ SYNTAX ERROR: {e}")
    shutil.move(backup, target)
    sys.exit(1)
PYEOF

# ═══════════════════════════════════════════════════════════════════════
# PATCH 2: attn_utils.py — guard for the SAME non-local layer issue
# ═══════════════════════════════════════════════════════════════════════
python3 << 'PYEOF'
import re, sys, os, shutil, py_compile, vllm

vllm_dir = os.path.dirname(vllm.__file__)
target = os.path.join(vllm_dir, "v1", "worker", "gpu", "attn_utils.py")

with open(target, "r") as f:
    source = f.read()

if "QWEN35_FIX" in source:
    print("  · attn_utils.py already patched, skipping")
    sys.exit(0)

backup = target + ".qwen35fix.bak"
shutil.copy2(target, backup)

# In init_attn_backend, line 48-49:
#   for layer_name in layer_names:
#       attn_backend = attn_layers[layer_name].get_attn_backend()
# Need to guard: if layer_name not in attn_layers: continue
patched = source.replace(
    "        for layer_name in layer_names:\n"
    "            attn_backend = attn_layers[layer_name].get_attn_backend()",
    "        for layer_name in layer_names:\n"
    "            if layer_name not in attn_layers:  # QWEN35_FIX v9\n"
    "                continue\n"
    "            attn_backend = attn_layers[layer_name].get_attn_backend()"
)

if patched == source:
    print("  · attn_utils.py: pattern not found (may not need patching)")
    os.remove(backup)
else:
    with open(target, "w") as f:
        f.write(patched)
    try:
        py_compile.compile(target, doraise=True)
        print("  ✓ attn_utils.py: guard added in init_attn_backend, syntax OK")
        os.remove(backup)
    except py_compile.PyCompileError as e:
        print(f"  ✗ SYNTAX ERROR: {e}")
        shutil.move(backup, target)
        sys.exit(1)
PYEOF

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Fix Qwen3.5 Linear Attention + PP — Complete (v9)       ║"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  Patch 1: Guard non-local layers in gpu_model_runner.py   ║"
echo "║  Patch 2: Guard non-local layers in attn_utils.py         ║"
echo "║  + Deep diagnostics at pre/post bind_kv_cache             ║"
echo "║  + Safety: Restore PP-sized placeholder for empty kv_cache ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
