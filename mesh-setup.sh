#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# mesh-setup.sh
#
# One-time setup for switchless direct-connect RDMA mesh clusters.
# Run this ONCE on the head node after cloning the repo and before
# the first launch. It will:
#
#   1. Clone + build the NCCL mesh plugin (libnccl-net.so)
#   2. Copy the built library to all nodes in the cluster
#   3. Print the environment variables and launch command to use
#
# Prerequisites:
#   - Passwordless SSH to all worker nodes
#   - Build tools: gcc, make
#   - libibverbs-dev, librdmacm-dev (installed automatically if missing)
#   - All nodes must be the same architecture (e.g., all aarch64)
#
# Usage:
#   ./mesh-setup.sh [OPTIONS]
#
#   Options:
#     --nodes <ip1,ip2,...>   Comma-separated node IPs (required)
#     --mgmt-if <name>       Management network interface for Gloo/NCCL
#                             socket traffic (e.g., enP7s7, eth0). Required.
#     --ib-hca <devs>        Comma-separated RDMA HCA device names
#                             (e.g., rocep1s0f0,roceP2p1s0f0). Optional,
#                             auto-detected if omitted.
#     --gid-index <n>        RoCE GID index (default: 3)
#     --stagger-sec <n>      Seconds of stagger per rank during NCCL init
#                             (default: 3). Prevents connection storms.
#     --model-path <path>    Local path to model weights. If set, will be
#                             included in the generated docker env vars.
#     --skip-build           Skip building the plugin (use existing lib/)
#     --skip-copy            Skip copying to remote nodes
#     --help                 Show this help
# ─────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLUGIN_DIR="$SCRIPT_DIR/nccl-mesh-plugin"
LIB_DIR="$SCRIPT_DIR/lib"
PLUGIN_REPO="https://github.com/autoscriptlabs/nccl-mesh-plugin.git"
PLUGIN_COMMIT="b3acb65bc44407610a66638191df09b6aa8a1f97"

NODES_ARG=""
MGMT_IF=""
IB_HCA=""
GID_INDEX="3"
STAGGER_SEC="3"
MODEL_PATH=""
SKIP_BUILD="false"
SKIP_COPY="false"

usage() {
    sed -n '/^# Usage:/,/^# ──/p' "$0" | head -n -1 | sed 's/^# //' | sed 's/^#//'
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nodes)       NODES_ARG="$2"; shift ;;
        --mgmt-if)     MGMT_IF="$2"; shift ;;
        --ib-hca)      IB_HCA="$2"; shift ;;
        --gid-index)   GID_INDEX="$2"; shift ;;
        --stagger-sec) STAGGER_SEC="$2"; shift ;;
        --model-path)  MODEL_PATH="$2"; shift ;;
        --skip-build)  SKIP_BUILD="true" ;;
        --skip-copy)   SKIP_COPY="true" ;;
        --help)        usage ;;
        *)             echo "Unknown option: $1"; usage ;;
    esac
    shift
done

if [[ -z "$NODES_ARG" ]]; then
    echo "ERROR: --nodes is required"
    echo "  Example: --nodes 192.168.3.105,192.168.3.106,192.168.3.107"
    exit 1
fi

if [[ -z "$MGMT_IF" ]]; then
    echo "ERROR: --mgmt-if is required"
    echo "  This is the network interface used for management traffic"
    echo "  (Gloo rendezvous, NCCL socket bootstrap, SSH, Ray)."
    echo "  Example: --mgmt-if enP7s7"
    echo ""
    echo "  To find it, run: ip -br addr show | grep UP"
    echo "  Pick the interface with your node's management IP."
    exit 1
fi

IFS=',' read -r -a ALL_NODES <<< "$NODES_ARG"
HEAD_IP="${ALL_NODES[0]}"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║              Mesh Cluster Setup                           ║"
echo "╠═══════════════════════════════════════════════════════════╣"
echo "║  Nodes:     ${NODES_ARG}                                  ║"
echo "║  Head:      ${HEAD_IP}                                    ║"
echo "║  Mgmt IF:   ${MGMT_IF}                                    ║"
echo "║  GID Index: ${GID_INDEX}                                  ║"
echo "║  Stagger:   ${STAGGER_SEC}s per rank                      ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# ── 1: clone +build mesh plugin ────────────────────────────
if [[ "$SKIP_BUILD" == "true" ]]; then
    echo "── Step 1: Build (skipped) ──"
    if [[ ! -f "$LIB_DIR/libnccl-net.so" ]]; then
        echo "ERROR: --skip-build but $LIB_DIR/libnccl-net.so not found"
        exit 1
    fi
    echo "  Using existing $LIB_DIR/libnccl-net.so"
else
    echo "── Step 1: Building NCCL mesh plugin ──"

    # basically just follow the instructions in the plugin README, but with some automation and error handling
    if ! dpkg -s libibverbs-dev &>/dev/null 2>&1; then
        echo "  Installing build dependencies..."
        sudo apt-get update -qq && sudo apt-get install -y -qq libibverbs-dev librdmacm-dev
    fi

    if [[ -d "$PLUGIN_DIR/.git" ]]; then
        echo "  Plugin source exists, fetching updates..."
        cd "$PLUGIN_DIR"
        git fetch origin
    else
        echo "  Cloning nccl-mesh-plugin..."
        git clone "$PLUGIN_REPO" "$PLUGIN_DIR"
        cd "$PLUGIN_DIR"
    fi

    # select the commit i used so that it actually works
    echo "  Checking out commit $PLUGIN_COMMIT..."
    git checkout "$PLUGIN_COMMIT" --quiet

    echo "  Building..."
    make clean 2>/dev/null || true
    make -j"$(nproc)"

    mkdir -p "$LIB_DIR"
    cp "$PLUGIN_DIR/libnccl-net.so" "$LIB_DIR/libnccl-net.so"
    echo "  Built: $LIB_DIR/libnccl-net.so"
    echo "    $(file "$LIB_DIR/libnccl-net.so" | cut -d: -f2)"

    cd "$SCRIPT_DIR"
fi

echo ""

# ── 2: copy this shit to all nodes ─────────────────────────────────────────
if [[ "$SKIP_COPY" == "true" ]]; then
    echo "── Step 2: Copy to nodes (skipped) ──"
else
    echo "── Step 2: Copying libnccl-net.so to all nodes ──"

    for node in "${ALL_NODES[@]}"; do
        node=$(echo "$node" | xargs)
        if ip addr show | grep -q "inet ${node}/"; then
            echo "  $node (local) — already built here"
            continue
        fi

        echo -n "  $node — "
        if ! ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$node" true 2>/dev/null; then
            echo "FAILED (SSH unreachable)"
            echo "  WARNING: You'll need to manually copy lib/libnccl-net.so to $node"
            continue
        fi

        ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$node" "mkdir -p $LIB_DIR"
        scp -o BatchMode=yes -o StrictHostKeyChecking=no "$LIB_DIR/libnccl-net.so" "$node:$LIB_DIR/libnccl-net.so"
        echo "OK"
    done
fi

echo ""

# ── 3: idk if this is useful but if shit isnt provided we defualt to finding if's like this ──────────────────────
if [[ -z "$IB_HCA" ]]; then
    echo "── Step 3: Auto-detecting RDMA HCAs ──"
    if command -v ibdev2netdev &>/dev/null; then
        IB_HCA=$(ibdev2netdev | awk '/Up\)/ {print $1}' | paste -sd, -)
        if [[ -n "$IB_HCA" ]]; then
            echo "  Detected: $IB_HCA"
        else
            echo "  WARNING: No active IB devices found via ibdev2netdev"
            echo "  You may need to set --ib-hca manually"
        fi
    else
        echo "  WARNING: ibdev2netdev not found, cannot auto-detect HCAs"
        echo "  You may need to set --ib-hca manually"
    fi
else
    echo "── Step 3: Using provided IB HCAs: $IB_HCA ──"
fi

echo ""

# ── 4: generate launch command ─────────────────────
echo "── Step 4: Configuration ──"
echo ""

PLUGIN_MOUNT="$LIB_DIR/libnccl-net.so:/opt/nccl-mesh/libnccl-net.so"

# Build the VLLM_SPARK_EXTRA_DOCKER_ARGS
EXTRA_ARGS=""
EXTRA_ARGS="$EXTRA_ARGS -v $PLUGIN_MOUNT"
if [[ -n "$MODEL_PATH" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS -v $MODEL_PATH:/model_data"
fi
EXTRA_ARGS="$EXTRA_ARGS -e NCCL_NET_PLUGIN=/opt/nccl-mesh/libnccl-net.so"
EXTRA_ARGS="$EXTRA_ARGS -e NCCL_MESH_GID_INDEX=$GID_INDEX"
EXTRA_ARGS="$EXTRA_ARGS -e NCCL_MESH_STAGGER_SEC=$STAGGER_SEC"
EXTRA_ARGS="$EXTRA_ARGS -e NCCL_DEBUG=INFO"
EXTRA_ARGS="$EXTRA_ARGS -e NCCL_SOCKET_IFNAME=$MGMT_IF"
EXTRA_ARGS="$EXTRA_ARGS -e GLOO_SOCKET_IFNAME=$MGMT_IF"
if [[ -n "$IB_HCA" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS -e NCCL_IB_HCA=$IB_HCA"
fi

# Write env file for convenience
ENV_FILE="$SCRIPT_DIR/mesh-env.sh"
cat > "$ENV_FILE" << ENVEOF
#!/bin/bash
# Generated by mesh-setup.sh on $(date -Iseconds)
# Source this before launching: source mesh-env.sh

export VLLM_SPARK_EXTRA_DOCKER_ARGS="\\
  -v $PLUGIN_MOUNT \\
$(if [[ -n "$MODEL_PATH" ]]; then echo "  -v $MODEL_PATH:/model_data \\"; fi)
  -e NCCL_NET_PLUGIN=/opt/nccl-mesh/libnccl-net.so \\
  -e NCCL_MESH_GID_INDEX=$GID_INDEX \\
  -e NCCL_MESH_STAGGER_SEC=$STAGGER_SEC \\
  -e NCCL_DEBUG=INFO \\
  -e NCCL_SOCKET_IFNAME=$MGMT_IF \\
  -e GLOO_SOCKET_IFNAME=$MGMT_IF$(if [[ -n "$IB_HCA" ]]; then echo " \\
  -e NCCL_IB_HCA=$IB_HCA"; fi)"
ENVEOF
chmod +x "$ENV_FILE"

echo "  ✓ Environment saved to: mesh-env.sh"
echo ""
echo "  To use:"
echo ""
echo "    source mesh-env.sh"
echo ""
echo "    ./launch-cluster-override.sh \\"
echo "      --nodes \"$NODES_ARG\" \\"
echo "      --apply-mod mods/ray-v2-executor-mod \\"
echo "      exec vllm serve /model_data \\"
echo "      --port 8000 --host 0.0.0.0 \\"
echo "      --tensor-parallel-size 1 \\"
echo "      --pipeline-parallel-size ${#ALL_NODES[@]} \\"
echo "      --distributed-executor-backend ray \\"
echo "      --gpu-memory-utilization 0.88 \\"
echo "      --max-model-len 65536 \\"
echo "      --max-num-batched-tokens 8192 \\"
echo "      --max-num-seqs 32 \\"
echo "      --trust-remote-code \\"
echo "      --dtype bfloat16 \\"
echo "      --kv-cache-dtype fp8 \\"
echo "      --attention-backend flashinfer \\"
echo "      --compilation-config.cudagraph_mode none"
echo ""

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║              Mesh Setup Complete                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
