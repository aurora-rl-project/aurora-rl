#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/scratch/luogeng/models/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/luogeng/tmp/prime-rl-delta-relay-e2e}"
SEED_A_GPU="${SEED_A_GPU:-0}"
PEER_A_GPU="${PEER_A_GPU:-1}"
SEED_B_GPU="${SEED_B_GPU:-2}"
PEER_B_GPU="${PEER_B_GPU:-3}"
TRAIN_GPU="${TRAIN_GPU:-3}"
SEED_A_PORT=8101
PEER_A_PORT=8102
SEED_B_PORT=8111
PEER_B_PORT=8112
SEED_A_RPC_PORT=13401
PEER_A_RPC_PORT=13402
SEED_B_RPC_PORT=13411
PEER_B_RPC_PORT=13412
UV_CACHE_DIR="${UV_CACHE_DIR:-.venv/uv-cache}"
HF_HOME="${HF_HOME:-/scratch/luogeng/hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

LOG_DIR="$OUTPUT_DIR/logs/e2e_harness"
pids=()

cleanup() {
    for pid in "${pids[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local name="$3"
    local attempts="${4:-600}"
    for _ in $(seq 1 "$attempts"); do
        if timeout 1 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
            echo "$name is listening on $host:$port"
            return 0
        fi
        sleep 1
    done
    echo "Timed out waiting for $name on $host:$port" >&2
    return 1
}

start_inference() {
    local name="$1"
    local gpu="$2"
    local port="$3"
    local rpc_port="$4"
    local staging_dir="$5"
    local relay_peer_url="${6:-}"
    local relay_args=(--relay.enabled false)
    if [[ -n "$relay_peer_url" ]]; then
        relay_args=(--relay.enabled true --relay.peers "[\"$relay_peer_url\"]")
    fi

    echo "Starting $name on GPU $gpu, port $port"
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        HF_HOME="$HF_HOME" \
        TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
        uv --cache-dir "$UV_CACHE_DIR" run --frozen inference \
            @ configs/debug/inference_delta_relay_e2e.toml \
            --model.name "$MODEL" \
            --server.port "$port" \
            --data-parallel-rpc-port "$rpc_port" \
            --server.staging-dir "$staging_dir" \
            "${relay_args[@]}" \
            > "$LOG_DIR/$name.log" 2>&1 &
    pids+=("$!")
}

trap cleanup EXIT

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

start_inference "seed-a" "$SEED_A_GPU" "$SEED_A_PORT" "$SEED_A_RPC_PORT" "$OUTPUT_DIR/staging/seed-a" "http://127.0.0.1:$PEER_A_PORT"
start_inference "peer-a" "$PEER_A_GPU" "$PEER_A_PORT" "$PEER_A_RPC_PORT" "$OUTPUT_DIR/staging/peer-a"
start_inference "seed-b" "$SEED_B_GPU" "$SEED_B_PORT" "$SEED_B_RPC_PORT" "$OUTPUT_DIR/staging/seed-b" "http://127.0.0.1:$PEER_B_PORT"
start_inference "peer-b" "$PEER_B_GPU" "$PEER_B_PORT" "$PEER_B_RPC_PORT" "$OUTPUT_DIR/staging/peer-b"

wait_for_port 127.0.0.1 "$SEED_A_PORT" "seed-a"
wait_for_port 127.0.0.1 "$PEER_A_PORT" "peer-a"
wait_for_port 127.0.0.1 "$SEED_B_PORT" "seed-b"
wait_for_port 127.0.0.1 "$PEER_B_PORT" "peer-b"

echo "Running relay E2E on trainer GPU $TRAIN_GPU"
env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
    WANDB_MODE=disabled \
    HF_HOME="$HF_HOME" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    uv --cache-dir "$UV_CACHE_DIR" run --frozen rl \
        @ configs/debug/rl_delta_relay_e2e.toml \
        --model.name "$MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --clean-output-dir false \
        --log.level debug

ORCH_LOG="$OUTPUT_DIR/logs/orchestrator.log"

grep -q "base_url=http://127.0.0.1:8101/v1, http://127.0.0.1:8102/v1, http://127.0.0.1:8111/v1, http://127.0.0.1:8112/v1" "$ORCH_LOG"
grep -q "\\[relay\\]\\[stage_stream_init\\].*8102" "$LOG_DIR/seed-a.log"
grep -q "\\[relay\\]\\[stage_stream_init\\].*8112" "$LOG_DIR/seed-b.log"
grep -q "Staged delta weights version 1" "$LOG_DIR/peer-a.log"
grep -q "Staged delta weights version 1" "$LOG_DIR/peer-b.log"
grep -q "Committed delta weights version 1" "$LOG_DIR/peer-a.log"
grep -q "Committed delta weights version 1" "$LOG_DIR/peer-b.log"

echo "Delta relay E2E passed."
echo "Logs: $OUTPUT_DIR/logs"
