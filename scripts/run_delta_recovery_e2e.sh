#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/scratch/luogeng/models/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/prime-rl-delta-recovery-e2e}"
INFER_A_GPU="${INFER_A_GPU:-1}"
INFER_B_GPU="${INFER_B_GPU:-2}"
TRAIN_GPU="${TRAIN_GPU:-3}"
PORT_A="${PORT_A:-8001}"
PORT_B="${PORT_B:-8002}"
PROXY_PORT="${PROXY_PORT:-8012}"
FAIL_PATH="${FAIL_PATH:-/stage_stream_init}"
UV_CACHE_DIR="${UV_CACHE_DIR:-.venv/uv-cache}"
HF_HOME="${HF_HOME:-/scratch/luogeng/hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

LOG_DIR="$OUTPUT_DIR/logs/e2e_harness"
mkdir -p "$LOG_DIR"

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

trap cleanup EXIT

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "Starting inference A on GPU $INFER_A_GPU, port $PORT_A"
env \
    CUDA_VISIBLE_DEVICES="$INFER_A_GPU" \
    HF_HOME="$HF_HOME" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    uv --cache-dir "$UV_CACHE_DIR" run --frozen inference \
        @ configs/debug/inference_delta_recovery_e2e.toml \
        --model.name "$MODEL" \
        --server.port "$PORT_A" \
        --server.staging-dir "$OUTPUT_DIR/staging/inference-a" \
        > "$LOG_DIR/inference-a.log" 2>&1 &
pids+=("$!")

echo "Starting inference B on GPU $INFER_B_GPU, port $PORT_B"
env \
    CUDA_VISIBLE_DEVICES="$INFER_B_GPU" \
    HF_HOME="$HF_HOME" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    uv --cache-dir "$UV_CACHE_DIR" run --frozen inference \
        @ configs/debug/inference_delta_recovery_e2e.toml \
        --model.name "$MODEL" \
        --server.port "$PORT_B" \
        --server.staging-dir "$OUTPUT_DIR/staging/inference-b" \
        > "$LOG_DIR/inference-b.log" 2>&1 &
pids+=("$!")

wait_for_port 127.0.0.1 "$PORT_A" "inference A"
wait_for_port 127.0.0.1 "$PORT_B" "inference B"

echo "Starting fault proxy on port $PROXY_PORT -> inference B, failing first POST $FAIL_PATH"
uv --cache-dir "$UV_CACHE_DIR" run --frozen scripts/fault_injection_proxy.py \
    --listen-host 127.0.0.1 \
    --listen-port "$PROXY_PORT" \
    --target-base-url "http://127.0.0.1:$PORT_B" \
    --fail-method POST \
    --fail-path "$FAIL_PATH" \
    --fail-count 1 \
    > "$LOG_DIR/fault-proxy.log" 2>&1 &
pids+=("$!")

wait_for_port 127.0.0.1 "$PROXY_PORT" "fault proxy"

echo "Running RL recovery E2E on trainer GPU $TRAIN_GPU"
env \
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" \
    WANDB_MODE=disabled \
    HF_HOME="$HF_HOME" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    uv --cache-dir "$UV_CACHE_DIR" run --frozen rl \
        @ configs/debug/rl_delta_recovery_e2e.toml \
        --model.name "$MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --clean-output-dir false \
        --log.level debug

ORCH_LOG="$OUTPUT_DIR/logs/orchestrator.log"
PROXY_LOG="$LOG_DIR/fault-proxy.log"

grep -q "injecting .* POST $FAIL_PATH" "$PROXY_LOG"
grep -q "endpoint state healthy -> retired" "$ORCH_LOG"
grep -q "endpoint state retired -> recovering" "$ORCH_LOG"
grep -q "replaying delta version" "$ORCH_LOG"
grep -q "endpoint state recovering -> healthy" "$ORCH_LOG"

echo "Delta recovery E2E passed."
echo "Logs: $OUTPUT_DIR/logs"
