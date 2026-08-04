#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Run the AuroraRL full-feature sparse-delta experiment inside one GPU container.

Required runtime:
  docker --gpus all with enough visible GPUs for inference + trainer

Environment variables:
  WANDB_API_KEY       Required for online W&B logging
  MODEL_NAME          HF model ID or container-local path (default: Qwen/Qwen3-4B)
  INFERENCE_GPU_IDS   Comma-separated inference GPUs (default: 0,1)
  INFERENCE_BASE_PORT First inference HTTP port (default: 8401)
  INFERENCE_RPC_BASE_PORT
                      First inference data-parallel RPC port (default: 13701)
  TRAINER_GPU_IDS     Comma-separated trainer GPUs (default: 2)
  MAX_STEPS           RL steps (default: 800)
  WANDB_PROJECT       W&B project (default: aurorarl)
  WANDB_RUN_NAME      W&B run name (default: generated from RUN_ID)
  RUN_ID              Unique run identifier (default: UTC timestamp)
  OUTPUT_DIR          RL output path (default: /dev/shm/<RUN_ID>/rl)
  LOG_DIR             Persistent launcher logs (default: /outputs/<RUN_ID>/logs)
  LOG_LEVEL           Prime-RL log level (default: info)
  CLEAN_OUTPUT_DIR    Whether RL cleans OUTPUT_DIR (default: true)

Any arguments after -- are appended to the `uv run rl` command.
EOF
  exit 0
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
if [[ -z "${INFERENCE_GPU_IDS:-}" ]]; then
  legacy_inference_a_gpu_id="${INFERENCE_A_GPU_ID:-${SEED_GPU_ID:-}}"
  legacy_inference_b_gpu_id="${INFERENCE_B_GPU_ID:-${PEER_GPU_ID:-}}"
  if [[ -n "$legacy_inference_a_gpu_id" || -n "$legacy_inference_b_gpu_id" ]]; then
    INFERENCE_GPU_IDS="${legacy_inference_a_gpu_id:-0},${legacy_inference_b_gpu_id:-1}"
  else
    INFERENCE_GPU_IDS="0,1"
  fi
fi
INFERENCE_BASE_PORT="${INFERENCE_BASE_PORT:-8401}"
INFERENCE_RPC_BASE_PORT="${INFERENCE_RPC_BASE_PORT:-13701}"
TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-2}"
MAX_STEPS="${MAX_STEPS:-800}"
RUN_ID="${RUN_ID:-aurorarl-$(date -u +%Y%m%d-%H%M%S)}"
WORK_ROOT="${WORK_ROOT:-/dev/shm/$RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK_ROOT/rl}"
STAGING_ROOT="${STAGING_ROOT:-$WORK_ROOT/staging}"
LOG_DIR="${LOG_DIR:-/outputs/$RUN_ID/logs}"
WANDB_PROJECT="${WANDB_PROJECT:-aurorarl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_ID}"
WANDB_MODE="${WANDB_MODE:-online}"
LOG_LEVEL="${LOG_LEVEL:-info}"
CLEAN_OUTPUT_DIR="${CLEAN_OUTPUT_DIR:-true}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-1800}"

export HF_HOME="${HF_HOME:-/cache/hf}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache/vllm}"
export WANDB_DIR="${WANDB_DIR:-/cache/wandb}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export WANDB_MODE

for command in uv curl nvidia-smi; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    exit 2
  fi
done

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is required when WANDB_MODE=online" >&2
  exit 2
fi

if ! [[ "$INFERENCE_BASE_PORT" =~ ^[0-9]+$ && "$INFERENCE_RPC_BASE_PORT" =~ ^[0-9]+$ ]]; then
  echo "INFERENCE_BASE_PORT and INFERENCE_RPC_BASE_PORT must be numeric ports" >&2
  exit 2
fi

IFS=',' read -r -a inference_gpu_ids <<< "$INFERENCE_GPU_IDS"
if [[ "${#inference_gpu_ids[@]}" -eq 0 ]]; then
  echo "INFERENCE_GPU_IDS must contain at least one GPU index" >&2
  exit 2
fi
for i in "${!inference_gpu_ids[@]}"; do
  inference_gpu_ids[$i]="${inference_gpu_ids[$i]//[[:space:]]/}"
done

IFS=',' read -r -a trainer_gpu_ids <<< "$TRAINER_GPU_IDS"
if [[ "${#trainer_gpu_ids[@]}" -eq 0 ]]; then
  echo "TRAINER_GPU_IDS must contain at least one GPU index" >&2
  exit 2
fi

declare -A selected_gpus=()
for i in "${!trainer_gpu_ids[@]}"; do
  trainer_gpu_ids[$i]="${trainer_gpu_ids[$i]//[[:space:]]/}"
done

if (( INFERENCE_BASE_PORT < 1 || INFERENCE_BASE_PORT + ${#inference_gpu_ids[@]} - 1 > 65535 )); then
  echo "Inference HTTP port range is outside 1-65535" >&2
  exit 2
fi
if (( INFERENCE_RPC_BASE_PORT < 1 || INFERENCE_RPC_BASE_PORT + ${#inference_gpu_ids[@]} - 1 > 65535 )); then
  echo "Inference RPC port range is outside 1-65535" >&2
  exit 2
fi

for gpu_id in "${inference_gpu_ids[@]}" "${trainer_gpu_ids[@]}"; do
  if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index: $gpu_id" >&2
    exit 2
  fi
  if [[ -n "${selected_gpus[$gpu_id]:-}" ]]; then
    echo "GPU $gpu_id is assigned to more than one role" >&2
    exit 2
  fi
  selected_gpus[$gpu_id]=1
done

visible_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
for gpu_id in "${!selected_gpus[@]}"; do
  if (( gpu_id >= visible_gpu_count )); then
    echo "GPU $gpu_id is not available; container exposes $visible_gpu_count GPU(s)" >&2
    exit 2
  fi
done

trainer_gpu_ids_csv="$(IFS=,; echo "${trainer_gpu_ids[*]}")"
trainer_gpu_count="${#trainer_gpu_ids[@]}"
inference_gpu_ids_csv="$(IFS=,; echo "${inference_gpu_ids[*]}")"

mkdir -p "$LOG_DIR" "$STAGING_ROOT" "$HF_HOME" "$VLLM_CACHE_ROOT" "$WANDB_DIR"
for idx in "${!inference_gpu_ids[@]}"; do
  mkdir -p "$STAGING_ROOT/inference-$idx"
done
printf 'starting\n' > "$LOG_DIR/status"

inference_pids=()

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  for pid in "${inference_pids[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${inference_pids[@]}"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ "$exit_code" -eq 0 ]]; then
    printf 'completed\n' > "$LOG_DIR/status"
  elif [[ "$(cat "$LOG_DIR/status" 2>/dev/null || true)" != failed* ]]; then
    printf 'failed exit_code=%s\n' "$exit_code" > "$LOG_DIR/status"
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "[aurorarl] run_id=$RUN_ID model=$MODEL_NAME"
echo "[aurorarl] inference_gpus=$inference_gpu_ids_csv trainer_gpus=$trainer_gpu_ids_csv"
echo "[aurorarl] output_dir=$OUTPUT_DIR log_dir=$LOG_DIR"

base_urls=()
admin_urls=()

for idx in "${!inference_gpu_ids[@]}"; do
  gpu_id="${inference_gpu_ids[$idx]}"
  port=$((INFERENCE_BASE_PORT + idx))
  rpc_port=$((INFERENCE_RPC_BASE_PORT + idx))
  name="inference-$idx"
  staging_dir="$STAGING_ROOT/$name"
  log_path="$LOG_DIR/$name.log"

  base_urls+=("http://127.0.0.1:$port/v1")
  admin_urls+=("http://127.0.0.1:$port")

  echo "[aurorarl] starting $name gpu=$gpu_id port=$port rpc_port=$rpc_port"
  env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    uv run inference \
      @ configs/docker/aurorarl_inference.toml \
      --model.name "$MODEL_NAME" \
      --server.port "$port" \
      --data-parallel-rpc-port "$rpc_port" \
      --server.staging-dir "$staging_dir" \
      > "$log_path" 2>&1 &
  inference_pids+=("$!")
  echo "${inference_pids[$idx]}" > "$LOG_DIR/$name.pid"
done

wait_for_health() {
  local name=$1
  local pid=$2
  local url=$3
  local log_path=$4
  local elapsed=0

  while (( elapsed < HEALTH_TIMEOUT_S )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'failed %s exited before health check\n' "$name" > "$LOG_DIR/status"
      tail -n 100 "$log_path" >&2 || true
      return 1
    fi
    if curl -fsS "$url/health" >/dev/null 2>&1; then
      echo "[aurorarl] $name is healthy"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done

  printf 'failed %s health check timeout\n' "$name" > "$LOG_DIR/status"
  return 1
}

for idx in "${!inference_gpu_ids[@]}"; do
  wait_for_health "inference $idx" "${inference_pids[$idx]}" "${admin_urls[$idx]}" "$LOG_DIR/inference-$idx.log"
done

printf 'training\n' > "$LOG_DIR/status"

rl_command=(
  uv run rl
  @ configs/docker/aurorarl_rl.toml
  --model.name "$MODEL_NAME"
  --output-dir "$OUTPUT_DIR"
  --max-steps "$MAX_STEPS"
  --clean-output-dir "$CLEAN_OUTPUT_DIR"
  --deployment.num-train-gpus "$trainer_gpu_count"
  --deployment.gpus-per-node "$trainer_gpu_count"
  --orchestrator.student.client.base-url "${base_urls[@]}"
  --orchestrator.student.client.admin-base-url "${admin_urls[@]}"
  --wandb.project "$WANDB_PROJECT"
  --wandb.name "$WANDB_RUN_NAME"
  --log.level "$LOG_LEVEL"
)
rl_command+=("$@")

printf '[aurorarl] rl command:'
printf ' %q' "${rl_command[@]}"
printf '\n'

set +e
env CUDA_VISIBLE_DEVICES="$trainer_gpu_ids_csv" "${rl_command[@]}" 2>&1 | tee "$LOG_DIR/rl.log"
rl_exit_code=${PIPESTATUS[0]}
set -e

if [[ "$rl_exit_code" -ne 0 ]]; then
  printf 'failed rl exit_code=%s\n' "$rl_exit_code" > "$LOG_DIR/status"
  exit "$rl_exit_code"
fi

exit 0
