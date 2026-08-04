# AuroraRL Container Handoff

This container packages the current upstream-based sparse-delta implementation
and a one-command, single-host multi-GPU launcher.

It replaces the old reference repo's `Dockerfile.reward` approach. The image is
built from the maintained `Dockerfile.cuda`, so its CUDA, vLLM, Torch, UCX, and
Prime-RL dependencies stay aligned with the current upstream code.

## Included workflow

One container starts:

```text
GPU 0..K    inference endpoints
GPU K+1..N  trainer
```

The RL configuration enables:

- sparse delta checkpoints
- HTTP streaming stage upload
- background stage and ordered commit
- direct stage/commit to all inference endpoints
- online W&B logging

The default experiment uses Qwen3-4B on the DeepScaleR Preview math dataset,
with learning rate 1e-6, batch size 512, group size 8, sequence length 2048,
and 800 steps. These values match the old reference math run where applicable.

Intentional differences from the old reference run:

- sparse delta replaces full-weight updates
- multiple inference endpoints replace the single endpoint
- HTTP streaming stage/commit remains enabled while relay remains disabled
- the current RL trainer uses token packing and has no equivalent
  `micro_batch_size` option
- the removed `async_level` option is represented operationally by
  `max_inflight_rollouts = 512`

## Build

Initialize the public submodules before building:

```bash
git submodule update --init --recursive -- \
  deps/pydantic-config \
  deps/verifiers \
  deps/renderers \
  deps/research-environments
```

Build the current branch:

```bash
docker build \
  -f Dockerfile.cuda \
  -t gengluo/aurorarl:v3 \
  .
```

The image does not include model weights. Model and kernel caches are stored in
a Docker volume at runtime.

The build installs `primeintellect/deepscaler-math@0.1.5` from the Prime
Environments Hub. The environment code is included in the image; the public
DeepScaleR dataset is downloaded from Hugging Face on first use and then reused
from the cache volume.

## GPU smoke test

```bash
docker run --rm --gpus all \
  gengluo/aurorarl:v3 \
  uv run --no-sync python -c 'import torch, vllm; print(torch.cuda.is_available(), torch.cuda.device_count(), vllm.__version__)'
```

Do not start training unless CUDA is available and the expected GPU count is
visible.

## Run detached

Create persistent volumes once:

```bash
docker volume create aurorarl-cache
docker volume create aurorarl-output
```

Start the default three-GPU run:

```bash
docker run -d \
  --name aurorarl-realistic \
  --gpus all \
  --ipc=host \
  --ulimit nofile=32000:32000 \
  -e WANDB_API_KEY \
  -e MODEL_NAME=Qwen/Qwen3-4B \
  -e INFERENCE_GPU_IDS=0,1 \
  -e TRAINER_GPU_IDS=2 \
  -e MAX_STEPS=800 \
  -e WANDB_PROJECT=aurorarl \
  -e WANDB_RUN_NAME=aurorarl-qwen3-4b-deepscaler \
  -v aurorarl-cache:/cache \
  -v aurorarl-output:/outputs \
  gengluo/aurorarl:v3 \
  scripts/run_aurorarl.sh
```

The command returns immediately. The container continues until training
finishes or fails, then shuts down all inference servers and exits.

Use more inference and trainer GPUs by assigning comma-separated lists that do
not overlap:

```bash
-e INFERENCE_GPU_IDS=0,1,2,3
-e TRAINER_GPU_IDS=4,5,6,7
```

The launcher starts one inference endpoint per `INFERENCE_GPU_IDS` entry, using
HTTP ports starting at `INFERENCE_BASE_PORT` (default `8401`) and RPC ports
starting at `INFERENCE_RPC_BASE_PORT` (default `13701`). It derives
`deployment.num_train_gpus` from `TRAINER_GPU_IDS`. The current sparse-delta
migration has been exercised with one trainer GPU; multi-GPU trainer execution
relies on upstream FSDP behavior and should be smoke-tested on the target host
before starting a long run.

For a local run without cloud logging, use `WANDB_MODE=offline`. Do not use
`WANDB_MODE=disabled` with the currently locked W&B 0.26.1: its disabled
summary object does not implement `_as_dict()`, so final summary serialization
fails after training has otherwise completed.

## Local E2E validation

The sparse-delta container path was validated on 2026-06-22 with Qwen3-0.6B,
the alphabet-sort smoke environment, and three A100 GPUs:

- four RL steps completed with container exit code 0
- both rollout endpoints served requests with zero request errors
- sparse delta versions 1 and 2 were uploaded as 11 streaming chunks each
- the relay seed staged and committed both versions to the peer
- version 1 applied 179 sparse tensors on each endpoint
- version 2 applied 180 sparse tensors on each endpoint
- all inference and trainer GPU memory was released at container exit

The retained launcher logs are under `.run/docker-e2e-offline/` in the build
workspace.

The DeepScaler environment was separately validated by importing
`deepscaler-math==0.1.5`, loading all 40,315 public DeepScaleR samples into a
`SingleTurnEnv`, and running the generated RL configuration in dry-run mode.
The earlier four-step GPU E2E used the relay-enabled v1/v2 configuration; the
v3 default intentionally disables relay and has each endpoint managed directly
by the orchestrator.

## Monitor

```bash
docker logs -f aurorarl-realistic
docker inspect -f '{{.State.Status}} {{.State.ExitCode}}' aurorarl-realistic
```

Logs are also retained in the `aurorarl-output` volume under:

```text
/outputs/<RUN_ID>/logs/status
/outputs/<RUN_ID>/logs/rl.log
/outputs/<RUN_ID>/logs/inference-0.log
/outputs/<RUN_ID>/logs/inference-1.log
```

Inspect the volume without entering the training container:

```bash
docker run --rm -v aurorarl-output:/outputs alpine \
  find /outputs -maxdepth 3 -type f -print
```

## Stop

```bash
docker stop --time 60 aurorarl-realistic
```

Docker then terminates any remaining vLLM EngineCore child processes in the
container namespace, so they cannot remain as orphan GPU processes on the host.

## Shared memory and outputs

The default live RL output and staged deltas are written under
`/dev/shm/<RUN_ID>`. `--ipc=host` is intentional because Docker's default shared
memory limit is only 64 MiB. Historical deltas are not retained after normal
cleanup, avoiding an unbounded 800-step delta chain in RAM.

To persist live RL output and checkpoints instead of only logs and W&B data,
set an output path on the mounted volume:

```bash
-e OUTPUT_DIR=/outputs/aurorarl-realistic/rl
```

This is slower than tmpfs but survives container removal.

## Export or push

Push to Docker Hub:

```bash
docker login
docker push gengluo/aurorarl:v3
```

Or create an offline handoff archive:

```bash
docker save gengluo/aurorarl:v3 | \
  gzip > aurorarl_v3.tar.gz
sha256sum aurorarl_v3.tar.gz \
  > aurorarl_v3.tar.gz.sha256
```

On the target machine:

```bash
sha256sum -c aurorarl_v3.tar.gz.sha256
docker load -i aurorarl_v3.tar.gz
```

## True multi-host deployment

This launcher is deliberately a single-container, single-host handoff. It uses
localhost URLs between the orchestrator and all inference endpoints.

A real cross-region deployment should run the same image with separate
inference/RL roles, externally reachable admin URLs, and authentication/TLS.
Do not expose the current unauthenticated admin routes directly to the public
internet.
