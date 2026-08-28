# AuroraRL

AuroraRL supports RL deployments where trainers and inference servers may be
separated by slow or long-distance networks. It adds sparse delta weight
updates, HTTP streaming stage/commit, multi-endpoint inference, load-aware
routing, endpoint recovery, and regional relay fan-out.

The full version will be released after the camera-ready.

## Features

- Exact sparse deltas with compact index encoding.
- Streaming extraction and background HTTP upload.
- Multiple inference endpoints for rollout routing and weight-update fan-out.
- Optional throughput-aware load balancing.
- Optional endpoint quarantine and delta-chain recovery.
- Optional relay seeds for region-local weight distribution.
- Single-host development and cross-region trainer/inference deployment.

## Installation

AuroraRL requires Linux, Python 3.12, NVIDIA GPUs, and
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/aurora-rl-project/aurora-rl.git
cd aurora-rl
```

Initialize the required submodules:

```bash
git submodule update --init -- deps/verifiers deps/renderers deps/research-environments deps/pydantic-config
```

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Install dependencies from the lock file:

```bash
uv sync --all-extras
```

Verify the environment:

```bash
uv run python -V
uv run rl @ configs/aurorarl/gsm8k.toml --dry-run --output-dir /tmp/aurorarl-dry-run
```

## Quick Start: GSM8K

The smallest standard example trains `Qwen/Qwen3-0.6B` on
[`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k). It uses one
inference GPU, one trainer GPU, and runs for 100 steps. Sparse delta,
HTTP streaming upload, background staging, and stage/commit are enabled.

Set a W&B key for online logging:

```bash
export WANDB_API_KEY="<your-key>"
```

Start the complete local run:

```bash
uv run rl @ configs/aurorarl/gsm8k.toml \
  --output-dir outputs/gsm8k
```

The `rl` entrypoint starts inference, the orchestrator, and the trainer from
the same config. Use `WANDB_MODE=offline` if online logging is not needed.

## Configuration

Configs are TOML files loaded with `@`. CLI values override TOML values:

```bash
uv run rl @ configs/aurorarl/gsm8k.toml --max-steps 10
uv run inference @ configs/aurorarl/inference.toml --server.port 8501
```

Direct profiles:

- `configs/aurorarl/gsm8k.toml`: two-GPU sparse-delta Quick Start.
- `configs/aurorarl/rl.toml`: external multi-endpoint RL with load balancing
  and lease recovery.
- `configs/aurorarl/inference.toml`: first standalone inference endpoint.
- `configs/aurorarl/inference_b.toml`: second standalone inference endpoint.

Relay profiles:

- `configs/aurorarl/relay_rl.toml`: rollout traffic to seed and peer, weight
  updates to the seed.
- `configs/aurorarl/inference_relay_seed.toml`: relay seed and its peer list.
- `configs/aurorarl/inference_relay_peer.toml`: inference peer.

### Sparse Delta Transfer

```toml
[weight_broadcast]
type = "filesystem"
mode = "delta"
update_protocol = "stage_commit"
stage_transport = "streaming_upload"
background_stage = true
retain_all_deltas = true
delta_stream_group_size = 4
```

`type = "filesystem"` selects file-based weight artifacts; it does not require
a shared filesystem. With `streaming_upload`, the orchestrator uploads the
trainer-local delta over HTTP to each inference server's staging directory.
It also enables trainer-side streaming extraction: tensor records are appended
to `delta.stream` and uploaded while later layers are still being scanned.
`delta_stream_group_size` controls how many transformer layers are processed
between file flushes. Other stage transports continue to use standard
`delta.safetensors` files.

### Multi-endpoint Routing

```toml
[orchestrator]
enable_load_balancing = true
throughput_ema_beta = 0.8

[orchestrator.student.client]
base_url = ["http://inference-a:8401/v1", "http://inference-b:8401/v1"]
```

Rollout requests are routed across `base_url`. The same endpoints receive
health checks and weight stage/commit. Load balancing uses observed throughput
EMA.

### Lease Recovery

```toml
[weight_broadcast]
retain_all_deltas = true

[orchestrator.student.client]
lease_enabled = true
lease_cooldown_s = 20.0
lease_recovery_enabled = true
lease_recovery_poll_interval_s = 5.0
```

Failed endpoints are removed from routing. After recovery, AuroraRL reloads
the base weights and replays the retained delta chain to the active version.

### Regional Relay

The relay profile explicitly separates rollout endpoints from the weight-update
target:

```toml
[orchestrator.student.client]
base_url = ["http://127.0.0.1:8401/v1", "http://127.0.0.1:8402/v1"]
admin_base_url = ["http://127.0.0.1:8401"]
```

The inference process using `inference_relay_seed.toml` is the relay seed. Its
`peers` list identifies the inference servers that receive forwarded stages
and commits:

```toml
[relay]
enabled = true
peers = ["http://127.0.0.1:8402"]
fail_on_peer_error = true
```

For a local three-GPU relay run, start these in separate terminals:

```bash
CUDA_VISIBLE_DEVICES=0 uv run inference @ configs/aurorarl/inference_relay_seed.toml
CUDA_VISIBLE_DEVICES=1 uv run inference @ configs/aurorarl/inference_relay_peer.toml
CUDA_VISIBLE_DEVICES=2 uv run rl @ configs/aurorarl/relay_rl.toml
```

## Separate Trainer and Inference

For a local three-GPU run, start the two inference endpoints and trainer in
separate terminals:

```bash
CUDA_VISIBLE_DEVICES=0 \
uv run inference @ configs/aurorarl/inference.toml
```

```bash
CUDA_VISIBLE_DEVICES=1 \
uv run inference @ configs/aurorarl/inference_b.toml
```

```bash
CUDA_VISIBLE_DEVICES=2 \
uv run rl @ configs/aurorarl/rl.toml
```

The default endpoints use `127.0.0.1`. For multi-machine deployment, replace
them in `configs/aurorarl/rl.toml` with the inference hosts' reachable
addresses.

The trainer's broadcast directory remains local. Deltas are uploaded to each
inference host over HTTP, so no NFS or cross-region shared filesystem is
required.

The weight-control routes handle staging, commit, pause/resume, and reload.
Keep them on a private network or behind an authenticated TLS proxy; do not
expose them directly to the public internet.

## Monitoring

Set `WANDB_API_KEY` for online W&B logging or `WANDB_MODE=offline` for local
runs. Useful metrics include:

- `time/stage_weights` and `time/commit_weights`
- `rollout_client/*/routing_weight` and `throughput_ema`
- `rollout_client/*/requests` and `errors`
- `inference_endpoint/*/state` and `recovery_attempts`
- `scheduler/async_level` and `inflight_rollouts`

## Acknowledgments

Reuses some training and inference code from [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl), including PyTorch FSDP2 training and vLLM inference components.

## Citation

```bibtex
@article{aurorrl2026,
  title   = {RL over Commodity Networks: Overcoming the
             Bandwidth Barrier with Lossless Sparse Deltas},
  author  = {Ruan, Chaoyi and Luo, Geng and Wan, Xinyi and Zhao, Long
             and Wang, Qinghe and Zhu, Jiaan and Xu, Duling and Xu, Guanbin
             and Wei, Dehui and Liu, Xiang and Li, Cheng and Sun, Haifeng
             and Miao, Congcong and Li, Jialin},
  journal = {arXiv preprint arXiv:2602.11456},
  year    = {2026}
}
```

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE).
