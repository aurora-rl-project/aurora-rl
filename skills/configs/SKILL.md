---
name: configs
description: How the prime-rl config system works — TOML files, CLI overrides, composition, and special patterns. Use when creating configs, debugging config errors, or overriding values via CLI.
---

# Configs

prime-rl uses [`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — a Pydantic-based TOML + CLI config system (no tyro). Every entrypoint accepts TOML files via `@` and CLI overrides.

## Loading and composition

```bash
uv run rl @ examples/reverse_text/rl.toml                                  # single TOML
uv run rl @ examples/reverse_text/rl.toml --max-steps 50                   # CLI override
uv run rl @ base.toml @ overlay.toml                                       # left-to-right merge
uv run rl --model @ model.toml --data @ data.toml                          # nested section files
uv run rl @ base.toml --trainer @ trainer.toml --trainer.lr 1e-3           # mixed
```

Resolution order: CLI > config files (left-to-right) > class defaults. Merging is deep — unset fields in an overlay are preserved from the base.

Naming: CLI uses kebab-case (`--model.max-model-len`); TOML uses snake_case (`max_model_len`).

## Inspect & validate

```bash
uv run rl --help                                  # all fields and defaults
uv run rl @ rl.toml --dry-run --output-dir /tmp/x # write resolved TOML to /tmp/x/configs
```

## Validators

Incompatible combinations (e.g. CP requires flash attention) must raise in a `model_validator` at resolve time, not at runtime. When renaming a field, emit a deprecation warning with a migration hint — never silently drop.

## Special syntax

**Booleans** — CLI `--flag` / `--no-flag`; TOML must be explicit (`enforce_eager = true`).

**None** — TOML has no null, use the string `"None"` (`max_model_len = "None"`); CLI: `--model.max-model-len None`.

**Lists** — TOML uses array of tables; later config files replace lists wholesale, so overlays must include the full desired list:

```toml
[[orchestrator.env]]
id = "reverse-text"
```

CLI: `--env.0.id reverse-text --env.1.id math-env`.

**Dicts** — TOML uses a section; CLI takes a JSON string: `--vllm-extra '{"key1": "value1"}'`.

**Discriminated unions** — set the `type` field to pick the variant (`[trainer.loss] type = "sft"`). Omit `type` to keep the default variant.

**`BaseModel | None` fields** — bare flag enables defaults; nested override enables and sets:

```bash
--model.compile             # enables compile with defaults
--model.compile.fullgraph   # enables and sets fullgraph=true
```

In TOML, an empty section header (`[ckpt]`) does the same.

## RL trainer token exports

For rollout debugging, enable trainer-side token export under `trainer.experimental.token_export` (or `experimental.token_export` when running the trainer entrypoint directly). It writes one JSONL record per exported sequence under `output_dir/token_exports/step_<step>/rank_<rank>.jsonl`. Each record stores aligned per-token arrays for token ids, loss mask, advantage, reward, entropy, mismatch KL, inference/trainer logprobs, importance ratios, probability deltas, and masking diagnostics. It does not decode token text in the trainer.

```toml
[trainer.experimental.token_export]
```

Leave it unset for normal training. When enabled, it exports every sequence from each exporting rank.

## Streaming sparse deltas

At the shared RL config level, `stage_transport = "streaming_upload"` automatically enables trainer-side streaming delta extraction. The trainer writes append-only records to `delta.stream`, allowing extraction, file output, and HTTP upload to overlap. `delta_stream_group_size` controls the number of transformer layers between flushes and defaults to 4.

```toml
[weight_broadcast]
type = "filesystem"
mode = "delta"
update_protocol = "stage_commit"
stage_transport = "streaming_upload"
background_stage = true
delta_stream_group_size = 4
```

When configuring the trainer entrypoint directly instead of using the shared RL config, set `weight_broadcast.delta_streaming_enabled = true` explicitly. Non-streaming extraction writes `delta.safetensors`; streaming extraction writes `delta.stream`.

Sparse updates include changed biases. Delta values normally retain the weight dtype, but tensors with subtraction/addition rounding loss use wider values to reproduce the target weights exactly. This does not change trainer optimization or reduction dtypes. Deploy the updated delta writer and inference loader together; older loaders may reject v2 artifacts, and older v1 paths downcast wider values before application. The verification script includes biases by default; use `--no-include-bias` only for older artifacts that intentionally omitted them.

Both outputs use the logical sparse-delta v2 layout: records retain Hugging Face parameter names and global shapes after the existing trainer-side state-dict gather, making the artifact independent of trainer FSDP partitioning. Inference workers project those records into local vLLM fused and TP-sharded parameters. The adapter supports unquantized dense models with Qwen3/Llama-style Hugging Face names and vLLM layouts, including replicated, row-parallel, column-parallel, vocabulary-parallel, QKV, gate/up, and grouped-query attention layouts. TP size is not fixed; the model dimensions and attention heads must satisfy vLLM's sharding constraints. Legacy v1 artifacts are still accepted. Quantized/packed weights, MoE or expert-parallel layouts, nonstandard fused-weight conventions, and pipeline-parallel loading are outside this path.

Delta inference requires PP=1, no expert parallelism, and unquantized weights. Stage/commit also requires one API server per inference endpoint, because its version and upload state live in that process. Use independent endpoints for replica scaling. LoRA and checkpoint resume without a synchronized full base are unsupported in delta mode; use full weight broadcasts when resuming.

Every staged delta must specify `base_version`, including `"0"` for a freshly loaded base model. Commit checks the base again and serializes weight changes. Repeating a committed version retries relay fan-out without applying the local delta twice. After a worker fails during application, reload the base and replay the retained chain before serving that endpoint again; retrying the failed delta alone is unsafe.

Keep `relay.fail_on_peer_error = true` (the default) when relay peers serve rollouts. A peer failure must fail the seed operation so the pool can retire the entire region until replay succeeds. Admin requests use the client `timeout` and `connect_timeout`; configure these to allow expected WAN transfer times while bounding failed requests. Lease recovery also runs when no rollout or evaluation endpoints are healthy, so an entirely quarantined pool can recover without waiting for another commit.

## Key files

- `packages/prime-rl-configs/src/prime_rl/` — config classes under `configs/`; `utils/config.py` re-exports `BaseConfig` and `cli`
- `configs/debug/` — minimal debug configs
- `configs/private/` — private configs submodule (internal)
- `examples/` — full example configs
