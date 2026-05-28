import asyncio
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import uvloop
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State, UploadFile
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import init_app_state
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.serve.lora.protocol import LoadLoRAAdapterRequest
from vllm.logger import init_logger
from vllm.utils.argparse_utils import FlexibleArgumentParser

from prime_rl.configs.inference import InferenceConfig
from prime_rl.utils.logger import get_logger

logger = get_logger()
from prime_rl.inference.patches import (
    monkey_patch_harmony_stop_token_propagation,
    monkey_patch_load_lora_adapter,
    monkey_patch_tokenize_params_validation,
    monkey_patch_vllm_padded_input_scrub,
)

# NOTE: Fix harmony stop token propagation for GPT-OSS models
# Upstream issue still open: https://github.com/vllm-project/vllm/issues/22519
monkey_patch_harmony_stop_token_propagation()
# NOTE: Monkeypatch LoadLoRAAdapter to allow loading the same adapter multiple times
# May be removable if we pass load_inplace=True (supported since vLLM 0.18, PR #31326)
monkey_patch_load_lora_adapter()
# NOTE: Monkeypatch TokenizeParams to fix overly conservative validation
# Still needed in vLLM 0.20 — upstream rejects prompt_len > max_model_len - max_tokens
monkey_patch_tokenize_params_validation()
# NOTE: Optional mitigation for vLLM padded decode inputs until the native fix
# is available in our pinned runtime.
monkey_patch_vllm_padded_input_scrub()

logger = init_logger("vllm.entrypoints.openai.api_server")

# Create our own router for custom endpoints
router = APIRouter()
WEIGHT_UPDATE_MODES = {"full", "delta"}
STAGE_READ_CHUNK_BYTES = 16 * 1024 * 1024


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


def models(request: Request) -> OpenAIServingModels:
    return request.app.state.openai_serving_models


def _ensure_weight_staging_state(state: State) -> None:
    if not hasattr(state, "staged_versions"):
        state.staged_versions = {}
    if not hasattr(state, "active_version"):
        state.active_version = None
    if not hasattr(state, "staging_dir"):
        state.staging_dir = Path("staging")
    state.staging_dir.mkdir(parents=True, exist_ok=True)


def _safe_path_component(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return safe or "unknown"


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


async def _read_request_fields(request: Request) -> tuple[dict[str, Any], UploadFile | None]:
    fields: dict[str, Any] = dict(request.query_params)
    uploaded_file = None
    content_type = request.headers.get("content-type", "")

    if content_type.startswith("application/json"):
        data = await request.json()
        if isinstance(data, dict):
            fields.update({key: value for key, value in data.items() if value is not None})
    elif content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        form = await request.form()
        for key, value in form.items():
            if isinstance(value, UploadFile):
                if key == "file":
                    uploaded_file = value
                continue
            fields[key] = value

    return fields, uploaded_file


def _cleanup_staged_file(state: State, entry: dict[str, Any]) -> None:
    if not entry.get("owned"):
        return
    path = Path(entry["path"])
    staging_dir = Path(state.staging_dir)
    if path.is_file() and path.parent == staging_dir:
        path.unlink()


WORKER_EXTENSION_CLS = {
    "nccl": "prime_rl.inference.vllm.worker.nccl.NCCLWeightUpdateWorker",
    "filesystem": "prime_rl.inference.vllm.worker.filesystem.FileSystemWeightUpdateWorker",
}


@router.post("/pause")
async def pause(request: Request):
    await engine_client(request).pause_generation(mode="keep", clear_cache=False)
    return {"status": "paused"}


@router.post("/resume")
async def resume(request: Request):
    await engine_client(request).resume_generation()
    return {"status": "resumed"}


@router.post("/update_weights")
async def update_weights(request: Request):
    data = await request.json()
    weight_dir = data.get("weight_dir")
    mode = data.get("mode", "full")
    if mode == "full":
        await engine_client(request).collective_rpc("update_weights_from_path", args=(weight_dir,))
    elif mode == "delta":
        await engine_client(request).collective_rpc("update_weights_from_delta_path", args=(weight_dir,))
    else:
        return JSONResponse({"error": f"unsupported weight update mode: {mode}"}, status_code=400)
    return {"status": "ok"}


@router.post("/stage")
async def stage_weights(request: Request):
    """Stage full or delta weights on the local inference server."""
    _ensure_weight_staging_state(request.app.state)
    fields, uploaded_file = await _read_request_fields(request)
    version = str(fields.get("version", "unknown"))
    mode = str(fields.get("mode", "full"))
    base_version = fields.get("base_version")
    active_version = request.app.state.active_version

    if mode not in WEIGHT_UPDATE_MODES:
        return _error_response(400, f"unsupported weight update mode: {mode}")
    if mode == "delta" and base_version and active_version and base_version != active_version:
        return _error_response(409, f"base_version mismatch: current={active_version}, got={base_version}")

    ingest_start = time.perf_counter()
    suffix = "delta" if mode == "delta" else "full"
    if uploaded_file is not None:
        filename = _safe_path_component(Path(uploaded_file.filename or "weights").name)
        staged_path = request.app.state.staging_dir / f"{_safe_path_component(version)}_{suffix}_{filename}"
        with staged_path.open("wb") as f:
            while chunk := await uploaded_file.read(STAGE_READ_CHUNK_BYTES):
                f.write(chunk)
        owned = True
    else:
        path = fields.get("path")
        if path is None:
            return _error_response(400, "No file uploaded or path provided")
        staged_path = Path(str(path)).expanduser()
        if not staged_path.exists():
            return _error_response(404, f"staged path does not exist: {staged_path}")
        owned = False

    request.app.state.staged_versions[version] = {"path": staged_path, "mode": mode, "owned": owned}
    ingest_ms = (time.perf_counter() - ingest_start) * 1000
    logger.info(
        "Staged %s weights version %s from %s (base_version=%s active_version=%s owned=%s ingest_ms=%.2f)",
        mode,
        version,
        staged_path.as_posix(),
        base_version,
        active_version,
        owned,
        ingest_ms,
    )
    return {"status": "ok", "version": version, "mode": mode, "path": staged_path.as_posix(), "ingest_ms": ingest_ms}


@router.post("/commit")
async def commit_weights(request: Request):
    """Commit a staged full checkpoint or sparse delta."""
    _ensure_weight_staging_state(request.app.state)
    fields, _ = await _read_request_fields(request)
    version = fields.get("version")
    if version is None:
        return _error_response(400, "version is required")
    version = str(version)

    staged_versions = request.app.state.staged_versions
    if version not in staged_versions:
        return _error_response(404, f"version {version} not found in staging")

    entry = staged_versions[version]
    mode = entry["mode"]
    requested_mode = fields.get("mode")
    if requested_mode is not None and str(requested_mode) != mode:
        return _error_response(409, f"staged mode mismatch: staged={mode}, requested={requested_mode}")

    path = Path(entry["path"])
    if mode == "full":
        await engine_client(request).collective_rpc("update_weights_from_path", args=(path.as_posix(),))
    elif mode == "delta":
        await engine_client(request).collective_rpc("update_weights_from_delta_path", args=(path.as_posix(),))
    else:
        return _error_response(400, f"unsupported weight update mode: {mode}")

    request.app.state.active_version = version
    for staged_version, old_entry in list(staged_versions.items()):
        if staged_version == version:
            continue
        _cleanup_staged_file(request.app.state, old_entry)
        staged_versions.pop(staged_version, None)

    logger.info("Committed %s weights version %s from %s", mode, version, path.as_posix())
    return {"status": "ok", "active_version": version, "mode": mode}


@router.post("/reload_weights")
async def reload_weights(request: Request):
    """Reload the base model weights and clear local staging state."""
    _ensure_weight_staging_state(request.app.state)
    await engine_client(request).collective_rpc("reload_weights")
    for entry in list(request.app.state.staged_versions.values()):
        _cleanup_staged_file(request.app.state, entry)
    request.app.state.active_version = None
    request.app.state.staged_versions.clear()
    logger.info("Reloaded base weights and cleared staged weight versions")
    return {"status": "ok"}


@router.post("/load_lora_adapter")
async def load_lora_adapter(lora_request: LoadLoRAAdapterRequest, raw_request: Request):
    """Wrapper around vLLM's /v1/load_lora_adapter."""
    handler = models(raw_request)
    response = await handler.load_lora_adapter(lora_request)
    if isinstance(response, ErrorResponse):
        return JSONResponse(content=response.model_dump(), status_code=response.error.code)
    return {"status": "ok"}


@router.get("/liveness")
async def liveness(raw_request: Request):
    """Check that the engine event loop can service a no-op worker RPC."""
    try:
        await asyncio.wait_for(
            engine_client(raw_request).collective_rpc("liveness_probe"),
            timeout=raw_request.app.state.liveness_timeout_seconds,
        )
    except asyncio.TimeoutError:
        return JSONResponse({"status": "engine_unresponsive"}, status_code=503)
    return {"status": "ok"}


@router.post("/init_broadcaster")
async def init_broadcaster(request: Request):
    data = await request.json()
    host = data.get("host")
    port = data.get("port")
    timeout = data.get("timeout")
    rank_offset = data.get("rank_offset")
    inference_world_size = data.get("inference_world_size")
    quantize_in_weight_transfer = data.get("quantize_in_weight_transfer", False)
    await engine_client(request).collective_rpc(
        "init_broadcaster",
        args=(host, port, rank_offset, inference_world_size, timeout, quantize_in_weight_transfer),
    )
    return {"status": "ok"}


async def custom_init_app_state(
    engine_client: EngineClient,
    state: State,
    args: Namespace,
    supported_tasks: tuple,
):
    """
    Modifies init_app_state:
    1. Call the original init_app_state to set up standard state, including
       vLLM 0.20's ``serving_tokens`` for ``/inference/v1/generate``.
    2. Replace ``serving_tokens`` with ``PrimeRlServingTokens`` so DP-rank
       routing and ``routed_experts`` export survive the migration off the
       legacy ``/v1/generate`` endpoint.
    """
    await init_app_state(engine_client, state, args, supported_tasks)

    state.reset_prefix_cache_after_update = getattr(args, "reset_prefix_cache_after_update", True)
    state.liveness_timeout_seconds = args.liveness_timeout_seconds
    state.staging_dir = Path(getattr(args, "staging_dir", "staging"))
    state.staged_versions = {}
    state.active_version = None
    state.staging_dir.mkdir(parents=True, exist_ok=True)

    # Swap in our ServingTokens subclass for /inference/v1/generate so the
    # X-data-parallel-rank header and routed_experts response field — both
    # used by prime-RL's renderer / router-replay paths — keep working.
    if "generate" in supported_tasks and state.serving_tokens is not None:
        from prime_rl.inference.vllm.serving_tokens import PrimeRlServingTokens

        upstream = state.serving_tokens
        prime_serving = object.__new__(PrimeRlServingTokens)
        prime_serving.__dict__.update(upstream.__dict__)
        state.serving_tokens = prime_serving


import vllm.entrypoints.openai.api_server
import vllm.v1.utils
from vllm.entrypoints.openai.api_server import build_app as _original_build_app
from vllm.v1.utils import run_api_server_worker_proc as _original_run_api_server_worker_proc


def custom_build_app(args: Namespace, supported_tasks: tuple, model_config=None):
    """
    Wrap build_app to include our custom router.
    """
    app = _original_build_app(args, supported_tasks, model_config)
    app.include_router(router)
    return app


def custom_run_api_server_worker_proc(listen_address, sock, args, client_config=None, **uvicorn_kwargs) -> None:
    """
    Re-import our module in child processes so monkey patches (custom routes,
    custom init_app_state) are applied in multi-API-server mode.
    """
    import prime_rl.inference.vllm.server  # noqa: F401

    _original_run_api_server_worker_proc(listen_address, sock, args, client_config, **uvicorn_kwargs)


vllm.entrypoints.openai.api_server.init_app_state = custom_init_app_state
vllm.entrypoints.openai.api_server.build_app = custom_build_app
vllm.v1.utils.run_api_server_worker_proc = custom_run_api_server_worker_proc


# Adapted from vllm/entrypoints/cli/serve.py
# Only difference we do some config translation (i.e. pass populated namespace
# to `parse_args`) and additional arg validation
def server(config: InferenceConfig, vllm_extra: dict[str, Any] | None = None):
    import os

    from vllm.entrypoints.cli.serve import run_headless, run_multi_api_server
    from vllm.entrypoints.openai.api_server import run_server

    # Signal worker processes to disable LoRA on MoE layers when LoRA targets don't include experts
    if config.lora_target_modules and not any("expert" in m for m in config.lora_target_modules):
        os.environ["PRIME_NO_MOE_LORA"] = "1"

    namespace = config.to_vllm()
    if vllm_extra:
        for key, value in vllm_extra.items():
            setattr(namespace, key, value)

    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=[], namespace=namespace)
    assert args is not None
    validate_parsed_serve_args(args)

    # Set the worker extension class based on the broadcast backend
    args.worker_extension_cls = WORKER_EXTENSION_CLS[config.weight_broadcast.type]

    if args.headless or args.api_server_count < 1:
        run_headless(args)
    else:
        if args.api_server_count > 1:
            run_multi_api_server(args)
        else:
            # Single API server (this process).
            uvloop.run(run_server(args))
