import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import verifiers as vf

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.client import (
    EndpointLeaseRuntime,
    EndpointOperationError,
    EndpointOperationResult,
    StaticInferencePool,
    _is_retryable_lora_error,
    load_lora_adapter,
    setup_clients,
    stage_weights,
)


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


def test_setup_clients_assigns_renderer_and_dp_rank_headers():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
        dp_rank_count=2,
        extra_headers_from_state={"X-Session-ID": "session_id"},
    )

    renderer_settings = Qwen3VLRendererConfig()
    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_config=renderer_settings,
    )

    assert [client.client_type for client in clients] == ["renderer", "renderer"]
    assert [client.renderer_config for client in clients] == [renderer_settings, renderer_settings]
    assert [client.renderer_model_name for client in clients] == [None, None]
    assert [client.api_base_url for client in clients] == ["http://worker-a:8000/v1"] * 2
    assert [client.extra_headers["X-data-parallel-rank"] for client in clients] == ["0", "1"]
    assert clients[0].extra_headers["X-Test"] == "test"
    assert clients[0].extra_headers_from_state == {"X-Session-ID": "session_id"}


def test_setup_clients_assigns_renderer_model_name():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_config=Qwen3VLRendererConfig(),
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    assert clients[0].renderer_model_name == "Qwen/Qwen3-VL-4B-Instruct"


def test_setup_clients_preserves_chat_client_defaults():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(client_config)

    assert clients == [
        vf.ClientConfig(
            client_idx=0,
            client_type="openai_chat_completions",
            api_key_var="PRIME_API_KEY",
            api_base_url="http://worker-a:8000/v1",
            timeout=client_config.timeout,
            connect_timeout=client_config.connect_timeout,
            max_connections=8192,
            max_keepalive_connections=8192,
            max_retries=10,
            extra_headers={},
            extra_headers_from_state={},
        )
    ]


def test_setup_clients_expands_multiple_base_urls_and_dp_ranks():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1", "http://worker-b:8000/v1"],
        api_key_var="PRIME_API_KEY",
        dp_rank_count=2,
    )

    clients = setup_clients(client_config)

    assert [client.client_idx for client in clients] == [0, 1, 2, 3]
    assert [client.api_base_url for client in clients] == [
        "http://worker-a:8000/v1",
        "http://worker-a:8000/v1",
        "http://worker-b:8000/v1",
        "http://worker-b:8000/v1",
    ]
    assert [client.extra_headers["X-data-parallel-rank"] for client in clients] == ["0", "1", "0", "1"]


def test_stage_weights_reports_partial_endpoint_failure(tmp_path):
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()

    async def run() -> None:
        async def ok_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/stage"
            return httpx.Response(200, json={"status": "ok"})

        async def fail_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/stage"
            return httpx.Response(500, json={"error": "failed"})

        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(ok_handler), base_url="http://worker-a") as ok_client,
            httpx.AsyncClient(transport=httpx.MockTransport(fail_handler), base_url="http://worker-b") as fail_client,
        ):
            await stage_weights([ok_client, fail_client], delta_dir, version="1", mode="delta")

    operation = None
    try:
        asyncio.run(run())
    except EndpointOperationError as exc:
        operation = exc.operation
        results = exc.results
    else:
        raise AssertionError("stage_weights should fail when one endpoint fails")

    assert operation == "stage_weights"
    assert [(result.endpoint, result.ok) for result in results] == [
        ("http://worker-a", True),
        ("http://worker-b", False),
    ]
    assert "HTTPStatusError" in (results[1].error or "")


def test_static_pool_quarantines_rollout_client_and_skips_endpoint():
    pool = StaticInferencePool(
        ClientConfig(
            base_url=["http://worker-a/v1", "http://worker-b/v1"],
            lease_enabled=True,
            lease_cooldown_s=0.0,
        ),
        model_name="test-model",
    )

    try:
        pool.quarantine_client(pool.train_clients[0], reason="connection failed")

        assert [client.api_base_url for client in pool.train_clients] == ["http://worker-b/v1"]
        assert [client.api_base_url for client in pool.eval_clients] == ["http://worker-b/v1"]

        selected_admin_clients = pool._healthy_admin_clients("stage_weights")
        assert [str(client.base_url) for client in selected_admin_clients] == ["http://worker-b"]
        assert pool._endpoint_runtime["http://worker-a"].state == "retired"
    finally:
        for admin_client in pool.admin_clients:
            asyncio.run(admin_client.aclose())


def test_static_pool_maps_relay_peer_to_admin_seed_for_lease():
    pool = StaticInferencePool(
        ClientConfig(
            base_url=["http://seed/v1", "http://peer/v1"],
            admin_base_url=["http://seed"],
            lease_enabled=True,
            lease_cooldown_s=0.0,
        ),
        model_name="test-model",
    )

    try:
        peer_client = pool.train_clients[1]
        pool.quarantine_client(peer_client, reason="peer rollout failed")

        assert pool.train_clients == []
        assert pool.eval_clients == []
        assert pool._endpoint_runtime["http://seed"].state == "quarantine"
    finally:
        for admin_client in pool.admin_clients:
            asyncio.run(admin_client.aclose())


def test_static_pool_groups_relay_rollout_urls_across_admin_seeds():
    pool = StaticInferencePool(
        ClientConfig(
            base_url=[
                "http://seed-a/v1",
                "http://peer-a/v1",
                "http://seed-b/v1",
                "http://peer-b/v1",
            ],
            admin_base_url=["http://seed-a", "http://seed-b"],
            lease_enabled=True,
            lease_cooldown_s=0.0,
        ),
        model_name="test-model",
    )

    try:
        pool.quarantine_client(pool.train_clients[3], reason="peer-b rollout failed")

        assert [client.api_base_url for client in pool.train_clients] == [
            "http://seed-a/v1",
            "http://peer-a/v1",
        ]
        assert pool._endpoint_runtime["http://seed-a"].state == "healthy"
        assert pool._endpoint_runtime["http://seed-b"].state == "quarantine"
    finally:
        for admin_client in pool.admin_clients:
            asyncio.run(admin_client.aclose())


def test_static_pool_recovers_retired_endpoint_by_replaying_delta(tmp_path):
    delta_dir = tmp_path / "step_1"
    delta_dir.mkdir()
    (delta_dir / "delta.safetensors").write_bytes(b"delta")
    (delta_dir / "STABLE").touch()

    calls: dict[str, list[str]] = {"worker-a": [], "worker-b": []}

    def handler(worker: str):
        async def _handle(request: httpx.Request) -> httpx.Response:
            calls[worker].append(request.url.path)
            if worker == "worker-b" and request.url.path == "/stage" and calls[worker].count("/stage") == 1:
                return httpx.Response(500, json={"error": "stage failed"})
            return httpx.Response(200, json={"status": "ok"})

        return _handle

    async def run() -> None:
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(handler("worker-a")), base_url="http://worker-a") as a,
            httpx.AsyncClient(transport=httpx.MockTransport(handler("worker-b")), base_url="http://worker-b") as b,
        ):
            pool = StaticInferencePool(
                ClientConfig(
                    base_url=["http://worker-a/v1", "http://worker-b/v1"],
                    lease_enabled=True,
                    lease_cooldown_s=0.0,
                    lease_recovery_enabled=True,
                    lease_recovery_poll_interval_s=0.01,
                ),
                model_name="test-model",
            )
            for admin_client in pool.admin_clients:
                await admin_client.aclose()
            pool._admin_clients = [a, b]
            pool._endpoint_stats = {pool._endpoint_key(client): {} for client in pool._admin_clients}
            pool._endpoint_runtime = {
                pool._endpoint_key(client): EndpointLeaseRuntime() for client in pool._admin_clients
            }

            await pool.stage_weights(
                delta_dir,
                version="1",
                mode="delta",
                base_version="0",
                upload=True,
            )

            assert pool._endpoint_runtime["http://worker-a"].state == "healthy"
            assert pool._endpoint_runtime["http://worker-b"].state == "retired"

            await pool.commit_weights(version="1", mode="delta")

            assert pool._endpoint_runtime["http://worker-a"].state == "healthy"
            assert pool._endpoint_runtime["http://worker-b"].state == "healthy"

    asyncio.run(run())

    assert calls["worker-a"] == ["/stage", "/pause", "/commit", "/resume"]
    assert calls["worker-b"] == [
        "/stage",
        "/health",
        "/pause",
        "/reload_weights",
        "/resume",
        "/stage",
        "/pause",
        "/commit",
        "/resume",
    ]


def test_static_inference_pool_records_endpoint_metrics():
    pool = StaticInferencePool.__new__(StaticInferencePool)
    pool._admin_clients = [
        httpx.AsyncClient(base_url="http://worker-a"),
        httpx.AsyncClient(base_url="http://worker-b"),
    ]
    pool._endpoint_stats = {str(client.base_url): {} for client in pool._admin_clients}

    pool._record_endpoint_results(
        [
            EndpointOperationResult("http://worker-a", "stage_weights", 0.25),
            EndpointOperationResult("http://worker-b", "stage_weights", 0.5, error="boom"),
            EndpointOperationResult("http://worker-a", "commit_weights", 0.1),
        ]
    )

    metrics = pool.get_metrics()
    assert metrics["inference_endpoint/0/stage_weights/requests"] == 1.0
    assert metrics["inference_endpoint/0/stage_weights/errors"] == 0.0
    assert metrics["inference_endpoint/0/stage_weights/last_duration_s"] == 0.25
    assert metrics["inference_endpoint/1/stage_weights/requests"] == 1.0
    assert metrics["inference_endpoint/1/stage_weights/errors"] == 1.0
    assert metrics["inference_endpoint/0/commit_weights/requests"] == 1.0

    asyncio.run(pool._admin_clients[0].aclose())
    asyncio.run(pool._admin_clients[1].aclose())
