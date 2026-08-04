from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Awaitable, Callable, Literal, Protocol, runtime_checkable

import httpx
import verifiers as vf
from httpx import AsyncClient
from openai import NotFoundError
from renderers import RendererConfig
from tenacity import retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential

from prime_rl.configs.shared import ClientConfig
from prime_rl.utils.logger import get_logger

STAGE_UPLOAD_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class EndpointOperationResult:
    endpoint: str
    operation: str
    duration_s: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class EndpointOperationError(RuntimeError):
    def __init__(self, operation: str, results: list[EndpointOperationResult]):
        self.operation = operation
        self.results = results
        failures = [result for result in results if not result.ok]
        message = "; ".join(f"{failure.endpoint}: {failure.error}" for failure in failures)
        super().__init__(f"{operation} failed on {len(failures)}/{len(results)} inference endpoint(s): {message}")


@dataclass
class EndpointStats:
    requests: int = 0
    errors: int = 0
    total_duration_s: float = 0.0
    last_duration_s: float = 0.0

    def record(self, result: EndpointOperationResult) -> None:
        self.requests += 1
        if not result.ok:
            self.errors += 1
        self.total_duration_s += result.duration_s
        self.last_duration_s = result.duration_s


EndpointLeaseState = Literal["healthy", "quarantine", "retired", "recovering"]


@dataclass
class EndpointLeaseRuntime:
    state: EndpointLeaseState = "healthy"
    quarantine_until: float | None = None
    quarantine_count: int = 0
    retired_count: int = 0
    recovery_attempt_count: int = 0
    recovery_success_count: int = 0
    last_reason: str | None = None


@dataclass(frozen=True)
class DeltaReplayEntry:
    weight_path: Path
    version: str
    base_version: str | None
    upload: bool
    upload_method: Literal["multipart", "chunked", "streaming"]
    done_path: Path | None


@runtime_checkable
class InferencePool(Protocol):
    """Protocol for inference pools (static or elastic)."""

    @property
    def model_name(self) -> str:
        """Get current model name for inference requests."""
        ...

    @property
    def train_clients(self) -> list[vf.ClientConfig]:
        """Get inference clients."""
        ...

    @property
    def admin_clients(self) -> list[AsyncClient]:
        """Get admin clients."""
        ...

    def update_model_name(self, model_name: str) -> None:
        """Update the model name."""
        ...

    async def get_eval_client(self) -> vf.ClientConfig:
        """Get next eval client in round-robin fashion."""
        ...

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None:
        """Wait for inference pool to be ready."""
        ...

    async def update_weights(
        self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0, mode: str = "full"
    ) -> None:
        """Update weights on all inference servers."""
        ...

    async def stage_weights(
        self,
        weight_path: Path,
        version: str,
        mode: str = "full",
        base_version: str | None = None,
        upload: bool = False,
        upload_method: Literal["multipart", "chunked", "streaming"] = "multipart",
        done_path: Path | None = None,
    ) -> None:
        """Stage weights on all inference servers."""
        ...

    async def commit_weights(self, version: str, mode: str | None = None) -> None:
        """Commit staged weights on all inference servers."""
        ...

    def get_metrics(self) -> dict[str, float]:
        """Get pool metrics."""
        ...

    async def stop(self) -> None:
        """Stop the inference pool."""
        ...


class StaticInferencePool:
    """Static inference pool with fixed rollout and admin endpoint lists."""

    def __init__(
        self,
        client_config: ClientConfig,
        model_name: str,
        train_client_type: str = "openai_chat_completions",
        eval_client_type: str = "openai_chat_completions",
        renderer_config: RendererConfig | None = None,
        pool_size: int | None = None,
    ):
        renderer_model_name = model_name if train_client_type == "renderer" else None
        self._train_clients = setup_clients(
            client_config,
            client_type=train_client_type,
            renderer_config=renderer_config,
            renderer_model_name=renderer_model_name,
            pool_size=pool_size,
        )
        self._eval_clients = setup_clients(client_config, client_type=eval_client_type)
        self._admin_clients = setup_admin_clients(client_config)
        self._skip_model_check = client_config.skip_model_check
        self._wait_for_ready_timeout = client_config.wait_for_ready_timeout
        self._eval_cycle = cycle(self._eval_clients)
        self._endpoint_aliases = self._build_endpoint_aliases(client_config)
        self._endpoint_stats: dict[str, dict[str, EndpointStats]] = {
            self._endpoint_key(admin_client): {} for admin_client in self._admin_clients
        }
        self._endpoint_runtime: dict[str, EndpointLeaseRuntime] = {
            self._endpoint_key(admin_client): EndpointLeaseRuntime() for admin_client in self._admin_clients
        }
        self._lease_enabled = client_config.lease_enabled
        self._lease_cooldown_s = client_config.lease_cooldown_s
        self._lease_recovery_enabled = client_config.lease_recovery_enabled
        self._lease_recovery_poll_interval_s = client_config.lease_recovery_poll_interval_s
        self._delta_replay_entries: dict[str, DeltaReplayEntry] = {}
        self._active_version: str | None = None
        self.model_name = model_name
        self._log_grouped_admin_aliases(client_config)

    @property
    def train_clients(self) -> list[vf.ClientConfig]:
        return self._healthy_clients(self._train_clients)

    @property
    def admin_clients(self) -> list[AsyncClient]:
        return self._admin_clients

    def update_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def eval_clients(self) -> list[vf.ClientConfig]:
        return self._healthy_clients(self._eval_clients)

    async def get_eval_client(self) -> vf.ClientConfig:
        while True:
            for _ in range(len(self._eval_clients)):
                client = next(self._eval_cycle)
                if self._client_is_healthy(client):
                    return client
            await asyncio.sleep(1)

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None:
        await check_health(
            self._admin_clients, timeout=timeout if timeout is not None else self._wait_for_ready_timeout
        )
        await maybe_check_has_model(self._admin_clients, model_name, skip_model_check=self._skip_model_check)

    async def update_weights(
        self, weight_dir: Path | None, lora_name: str | None = None, step: int = 0, mode: str = "full"
    ) -> None:
        await self._record_admin_results(
            update_weights(
                self._healthy_admin_clients("update_weights"),
                weight_dir,
                lora_name=lora_name,
                step=step,
                mode=mode,
            ),
            allow_partial=self._lease_enabled,
        )

    async def stage_weights(
        self,
        weight_path: Path,
        version: str,
        mode: str = "full",
        base_version: str | None = None,
        upload: bool = False,
        upload_method: Literal["multipart", "chunked", "streaming"] = "multipart",
        done_path: Path | None = None,
    ) -> None:
        await self._record_admin_results(
            stage_weights(
                self._healthy_admin_clients("stage_weights"),
                weight_path,
                version=version,
                mode=mode,
                base_version=base_version,
                upload=upload,
                upload_method=upload_method,
                done_path=done_path,
            ),
            allow_partial=self._lease_enabled,
        )
        self._remember_delta_replay_entry(
            weight_path=weight_path,
            version=version,
            mode=mode,
            base_version=base_version,
            upload=upload,
            upload_method=upload_method,
            done_path=done_path,
        )

    async def commit_weights(self, version: str, mode: str | None = None) -> None:
        await self._record_admin_results(
            commit_weights(self._healthy_admin_clients("commit_weights"), version=version, mode=mode),
            allow_partial=self._lease_enabled,
        )
        self._active_version = version
        if mode == "delta" and self._lease_recovery_enabled:
            await self.recover_unhealthy_endpoints()

    async def reload_weights(self) -> None:
        await self._record_admin_results(
            reload_weights(self._healthy_admin_clients("reload_weights")),
            allow_partial=self._lease_enabled,
        )

    def quarantine_client(self, client_config: vf.ClientConfig, reason: str) -> None:
        """Quarantine the endpoint backing a rollout client after a request-level failure."""
        if not self._lease_enabled:
            return
        endpoint = self._client_endpoint_key(client_config)
        if endpoint not in self._endpoint_runtime:
            return
        self._set_endpoint_state(endpoint, "quarantine", reason=reason)

    def get_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for endpoint_idx, endpoint in enumerate(self._admin_clients):
            endpoint_key = self._endpoint_key(endpoint)
            runtime = getattr(self, "_endpoint_runtime", {}).get(endpoint_key)
            if runtime is not None:
                state_values = {"healthy": 0.0, "quarantine": 1.0, "retired": 2.0, "recovering": 3.0}
                metrics[f"inference_endpoint/{endpoint_idx}/state"] = state_values[runtime.state]
                metrics[f"inference_endpoint/{endpoint_idx}/quarantine_count"] = float(runtime.quarantine_count)
                metrics[f"inference_endpoint/{endpoint_idx}/retired_count"] = float(runtime.retired_count)
                metrics[f"inference_endpoint/{endpoint_idx}/recovery_attempts"] = float(runtime.recovery_attempt_count)
                metrics[f"inference_endpoint/{endpoint_idx}/recovery_successes"] = float(runtime.recovery_success_count)

            endpoint_stats = self._endpoint_stats.get(endpoint_key, {})
            for operation, stats in endpoint_stats.items():
                prefix = f"inference_endpoint/{endpoint_idx}/{operation}"
                metrics[f"{prefix}/requests"] = float(stats.requests)
                metrics[f"{prefix}/errors"] = float(stats.errors)
                metrics[f"{prefix}/total_duration_s"] = stats.total_duration_s
                metrics[f"{prefix}/last_duration_s"] = stats.last_duration_s
        return metrics

    async def stop(self) -> None:
        pass

    def _build_endpoint_aliases(self, client_config: ClientConfig) -> dict[str, str]:
        admin_urls = client_config.admin_base_url if client_config.admin_base_url else client_config.base_url
        if len(client_config.base_url) == len(admin_urls):
            return {
                self._normalize_endpoint_url(base_url): self._normalize_endpoint_url(admin_url)
                for base_url, admin_url in zip(client_config.base_url, admin_urls)
            }
        if len(admin_urls) == 0 or len(client_config.base_url) % len(admin_urls) != 0:
            return {}

        group_size = len(client_config.base_url) // len(admin_urls)
        return {
            self._normalize_endpoint_url(base_url): self._normalize_endpoint_url(admin_urls[idx // group_size])
            for idx, base_url in enumerate(client_config.base_url)
        }

    def _log_grouped_admin_aliases(self, client_config: ClientConfig) -> None:
        admin_urls = client_config.admin_base_url if client_config.admin_base_url else client_config.base_url
        if not self._lease_enabled or len(client_config.base_url) <= len(admin_urls):
            return
        if len(admin_urls) == 0 or len(client_config.base_url) % len(admin_urls) != 0:
            get_logger().warning(
                f"Lease is enabled with {len(client_config.base_url)} rollout endpoint(s) and {len(admin_urls)} "
                "admin endpoint(s), but they cannot be grouped evenly; rollout-only endpoints are not directly "
                "lease-managed."
            )
            return
        group_size = len(client_config.base_url) // len(admin_urls)
        get_logger().info(
            f"Lease is enabled with {len(client_config.base_url)} rollout endpoint(s) and {len(admin_urls)} "
            f"admin endpoint(s); mapping each contiguous group of {group_size} rollout endpoint(s) to one "
            "admin endpoint."
        )

    def _endpoint_key(self, admin_client: AsyncClient) -> str:
        return self._normalize_endpoint_url(str(admin_client.base_url))

    def _normalize_endpoint_url(self, url: str) -> str:
        return url.rstrip("/").removesuffix("/v1")

    def _client_endpoint_key(self, client_config: vf.ClientConfig) -> str:
        endpoint = self._normalize_endpoint_url(str(client_config.api_base_url))
        return self._endpoint_aliases.get(endpoint, endpoint)

    def _client_is_healthy(self, client_config: vf.ClientConfig) -> bool:
        if not self._lease_enabled:
            return True
        runtime = self._endpoint_runtime.get(self._client_endpoint_key(client_config))
        return runtime is None or runtime.state == "healthy"

    def _healthy_clients(self, clients: list[vf.ClientConfig]) -> list[vf.ClientConfig]:
        return [client for client in clients if self._client_is_healthy(client)]

    def _healthy_admin_clients(self, operation: str) -> list[AsyncClient]:
        if not self._lease_enabled:
            return self._admin_clients

        selected_clients = []
        skipped_endpoints = []
        for admin_client in self._admin_clients:
            endpoint = self._endpoint_key(admin_client)
            runtime = self._endpoint_runtime.setdefault(endpoint, EndpointLeaseRuntime())
            if runtime.state == "healthy":
                selected_clients.append(admin_client)
                continue

            skipped_endpoints.append(f"{endpoint}({runtime.state})")
            if runtime.state == "quarantine":
                self._set_endpoint_state(
                    endpoint,
                    "retired",
                    reason=f"missed {operation}; delta catch-up is required before reuse",
                )

        if skipped_endpoints:
            get_logger().warning(
                f"[weights/{operation}] skipping non-healthy endpoints: {', '.join(skipped_endpoints)}"
            )
        if not selected_clients:
            raise RuntimeError(f"No healthy inference endpoints are available for {operation}.")
        return selected_clients

    def _set_endpoint_state(self, endpoint: str, state: EndpointLeaseState, reason: str) -> None:
        runtime = self._endpoint_runtime.setdefault(endpoint, EndpointLeaseRuntime())
        previous_state = runtime.state
        if previous_state == state and state != "retired":
            runtime.last_reason = reason
            return

        runtime.state = state
        runtime.last_reason = reason
        if state == "quarantine":
            runtime.quarantine_count += 1
            runtime.quarantine_until = time.monotonic() + self._lease_cooldown_s
        elif state == "retired":
            runtime.retired_count += 1
            runtime.quarantine_until = time.monotonic() + self._lease_cooldown_s
        elif state == "recovering":
            runtime.recovery_attempt_count += 1
            runtime.quarantine_until = None
        elif state == "healthy":
            runtime.quarantine_until = None
            if previous_state == "recovering":
                runtime.recovery_success_count += 1

        get_logger().warning(f"[{endpoint}] endpoint state {previous_state} -> {state}: {reason}")

    def _mark_failed_endpoints_retired(self, results: list[EndpointOperationResult]) -> None:
        if not self._lease_enabled:
            return
        for result in results:
            if result.ok:
                continue
            self._set_endpoint_state(
                result.endpoint,
                "retired",
                reason=f"{result.operation} failed: {result.error}",
            )

    async def _record_admin_results(
        self, operation: Awaitable[list[EndpointOperationResult]], *, allow_partial: bool = False
    ) -> list[EndpointOperationResult]:
        try:
            results = await operation
        except EndpointOperationError as exc:
            self._record_endpoint_results(exc.results)
            self._mark_failed_endpoints_retired(exc.results)
            if allow_partial and any(result.ok for result in exc.results):
                get_logger().warning(f"Ignoring partial inference endpoint failure during {exc.operation}: {exc}")
                return exc.results
            raise
        self._record_endpoint_results(results)
        return results

    def _record_endpoint_results(self, results: list[EndpointOperationResult]) -> None:
        for result in results:
            endpoint_stats = self._endpoint_stats.setdefault(result.endpoint, {})
            operation_stats = endpoint_stats.setdefault(result.operation, EndpointStats())
            operation_stats.record(result)

    def _remember_delta_replay_entry(
        self,
        *,
        weight_path: Path,
        version: str,
        mode: str,
        base_version: str | None,
        upload: bool,
        upload_method: Literal["multipart", "chunked", "streaming"],
        done_path: Path | None,
    ) -> None:
        if mode != "delta":
            return
        self._delta_replay_entries[version] = DeltaReplayEntry(
            weight_path=weight_path,
            version=version,
            base_version=base_version,
            upload=upload,
            upload_method=upload_method,
            done_path=done_path,
        )

    async def recover_unhealthy_endpoints(self) -> None:
        if not self._lease_recovery_enabled or self._active_version is None:
            return

        now = time.monotonic()
        for admin_client in self._admin_clients:
            endpoint = self._endpoint_key(admin_client)
            runtime = self._endpoint_runtime.setdefault(endpoint, EndpointLeaseRuntime())
            if runtime.state not in {"quarantine", "retired"}:
                continue
            if runtime.quarantine_until is not None and now < runtime.quarantine_until:
                continue
            if not await self._endpoint_health_check(admin_client):
                continue

            if runtime.state == "quarantine":
                self._set_endpoint_state(endpoint, "healthy", reason="health check succeeded after quarantine")
                continue

            await self._recover_retired_endpoint(admin_client)

    async def _endpoint_health_check(self, admin_client: AsyncClient) -> bool:
        try:
            response = await admin_client.get("/health", timeout=self._lease_recovery_poll_interval_s)
            if response.status_code != 404:
                response.raise_for_status()
        except Exception:
            return False
        return True

    async def _recover_retired_endpoint(self, admin_client: AsyncClient) -> None:
        endpoint = self._endpoint_key(admin_client)
        self._set_endpoint_state(endpoint, "recovering", reason="starting delta replay recovery")
        try:
            await self._record_admin_results(reload_weights([admin_client]))
            for entry in self._delta_replay_plan():
                get_logger().info(f"[{endpoint}] replaying delta version {entry.version}")
                await self._record_admin_results(
                    stage_weights(
                        [admin_client],
                        entry.weight_path,
                        version=entry.version,
                        mode="delta",
                        base_version=entry.base_version,
                        upload=entry.upload,
                        upload_method=entry.upload_method,
                        done_path=entry.done_path,
                    )
                )
                await self._record_admin_results(commit_weights([admin_client], version=entry.version, mode="delta"))
        except Exception as exc:
            self._set_endpoint_state(endpoint, "retired", reason=f"delta replay recovery failed: {exc}")
            return
        self._set_endpoint_state(endpoint, "healthy", reason=f"replayed delta chain to version {self._active_version}")

    def _delta_replay_plan(self) -> list[DeltaReplayEntry]:
        if self._active_version is None:
            return []
        active_version = self._parse_version(self._active_version)
        if active_version is None:
            raise RuntimeError(f"delta recovery requires numeric versions, got {self._active_version!r}")

        entries = sorted(
            self._delta_replay_entries.values(), key=lambda entry: self._parse_version(entry.version) or -1
        )
        plan = []
        expected_version = 1
        for entry in entries:
            entry_version = self._parse_version(entry.version)
            if entry_version is None:
                raise RuntimeError(f"delta recovery requires numeric versions, got {entry.version!r}")
            if entry_version > active_version:
                continue
            if entry_version != expected_version:
                raise RuntimeError(
                    f"missing delta replay entry for version {expected_version}; cannot recover to {active_version}"
                )
            plan.append(entry)
            expected_version += 1

        if expected_version <= active_version:
            raise RuntimeError(f"missing delta replay entry for version {expected_version}")
        return plan

    def _parse_version(self, version: str) -> int | None:
        try:
            return int(version)
        except ValueError:
            return None


async def setup_inference_pool(
    client_config: ClientConfig,
    model_name: str,
    train_client_type: str = "openai_chat_completions",
    eval_client_type: str = "openai_chat_completions",
    renderer_config: RendererConfig | None = None,
    pool_size: int | None = None,
) -> InferencePool:
    """Create an inference pool from config (static or elastic)."""
    if client_config.is_elastic:
        from prime_rl.utils.elastic import ElasticInferencePool

        return await ElasticInferencePool.from_config(
            client_config,
            model_name=model_name,
            train_client_type=train_client_type,
            eval_client_type=eval_client_type,
            renderer_config=renderer_config,
            pool_size=pool_size,
        )

    return StaticInferencePool(
        client_config,
        model_name=model_name,
        train_client_type=train_client_type,
        eval_client_type=eval_client_type,
        renderer_config=renderer_config,
        pool_size=pool_size,
    )


def setup_clients(
    client_config: ClientConfig,
    client_type: str = "openai_chat_completions",
    renderer_config: RendererConfig | None = None,
    renderer_model_name: str | None = None,
    pool_size: int | None = None,
) -> list[vf.ClientConfig]:
    clients = []
    client_idx = 0
    # Only forward the renderer config when the client actually uses a
    # renderer — MITO/TITO clients ignore it.
    renderer_extra: dict = {}
    if client_type == "renderer":
        renderer_extra = {
            "renderer_config": renderer_config,
            "renderer_model_name": renderer_model_name,
            "renderer_pool_size": pool_size,
        }
    env_headers = {
        k: v for k, v in ((k, os.getenv(v)) for k, v in client_config.headers_from_env.items()) if v is not None
    }
    for base_url in client_config.base_url:
        for dp_rank in range(client_config.dp_rank_count):
            headers = {**client_config.headers, **env_headers}
            if client_config.dp_rank_count > 1:
                headers["X-data-parallel-rank"] = str(dp_rank)
            clients.append(
                vf.ClientConfig(
                    client_idx=client_idx,
                    client_type=client_type,
                    api_base_url=base_url,
                    api_key_var=client_config.api_key_var,
                    timeout=client_config.timeout,
                    connect_timeout=client_config.connect_timeout,
                    max_connections=8192,
                    max_keepalive_connections=8192,
                    max_retries=10,
                    extra_headers=headers,
                    extra_headers_from_state=client_config.extra_headers_from_state,
                    **renderer_extra,
                )
            )
            client_idx += 1
    return clients


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
    """Create dedicated admin clients for weight update operations.

    Uses a separate connection pool to avoid queueing behind streaming requests.
    When admin_base_url is set, uses those URLs instead of base_url, allowing
    weight updates to bypass routers in disaggregated P/D deployments.
    """
    urls = client_config.admin_base_url if client_config.admin_base_url else client_config.base_url

    def _setup_admin_client(base_url: str) -> httpx.AsyncClient:
        env_headers = {
            k: v for k, v in ((k, os.getenv(v)) for k, v in client_config.headers_from_env.items()) if v is not None
        }
        headers = {**client_config.headers, **env_headers}
        api_key = os.getenv(client_config.api_key_var, "EMPTY")
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"

        # Strip /v1 suffix since admin endpoints are at root level
        base_url = base_url.rstrip("/").removesuffix("/v1")

        return AsyncClient(
            base_url=base_url,
            headers=headers,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=1),
            timeout=httpx.Timeout(None),
        )

    return [_setup_admin_client(base_url) for base_url in urls]


def _endpoint_key(admin_client: AsyncClient) -> str:
    return str(admin_client.base_url)


def _format_exception(exception: BaseException) -> str:
    message = str(exception).strip()
    if message:
        return f"{exception.__class__.__name__}: {message}"
    return exception.__class__.__name__


async def _run_endpoint_operations(
    admin_clients: list[AsyncClient],
    operation: str,
    call: Callable[[AsyncClient], Awaitable[None]],
) -> list[EndpointOperationResult]:
    async def _run(admin_client: AsyncClient) -> EndpointOperationResult:
        start = time.perf_counter()
        try:
            await call(admin_client)
        except Exception as exc:
            return EndpointOperationResult(
                endpoint=_endpoint_key(admin_client),
                operation=operation,
                duration_s=time.perf_counter() - start,
                error=_format_exception(exc),
            )
        return EndpointOperationResult(
            endpoint=_endpoint_key(admin_client),
            operation=operation,
            duration_s=time.perf_counter() - start,
        )

    results = await asyncio.gather(*[_run(admin_client) for admin_client in admin_clients])
    if any(not result.ok for result in results):
        raise EndpointOperationError(operation, results)
    return results


async def maybe_check_has_model(
    admin_clients: list[AsyncClient], model_name: str, skip_model_check: bool = False
) -> None:
    if skip_model_check:
        return
    logger = get_logger()
    logger.debug(f"Checking if model {model_name} is in the inference pool")
    results = await asyncio.gather(*[admin_client.get("/v1/models") for admin_client in admin_clients])
    for admin_client, result in zip(admin_clients, results):
        models = result.json()["data"]
        if not any(model["id"] == model_name for model in models):
            raise ValueError(f"Model {model_name} was not found in the inference pool on {admin_client.base_url}")
    logger.debug(f"Model {model_name} was found in the inference pool")


async def check_health(
    admin_clients: list[AsyncClient], interval: int = 1, log_interval: int = 10, timeout: int = 1800
) -> None:
    logger = get_logger()

    async def _check_health(admin_client: AsyncClient) -> None:
        wait_time = 0
        logger.debug("Starting pinging /health to check health")
        while wait_time < timeout:
            try:
                await admin_client.get("/health")
                logger.debug(f"Inference pool is ready after {wait_time} seconds")
                return
            except NotFoundError:
                logger.warning("The route /health does not exist. Skipping health check.")
                return
            except Exception as e:
                if wait_time % log_interval == 0 and wait_time > 0:
                    logger.warning(
                        f"Inference server was not reached after {wait_time} seconds (Error: {e}) on {admin_client.base_url}"
                    )
                await asyncio.sleep(interval)
                wait_time += interval
        msg = f"Inference server is not ready after {wait_time} (>{timeout}) seconds. Aborting..."
        logger.error(msg)
        raise TimeoutError(msg)

    await asyncio.gather(*[_check_health(admin_client) for admin_client in admin_clients])


NCCL_READY_MARKER = "NCCL_READY"


async def _pause_engines(admin_clients: list[AsyncClient]) -> None:
    """Pause all inference engines, waiting for in-flight requests to drain."""
    logger = get_logger()
    logger.info("Pausing inference engines for weight update")

    await asyncio.gather(*[_pause_engine(client) for client in admin_clients])
    logger.info("All inference engines paused")


async def _resume_engines(admin_clients: list[AsyncClient]) -> None:
    """Resume all inference engines after weight update."""
    logger = get_logger()

    await asyncio.gather(*[_resume_engine(client) for client in admin_clients])
    logger.info("All inference engines resumed")


async def _pause_engine(client: AsyncClient) -> None:
    response = await client.post("/pause", params={"mode": "keep", "clear_cache": "false"})
    response.raise_for_status()


async def _resume_engine(client: AsyncClient) -> None:
    response = await client.post("/resume")
    response.raise_for_status()


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | None,
    lora_name: str | None = None,
    step: int = 0,
    mode: str = "full",
) -> list[EndpointOperationResult]:
    """Update weights on static inference servers.

    Pauses all engines first to drain in-flight requests, then performs the
    weight update, then resumes. This ensures all DP workers are idle and can
    participate in the collective weight transfer.

    Note: The server-side /update_weights endpoint automatically resets the prefix cache
    to invalidate any cached KV states computed with the old weights.
    """
    logger = get_logger()

    weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None

    if lora_name is not None and weight_dir is not None:
        await load_lora_adapter(admin_clients, lora_name, weight_dir)
        return [
            EndpointOperationResult(endpoint=_endpoint_key(admin_client), operation="load_lora", duration_s=0.0)
            for admin_client in admin_clients
        ]
    else:

        async def _update_weights(admin_client: AsyncClient, weight_dir: str | None) -> None:
            response = await admin_client.post("/update_weights", json={"weight_dir": weight_dir, "mode": mode})
            response.raise_for_status()

        # Pause engines so all DP workers drain in-flight work and can join the NCCL broadcast
        await _pause_engines(admin_clients)

        try:
            # Create ready marker before servers enter receive path (used by NCCL broadcast)
            if weight_dir is not None:
                nccl_ready_file = weight_dir / NCCL_READY_MARKER
                nccl_ready_file.parent.mkdir(parents=True, exist_ok=True)
                nccl_ready_file.touch()
                logger.debug(f"Created NCCL_READY marker at {nccl_ready_file}")

            return await _run_endpoint_operations(
                admin_clients,
                "update_weights",
                lambda admin_client: _update_weights(admin_client, weight_dir_posix),
            )
        finally:
            await _resume_engines(admin_clients)


async def stage_weights(
    admin_clients: list[AsyncClient],
    weight_path: Path,
    version: str,
    mode: str = "full",
    base_version: str | None = None,
    upload: bool = False,
    upload_method: Literal["multipart", "chunked", "streaming"] = "multipart",
    chunk_size_bytes: int = STAGE_UPLOAD_CHUNK_BYTES,
    done_path: Path | None = None,
    poll_interval_s: float = 0.1,
) -> list[EndpointOperationResult]:
    """Stage weights on static inference servers.

    By default this records a path that is already visible to the inference
    server, matching the shared-filesystem delta path used by local migration
    runs. Set ``upload=True`` to stream a single file to the server's staging
    directory. ``upload_method="chunked"`` uses offset-based chunk upload plus
    final size/hash verification.
    """
    if mode not in {"full", "delta"}:
        raise ValueError(f"unsupported weight update mode: {mode}")
    if upload_method not in {"multipart", "chunked", "streaming"}:
        raise ValueError(f"unsupported stage upload method: {upload_method}")
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be positive")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")

    data = {"version": version, "mode": mode}
    if base_version is not None:
        data["base_version"] = base_version

    def _upload_path() -> Path:
        upload_path = weight_path
        if upload_path.is_dir():
            if mode != "delta":
                raise ValueError("upload=True requires a file path for full checkpoint staging")
            preferred_name = "delta.stream" if upload_method == "streaming" else "delta.safetensors"
            fallback_name = "delta.safetensors" if upload_method == "streaming" else "delta.stream"
            preferred_path = upload_path / preferred_name
            fallback_path = upload_path / fallback_name
            upload_path = preferred_path if preferred_path.exists() or not fallback_path.exists() else fallback_path
        return upload_path

    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(STAGE_UPLOAD_CHUNK_BYTES):
                digest.update(chunk)
        return digest.hexdigest()

    async def _stage_path(admin_client: AsyncClient) -> None:
        response = await admin_client.post("/stage", data={**data, "path": weight_path.as_posix()})
        response.raise_for_status()

    async def _stage_multipart_upload(admin_client: AsyncClient, upload_path: Path) -> None:
        with upload_path.open("rb") as f:
            files = {"file": (upload_path.name, f, "application/octet-stream")}
            response = await admin_client.post("/stage", data=data, files=files)
        response.raise_for_status()

    async def _stage_chunked_upload(
        admin_client: AsyncClient,
        upload_path: Path,
        total_size: int,
        expected_sha256: str,
    ) -> None:
        chunk_data = {
            **data,
            "filename": upload_path.name,
            "total_size": str(total_size),
            "sha256": expected_sha256,
        }
        with upload_path.open("rb") as f:
            offset = 0
            sent_chunk = False
            while chunk := f.read(chunk_size_bytes):
                sent_chunk = True
                files = {"file": (upload_path.name, chunk, "application/octet-stream")}
                response = await admin_client.post(
                    "/stage_chunk",
                    data={**chunk_data, "offset": str(offset)},
                    files=files,
                )
                response.raise_for_status()
                offset += len(chunk)

            if not sent_chunk:
                files = {"file": (upload_path.name, b"", "application/octet-stream")}
                response = await admin_client.post("/stage_chunk", data={**chunk_data, "offset": "0"}, files=files)
                response.raise_for_status()

        response = await admin_client.post("/stage_finalize", data=chunk_data)
        response.raise_for_status()

    async def _stage_streaming_upload(
        admin_client: AsyncClient,
        upload_path: Path,
        done_path: Path | None,
    ) -> None:
        if mode != "delta":
            raise ValueError("streaming upload currently supports delta mode only")
        if done_path is None and not upload_path.exists():
            raise FileNotFoundError(upload_path)

        init_data = {
            **data,
            "filename": upload_path.name,
        }
        response = await admin_client.post("/stage_stream_init", json=init_data)
        response.raise_for_status()
        upload_id = response.json().get("upload_id")
        if not upload_id:
            raise RuntimeError("stage_stream_init response did not include upload_id")

        chunk_index = 0
        offset = 0
        while True:
            if upload_path.exists():
                size = upload_path.stat().st_size
                if size > offset:
                    to_read = min(chunk_size_bytes, size - offset)
                    with upload_path.open("rb") as f:
                        f.seek(offset)
                        chunk = f.read(to_read)
                    if chunk:
                        response = await admin_client.post(
                            "/stage_stream_chunk",
                            params={"upload_id": upload_id, "chunk_index": chunk_index, "offset": offset},
                            content=chunk,
                        )
                        response.raise_for_status()
                        offset += len(chunk)
                        chunk_index += 1
                        continue

            if done_path is None:
                if upload_path.exists() and offset >= upload_path.stat().st_size:
                    break
            elif done_path.exists():
                final_size = upload_path.stat().st_size if upload_path.exists() else 0
                if offset >= final_size:
                    break

            await asyncio.sleep(poll_interval_s)

        if not upload_path.exists():
            raise FileNotFoundError(upload_path)
        final_size = upload_path.stat().st_size
        finalize_data = {
            "upload_id": upload_id,
            "final_size": str(final_size),
            "sha256": _sha256_file(upload_path),
        }
        response = await admin_client.post("/stage_stream_finalize", data=finalize_data)
        response.raise_for_status()

    if upload:
        upload_path = _upload_path()
        if upload_method == "multipart":
            return await _run_endpoint_operations(
                admin_clients,
                "stage_weights",
                lambda admin_client: _stage_multipart_upload(admin_client, upload_path),
            )
        if upload_method == "chunked":
            total_size = upload_path.stat().st_size
            expected_sha256 = _sha256_file(upload_path)
            return await _run_endpoint_operations(
                admin_clients,
                "stage_weights",
                lambda admin_client: _stage_chunked_upload(admin_client, upload_path, total_size, expected_sha256),
            )
        return await _run_endpoint_operations(
            admin_clients,
            "stage_weights",
            lambda admin_client: _stage_streaming_upload(admin_client, upload_path, done_path),
        )
    return await _run_endpoint_operations(admin_clients, "stage_weights", _stage_path)


async def commit_weights(
    admin_clients: list[AsyncClient], version: str, mode: str | None = None
) -> list[EndpointOperationResult]:
    """Commit a staged weight version on static inference servers."""
    data = {"version": version}
    if mode is not None:
        data["mode"] = mode

    async def _commit(admin_client: AsyncClient) -> None:
        response = await admin_client.post("/commit", data=data)
        response.raise_for_status()

    async def _commit_with_pause(admin_client: AsyncClient) -> None:
        await _pause_engine(admin_client)
        try:
            await _commit(admin_client)
        finally:
            await _resume_engine(admin_client)

    return await _run_endpoint_operations(admin_clients, "commit_weights", _commit_with_pause)


async def reload_weights(admin_clients: list[AsyncClient]) -> list[EndpointOperationResult]:
    """Reload base model weights on static inference servers."""

    async def _reload(admin_client: AsyncClient) -> None:
        response = await admin_client.post("/reload_weights")
        response.raise_for_status()

    async def _reload_with_pause(admin_client: AsyncClient) -> None:
        await _pause_engine(admin_client)
        try:
            await _reload(admin_client)
        finally:
            await _resume_engine(admin_client)

    return await _run_endpoint_operations(admin_clients, "reload_weights", _reload_with_pause)


def _is_retryable_lora_error(exception: BaseException) -> bool:
    """Check if an exception should trigger a retry for LoRA loading."""
    if isinstance(exception, httpx.HTTPStatusError):
        # Retry on 404 (adapter not found) or 500 (server error during loading)
        return exception.response.status_code in (404, 500)
    # Retry on transport-level failures (timeouts, connection resets, etc.) so
    # the per-call read timeout below turns a stuck server into a bounded retry
    # loop instead of propagating as a hard failure on the first hiccup.
    if isinstance(exception, (httpx.TimeoutException, httpx.TransportError)):
        return True
    return False


# Per-attempt and total bounds for `/load_lora_adapter`. A LoRA load is fast
# (small adapter file + KV cache reset, single-digit seconds in practice) but
# the global admin AsyncClient uses `timeout=None`, so a stuck server hangs
# the orchestrator forever inside `ElasticInferencePool._sync_server_adapter`.
# `_PER_ATTEMPT` converts a hang into a TimeoutException so tenacity retries;
# `_TOTAL` is the wall-clock budget across all retries — pick whichever
# stop condition fires first.
LORA_LOAD_READ_TIMEOUT_S = 30.0
LORA_LOAD_TOTAL_TIMEOUT_S = 120.0


async def load_lora_adapter(admin_clients: list[AsyncClient], lora_name: str, lora_path: Path) -> None:
    """Make a HTTP post request to the vLLM server to load a LoRA adapter.

    Uses our wrapper endpoint that also resets the prefix cache to invalidate
    KV states computed with old weights.

    Retries with exponential backoff if the adapter files are not found,
    which can happen due to NFS propagation delays.
    """
    logger = get_logger()
    lora_path_posix = lora_path.as_posix()

    @retry(
        retry=retry_if_exception(_is_retryable_lora_error),
        stop=stop_after_delay(LORA_LOAD_TOTAL_TIMEOUT_S) | stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _load_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to load LoRA adapter {lora_name} from {lora_path}")
        response = await admin_client.post(
            "/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path_posix},
            timeout=httpx.Timeout(connect=10.0, read=LORA_LOAD_READ_TIMEOUT_S, write=60.0, pool=10.0),
        )
        response.raise_for_status()

    await asyncio.gather(*[_load_lora_adapter(admin_client) for admin_client in admin_clients])


async def unload_lora_adapter(admin_clients: list[AsyncClient], lora_name: str) -> None:
    """Make a HTTP post request to the vLLM server to unload a LoRA adapter."""
    logger = get_logger()

    async def _unload_lora_adapter(admin_client: AsyncClient) -> None:
        logger.debug(f"Sending request to unload LoRA adapter {lora_name}")
        await admin_client.post("/v1/unload_lora_adapter", json={"lora_name": lora_name})
        # TODO: The first one can fail, but subsequent ones should succeed.
        # response.raise_for_status()

    await asyncio.gather(*[_unload_lora_adapter(admin_client) for admin_client in admin_clients])


async def init_nccl_broadcast(
    admin_clients: list[AsyncClient],
    host: str,
    port: int,
    timeout: int,
    inference_world_size: int | None = None,
    quantize_in_weight_transfer: bool = False,
) -> None:
    """Initialize NCCL broadcast on all inference servers.

    Each admin client represents one vLLM server. The function computes
    per-server rank_offset and gpus_per_server so that every inference GPU
    gets a unique rank in the NCCL broadcast group.
    """
    logger = get_logger()

    if inference_world_size is None:
        inference_world_size = len(admin_clients)
        logger.warning(
            f"inference_world_size not provided, defaulting to {inference_world_size} (one GPU per admin client)"
        )

    gpus_per_server = inference_world_size // len(admin_clients)

    logger.info(
        f"Initializing NCCL broadcast: {len(admin_clients)} servers, "
        f"inference_world_size={inference_world_size}, gpus_per_server={gpus_per_server}"
    )

    async def _init_nccl_broadcast(admin_client: AsyncClient, rank_offset: int) -> None:
        try:
            response = await admin_client.post(
                "/init_broadcaster",
                json={
                    "host": host,
                    "port": port,
                    "rank_offset": rank_offset,
                    "inference_world_size": inference_world_size,
                    "timeout": timeout,
                    "quantize_in_weight_transfer": quantize_in_weight_transfer,
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning("The route /init_broadcaster does not exist. Skipping NCCL broadcast initialization.")
                return

    await asyncio.gather(
        *[
            _init_nccl_broadcast(admin_client, client_num * gpus_per_server)
            for client_num, admin_client in enumerate(admin_clients)
        ]
    )
