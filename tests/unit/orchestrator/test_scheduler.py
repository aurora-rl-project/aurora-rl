import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import verifiers as vf

from prime_rl.orchestrator.scheduler import GroupState, InflightRequest, RolloutClientStats, Scheduler
from prime_rl.utils.async_utils import safe_cancel


def make_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.step = 9
    scheduler.ckpt_step = 7
    scheduler.config = SimpleNamespace(
        output_dir=Path("/tmp/prime-rl-test"),
        enable_load_balancing=False,
        throughput_ema_beta=0.8,
    )
    scheduler.logger = MagicMock()
    scheduler.checkpoint_ready = asyncio.Event()
    scheduler.checkpoint_ready.set()
    scheduler.lora_name = None
    scheduler.model_name = "test-model"
    scheduler.update_weights_time = 0
    scheduler.wait_for_ckpt_time = 0
    scheduler.stage_weights_time = 0
    scheduler.commit_weights_time = 0
    scheduler.inflight_requests = {}
    scheduler.groups = {}
    scheduler.max_off_policy_steps = 1
    scheduler.cancelled_rollouts_count = 0
    scheduler.policy_update_lock = asyncio.Lock()
    scheduler.inflight_policy_update_task = None
    scheduler.inflight_stage_task = None
    scheduler.inflight_stage_step = None
    scheduler.update_policy_task = None
    scheduler.rate_limiter = None
    scheduler.empty_rollouts_by_env = {}
    scheduler.errored_rollouts_by_env = {}
    scheduler.errors_by_type = {}
    scheduler.total_rollouts_by_env = {}
    scheduler.dropped_groups_by_env = {}
    scheduler.enable_load_balancing = False
    scheduler.throughput_ema_beta = 0.8
    scheduler._rollout_client_stats = {}
    return scheduler


def test_update_off_policy_does_not_increment_interleaved_on_policy_tasks():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_off_policy_steps = 1
        scheduler.cancelled_rollouts_count = 0
        scheduler.logger = MagicMock()

        client = SimpleNamespace(api_base_url="http://test")
        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))
        interleaved_task = None

        scheduler.inflight_requests = {
            stale_task: InflightRequest(off_policy_steps=1, client_config=client, env_name="test", group_id=1),
            survivor_task: InflightRequest(off_policy_steps=0, client_config=client, env_name="test", group_id=2),
        }

        async def drop_group(group_id: int) -> int:
            tasks_to_remove = [
                task for task, info in list(scheduler.inflight_requests.items()) if info.group_id == group_id
            ]
            for task in tasks_to_remove:
                scheduler.inflight_requests.pop(task, None)
                task.cancel()

            await asyncio.sleep(0)

            nonlocal interleaved_task
            if interleaved_task is None:
                interleaved_task = asyncio.create_task(asyncio.sleep(60))
                scheduler.inflight_requests[interleaved_task] = InflightRequest(
                    off_policy_steps=0,
                    client_config=client,
                    env_name="test",
                    group_id=3,
                )
            return len(tasks_to_remove)

        scheduler.drop_group = drop_group

        await scheduler._update_off_policy()

        assert stale_task not in scheduler.inflight_requests
        assert scheduler.inflight_requests[survivor_task].off_policy_steps == 1
        assert interleaved_task is not None
        assert scheduler.inflight_requests[interleaved_task].off_policy_steps == 0
        assert scheduler.cancelled_rollouts_count == 1

        for task in (stale_task, survivor_task, interleaved_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_maybe_update_policy_reuses_inflight_update_after_cancellation():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        release = asyncio.Event()
        applied_steps: list[int] = []

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            applied_steps.append(step)
            started.set()
            await release.wait()

        scheduler.student_inference = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler.rollout_inference = scheduler.student_inference
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            first = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await safe_cancel(first)

            second = asyncio.create_task(scheduler.maybe_update_policy())
            await asyncio.sleep(0)
            assert applied_steps == [8]

            release.set()
            await second

        assert applied_steps == [8]
        assert scheduler.ckpt_step == 8

    asyncio.run(run())


def test_policy_update_passes_delta_mode():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.config.weight_broadcast = SimpleNamespace(mode="delta")
        seen: dict[str, object] = {}

        async def update_weights(weight_dir, lora_name=None, step=0, mode="full") -> None:
            seen["weight_dir"] = weight_dir
            seen["step"] = step
            seen["mode"] = mode

        scheduler.student_inference = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler.rollout_inference = scheduler.student_inference
        scheduler._update_off_policy = AsyncMock()

        with patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()):
            await scheduler._apply_policy_update(8)

        assert seen["step"] == 8
        assert seen["mode"] == "delta"

    asyncio.run(run())


def test_delta_policy_update_advances_one_checkpoint_at_a_time():
    scheduler = make_scheduler()
    scheduler.ckpt_step = 7
    scheduler.step = 20
    scheduler.config.weight_broadcast = SimpleNamespace(mode="delta")

    with patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=12):
        assert scheduler._compute_next_ckpt_step() == 8


def test_full_policy_update_can_skip_to_latest_checkpoint():
    scheduler = make_scheduler()
    scheduler.ckpt_step = 7
    scheduler.step = 20
    scheduler.config.weight_broadcast = SimpleNamespace(mode="full")

    with patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=12):
        assert scheduler._compute_next_ckpt_step() == 19


def test_policy_update_can_use_stage_commit_protocol():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.config.weight_broadcast = SimpleNamespace(
            type="filesystem",
            mode="delta",
            update_protocol="stage_commit",
            stage_transport="http_upload",
        )
        calls: list[tuple[str, object]] = []

        async def update_weights(*args, **kwargs) -> None:
            raise AssertionError("direct update_weights should not be used")

        async def stage_weights(weight_path, version, mode="full", base_version=None, upload=False) -> None:
            calls.append(("stage", weight_path, version, mode, base_version, upload))

        async def commit_weights(version, mode=None) -> None:
            calls.append(("commit", version, mode))

        scheduler.student_inference = SimpleNamespace(
            update_weights=update_weights,
            stage_weights=stage_weights,
            commit_weights=commit_weights,
            update_model_name=MagicMock(),
        )
        scheduler.rollout_inference = scheduler.student_inference
        scheduler._update_off_policy = AsyncMock()

        with patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()):
            await scheduler._apply_policy_update(8)

        assert calls == [
            ("stage", Path("/tmp/prime-rl-test/broadcasts/step_8"), "8", "delta", "7", True),
            ("commit", "8", "delta"),
        ]
        assert scheduler.ckpt_step == 8
        assert scheduler.stage_weights_time >= 0
        assert scheduler.commit_weights_time >= 0

    asyncio.run(run())


def test_policy_update_failure_reopens_checkpoint_ready_after_wait():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.config.weight_broadcast = SimpleNamespace(mode="delta")

        async def update_weights(*args, **kwargs) -> None:
            raise RuntimeError("update failed")

        scheduler.student_inference = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler.rollout_inference = scheduler.student_inference
        scheduler._update_off_policy = AsyncMock()

        with patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="update failed"):
                await scheduler._apply_policy_update(8)

        assert scheduler.ckpt_step == 7
        assert scheduler.checkpoint_ready.is_set()

    asyncio.run(run())


def test_chunked_stage_transport_uses_chunked_upload_method():
    scheduler = make_scheduler()
    scheduler.config.weight_broadcast = SimpleNamespace(
        type="filesystem",
        mode="delta",
        update_protocol="stage_commit",
        stage_transport="chunked_upload",
    )

    assert scheduler._stage_upload_kwargs() == {"upload": True, "upload_method": "chunked"}


def test_streaming_stage_transport_uses_streaming_upload_method():
    scheduler = make_scheduler()
    scheduler.config.weight_broadcast = SimpleNamespace(
        type="filesystem",
        mode="delta",
        update_protocol="stage_commit",
        stage_transport="streaming_upload",
    )

    assert scheduler._stage_upload_kwargs() == {"upload": True, "upload_method": "streaming"}


def test_streaming_background_stage_waits_for_delta_file_and_finalizes_on_stable():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.config.weight_broadcast = SimpleNamespace(
            type="filesystem",
            mode="delta",
            update_protocol="stage_commit",
            stage_transport="streaming_upload",
            background_stage=True,
        )
        calls: list[tuple[str, object]] = []

        async def wait_for_ready(path) -> None:
            calls.append(("wait", path))

        async def stage_weights(
            weight_path,
            version,
            mode="full",
            base_version=None,
            upload=False,
            upload_method="multipart",
            done_path=None,
        ) -> None:
            calls.append(("stage", weight_path, version, mode, base_version, upload, upload_method, done_path))

        scheduler.student_inference = SimpleNamespace(
            update_weights=AsyncMock(),
            stage_weights=stage_weights,
            commit_weights=AsyncMock(),
            update_model_name=MagicMock(),
        )

        with patch("prime_rl.orchestrator.scheduler.wait_for_path", new=wait_for_ready):
            await scheduler._stage_policy_update(8)

        step_path = Path("/tmp/prime-rl-test/broadcasts/step_8")
        assert calls == [
            ("wait", step_path / "delta.stream"),
            ("stage", step_path, "8", "delta", "7", True, "streaming", step_path / "STABLE"),
        ]

    asyncio.run(run())


def test_background_stage_does_not_clear_checkpoint_ready_until_commit():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.config.weight_broadcast = SimpleNamespace(
            type="filesystem",
            mode="delta",
            update_protocol="stage_commit",
            stage_transport="http_upload",
            background_stage=True,
        )
        wait_started = asyncio.Event()
        release_wait = asyncio.Event()
        calls: list[tuple[str, object]] = []

        async def wait_for_stable(path) -> None:
            calls.append(("wait", path))
            wait_started.set()
            await release_wait.wait()

        async def stage_weights(weight_path, version, mode="full", base_version=None, upload=False) -> None:
            calls.append(("stage", weight_path, version, mode, base_version, upload))

        async def commit_weights(version, mode=None) -> None:
            calls.append(("commit", version, mode))

        scheduler.student_inference = SimpleNamespace(
            update_weights=AsyncMock(),
            stage_weights=stage_weights,
            commit_weights=commit_weights,
            update_model_name=MagicMock(),
        )
        scheduler.rollout_inference = scheduler.student_inference
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=wait_for_stable),
        ):
            await scheduler.maybe_update_policy(block=False)
            await wait_started.wait()

            assert scheduler.checkpoint_ready.is_set()
            assert calls == [("wait", Path("/tmp/prime-rl-test/broadcasts/step_8/STABLE"))]

            release_wait.set()
            await scheduler.maybe_update_policy(block=True)

        assert calls == [
            ("wait", Path("/tmp/prime-rl-test/broadcasts/step_8/STABLE")),
            ("stage", Path("/tmp/prime-rl-test/broadcasts/step_8"), "8", "delta", "7", True),
            ("commit", "8", "delta"),
        ]
        assert scheduler.ckpt_step == 8
        assert scheduler.checkpoint_ready.is_set()

    asyncio.run(run())


def test_background_stage_failure_clears_inflight_task_and_reopens_checkpoint_ready():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.config.weight_broadcast = SimpleNamespace(
            type="filesystem",
            mode="delta",
            update_protocol="stage_commit",
            stage_transport="http_upload",
            background_stage=True,
        )

        async def stage_weights(*args, **kwargs) -> None:
            raise RuntimeError("stage failed")

        scheduler.student_inference = SimpleNamespace(
            update_weights=AsyncMock(),
            stage_weights=stage_weights,
            commit_weights=AsyncMock(),
            update_model_name=MagicMock(),
        )
        scheduler.rollout_inference = scheduler.student_inference
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            await scheduler.maybe_update_policy(block=False)
            await asyncio.sleep(0)

            with pytest.raises(RuntimeError, match="stage failed"):
                await scheduler.maybe_update_policy(block=True)

        assert scheduler.inflight_stage_task is None
        assert scheduler.inflight_stage_step is None
        assert scheduler.checkpoint_ready.is_set()
        assert scheduler.ckpt_step == 7

    asyncio.run(run())


def test_stop_cancels_inflight_policy_update_task():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        scheduler.student_inference = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler.rollout_inference = scheduler.student_inference
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            scheduler.update_policy_task = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await asyncio.wait_for(scheduler.stop(), timeout=0.2)

        assert cancelled.is_set()
        assert scheduler.update_policy_task is None
        assert scheduler.inflight_policy_update_task is None

    asyncio.run(run())


def test_client_identity_distinguishes_base_url_and_dp_rank():
    client_a = vf.ClientConfig(
        api_base_url="http://worker-a:8000/v1",
        extra_headers={"X-data-parallel-rank": "0"},
    )
    client_b = vf.ClientConfig(
        api_base_url="http://worker-a:8000/v1",
        extra_headers={"X-data-parallel-rank": "1"},
    )

    assert Scheduler._client_identity(client_a) != Scheduler._client_identity(client_b)


def test_load_balancing_prefers_higher_capacity_client_at_equal_inflight_count():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.enable_load_balancing = True
        fast = vf.ClientConfig(client_idx=0, api_base_url="http://fast/v1", extra_headers={})
        slow = vf.ClientConfig(client_idx=1, api_base_url="http://slow/v1", extra_headers={})
        scheduler.rollout_inference = SimpleNamespace(train_clients=[slow, fast])
        scheduler._rollout_client_stats[Scheduler._client_identity(fast)] = RolloutClientStats(throughput_ema=100.0)
        scheduler._rollout_client_stats[Scheduler._client_identity(slow)] = RolloutClientStats(throughput_ema=10.0)

        fast_task = asyncio.create_task(asyncio.sleep(60))
        slow_task = asyncio.create_task(asyncio.sleep(60))
        scheduler.inflight_requests = {
            fast_task: InflightRequest(off_policy_steps=0, client_config=fast, env_name="test"),
            slow_task: InflightRequest(off_policy_steps=0, client_config=slow, env_name="test"),
        }

        try:
            assert await scheduler._select_least_loaded_client() is fast
        finally:
            for task in (fast_task, slow_task):
                task.cancel()
            await asyncio.sleep(0)

    asyncio.run(run())


def test_rollout_client_metrics_track_throughput_ema_and_reset_step_window():
    scheduler = make_scheduler()
    scheduler.enable_load_balancing = True
    client = vf.ClientConfig(client_idx=0, api_base_url="http://worker/v1", extra_headers={})
    scheduler.rollout_inference = SimpleNamespace(train_clients=[client], get_metrics=lambda: {})
    rollout = vf.RolloutOutput(
        env_name="test",
        error=None,
        info={},
        metrics={},
        reward=1.0,
        state={},
        trajectory=[
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 4, 5],
                },
                "response": {},
            }
        ],
    )

    scheduler._record_rollout_client_result(
        client,
        duration_s=2.0,
        rollouts=[rollout],
        error=False,
    )

    metrics = scheduler.get_metrics()

    assert metrics["scheduler/load_balancing_enabled"] == 1.0
    assert metrics["rollout_client/0/requests"] == 1.0
    assert metrics["rollout_client/0/tokens"] == 5.0
    assert metrics["rollout_client/0/last_throughput"] == 2.5
    assert metrics["rollout_client/0/throughput_ema"] == 2.5
    assert metrics["rollout_client/0/routing_weight"] == 1.0

    next_metrics = scheduler.get_metrics()

    assert next_metrics["rollout_client/0/requests"] == 0.0
    assert next_metrics["rollout_client/0/tokens"] == 0.0
    assert next_metrics["rollout_client/0/throughput_ema"] == 2.5


def test_lora_policy_update_in_sft_keeps_teacher_model_name():
    """In sft mode, train_pool is the teacher. LoRA updates the student inference
    pool but must not change scheduler.model_name (which is what gets sent to the
    teacher endpoint on each rollout request)."""

    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.model_name = "teacher-model"
        scheduler.lora_name = "student-lora"

        student_inference = SimpleNamespace(
            update_weights=AsyncMock(),
            update_model_name=MagicMock(),
        )
        teacher_inference = SimpleNamespace()
        scheduler.student_inference = student_inference
        scheduler.rollout_inference = teacher_inference  # sft: train_pool != student_inference
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            await scheduler.maybe_update_policy()

        student_inference.update_weights.assert_awaited_once()
        student_inference.update_model_name.assert_called_once_with("student-lora")
        assert scheduler.model_name == "teacher-model"

    asyncio.run(run())


def test_lora_policy_update_in_rl_updates_model_name():
    """In rl/opd mode, train_pool is the student. LoRA updates redirect rollout
    requests to the new LoRA name."""

    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.model_name = "student-model"
        scheduler.lora_name = "student-lora"

        student_inference = SimpleNamespace(
            update_weights=AsyncMock(),
            update_model_name=MagicMock(),
        )
        scheduler.student_inference = student_inference
        scheduler.rollout_inference = student_inference  # rl/opd: same pool
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            await scheduler.maybe_update_policy()

        student_inference.update_weights.assert_awaited_once()
        student_inference.update_model_name.assert_called_once_with("student-lora")
        assert scheduler.model_name == "student-lora"

    asyncio.run(run())


def test_schedule_rollout_uses_train_pool():
    """schedule_rollout dispatches to train_pool's clients with train_pool's model name."""

    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.model_name = "teacher-model"
        teacher_client = vf.ClientConfig(api_base_url="http://teacher.example/v1")
        env = SimpleNamespace(
            requires_group_scoring=False,
            run_rollout=AsyncMock(return_value=[]),
        )
        scheduler.rollout_inference = SimpleNamespace(train_clients=[teacher_client])
        scheduler.train_envs = SimpleNamespace(get=MagicMock(return_value=env))
        scheduler.groups = {
            0: GroupState(
                example={"env_name": "math", "example_id": "ex-1"},
                rollouts_to_schedule=1,
            )
        }

        await scheduler.schedule_rollout(group_id=0)
        await asyncio.gather(*scheduler.inflight_requests)

        env.run_rollout.assert_awaited_once_with(
            client=teacher_client,
            example={"env_name": "math", "example_id": "ex-1"},
            model_name="teacher-model",
            cache_salt="7",
        )
        assert scheduler.groups[0].pinned_client is teacher_client

    asyncio.run(run())
