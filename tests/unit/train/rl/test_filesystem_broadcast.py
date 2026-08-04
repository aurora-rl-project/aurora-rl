from types import SimpleNamespace

import pytest
import torch

import prime_rl.trainer.rl.broadcast.filesystem as filesystem_module
from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.utils.delta import verify_sparse_delta_state_dicts


@pytest.mark.parametrize(
    ("streaming", "delta_filename"),
    [(False, "delta.safetensors"), (True, "delta.stream")],
)
def test_filesystem_delta_broadcast_writes_base_and_delta(
    tmp_path,
    monkeypatch,
    streaming: bool,
    delta_filename: str,
) -> None:
    manager = SimpleNamespace(
        used_idxs=[0],
        ready_to_update_idxs=[0],
        ready_to_update={0: True},
        progress={0: SimpleNamespace(step=1)},
        idx_2_id={0: "run-0"},
        config={},
        get_run_dir=lambda _idx: tmp_path,
        get_orchestrator_config=lambda _run_id: object(),
    )
    monkeypatch.setattr(filesystem_module, "get_world", lambda: SimpleNamespace(is_master=True))
    monkeypatch.setattr(filesystem_module, "get_multi_run_manager", lambda: manager)

    broadcast = FileSystemWeightBroadcast(
        tmp_path,
        FileSystemWeightBroadcastConfig(
            mode="delta",
            save_sharded=False,
            delta_streaming_enabled=streaming,
        ),
    )
    base = {"linear.weight": torch.zeros((2, 2))}
    target = {"linear.weight": torch.tensor([[0.0, 1.0], [0.0, 2.0]])}
    states = [base, target]
    broadcast._gather_hf_state = lambda _model: states.pop(0)  # type: ignore[method-assign]

    model = torch.nn.Linear(2, 2, bias=False)
    broadcast.initialize(model, step=0)
    broadcast.broadcast_weights(model, step=1)

    base_dir = tmp_path / "broadcasts" / "step_0"
    delta_dir = tmp_path / "broadcasts" / "step_1"
    delta_path = delta_dir / delta_filename

    assert (base_dir / "model.safetensors").exists()
    assert (base_dir / "STABLE").exists()
    assert delta_path.exists()
    assert (delta_dir / "STABLE").exists()
    assert verify_sparse_delta_state_dicts(base, target, delta_path).ok
