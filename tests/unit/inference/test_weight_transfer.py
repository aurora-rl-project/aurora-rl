import torch

from prime_rl.inference.vllm.worker.weight_transfer import load_sparse_delta_weights
from prime_rl.utils.delta import ModelDeltaManager


def test_load_sparse_delta_weights_applies_delta(tmp_path) -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.zero_()

    base = {"weight": module.weight.detach().clone()}
    target = {"weight": torch.tensor([[0.0, 1.0], [0.0, 2.0]])}
    delta_path = tmp_path / "delta.safetensors"
    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, target["weight"])


def test_load_sparse_delta_weights_applies_full_tensor_delta(tmp_path) -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.zero_()

    base = {"weight": module.weight.detach().clone()}
    target = {"weight": torch.ones_like(module.weight)}
    delta_path = tmp_path / "delta.safetensors"
    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, target["weight"])


def test_load_sparse_delta_weights_accepts_empty_delta(tmp_path) -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(1.0)

    base = {"weight": module.weight.detach().clone()}
    target = {"weight": module.weight.detach().clone()}
    delta_path = tmp_path / "delta.safetensors"
    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, target["weight"])


def test_load_sparse_delta_weights_applies_streaming_delta(tmp_path) -> None:
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.zero_()

    base = {"weight": module.weight.detach().clone()}
    target = {"weight": torch.tensor([[0.0, 1.0], [0.0, 2.0]])}
    delta_path = tmp_path / "delta.stream"
    ModelDeltaManager().extract_sparse_delta_streaming_from_state_dicts(base, target, delta_path)

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, target["weight"])
