import pytest
import torch

from prime_rl.inference.vllm.worker.weight_transfer import load_sparse_delta_weights
from prime_rl.utils.delta import ModelDeltaManager


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_load_sparse_delta_weights_applies_delta(tmp_path, streaming, dtype) -> None:
    module = torch.nn.Linear(2, 2, bias=True, dtype=dtype)
    with torch.no_grad():
        module.weight.fill_(1.0)
        module.bias.zero_()

    base = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    target = {"weight": torch.tensor([[-0.001, 1.0], [1.0, 2.0]], dtype=dtype), "bias": torch.ones(2, dtype=dtype)}
    delta_path = tmp_path / "delta.safetensors"
    manager = ModelDeltaManager()
    extract = (
        manager.extract_sparse_delta_streaming_from_state_dicts
        if streaming
        else manager.extract_sparse_delta_from_state_dicts
    )
    extract(base, target, delta_path)

    load_sparse_delta_weights(module, delta_path.as_posix())

    assert torch.equal(module.weight, target["weight"])
    assert torch.equal(module.bias, target["bias"])


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
