import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from prime_rl.utils.delta import (
    DELTA_INDEX_SUFFIX,
    DELTA_VALUE_SUFFIX,
    ModelDeltaManager,
    decode_sparse_indices,
    decode_varint_delta_indices,
    encode_varint_delta_indices,
    verify_sparse_delta_file,
    verify_sparse_delta_state_dicts,
)


def test_varint_delta_indices_round_trip() -> None:
    indices = torch.tensor([0, 1, 2, 127, 128, 129, 16_384, 100_000], dtype=torch.int64)

    encoded = encode_varint_delta_indices(indices)
    decoded = decode_varint_delta_indices(encoded, expected_count=indices.numel())

    assert encoded.dtype == torch.uint8
    assert torch.equal(decoded, indices)


def test_varint_delta_indices_fast_path_round_trip_dense_range() -> None:
    indices = torch.arange(4096, dtype=torch.int64)

    encoded = encode_varint_delta_indices(indices)
    decoded = decode_varint_delta_indices(encoded, expected_count=indices.numel())

    assert encoded.numel() == indices.numel()
    assert torch.equal(decoded, indices)


def test_varint_delta_indices_reject_unsorted_indices() -> None:
    with pytest.raises(ValueError, match="sorted"):
        encode_varint_delta_indices(torch.tensor([2, 1], dtype=torch.int64))


def test_sparse_delta_from_state_dicts_round_trips_and_skips_bias(tmp_path) -> None:
    base = {
        "linear.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "linear.bias": torch.tensor([1.0, 1.0]),
    }
    target = {
        "linear.weight": torch.tensor([[1.0, 2.5], [3.0, 6.0]]),
        "linear.bias": torch.tensor([4.0, 5.0]),
    }
    delta_path = tmp_path / "delta.safetensors"

    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    result = verify_sparse_delta_state_dicts(base, target, delta_path)
    assert result.ok
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        keys = set(delta.keys())
        assert f"linear.weight{DELTA_INDEX_SUFFIX}" in keys
        assert f"linear.weight{DELTA_VALUE_SUFFIX}" in keys
        assert f"linear.bias{DELTA_INDEX_SUFFIX}" not in keys


def test_sparse_delta_uses_fused_names_with_uneven_qkv_parts(tmp_path) -> None:
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.zeros((2, 2)),
        "model.layers.0.self_attn.k_proj.weight": torch.zeros((1, 2)),
        "model.layers.0.self_attn.v_proj.weight": torch.zeros((1, 2)),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros((1, 2)),
        "model.layers.0.mlp.up_proj.weight": torch.zeros((1, 2)),
    }
    target = {name: tensor.clone() for name, tensor in base.items()}
    target["model.layers.0.self_attn.q_proj.weight"][1, 1] = 10.0
    target["model.layers.0.self_attn.k_proj.weight"][0, 1] = 20.0
    target["model.layers.0.self_attn.v_proj.weight"][0, 0] = -1.0
    target["model.layers.0.mlp.gate_proj.weight"][0, 0] = 3.0
    target["model.layers.0.mlp.up_proj.weight"][0, 1] = 4.0
    delta_path = tmp_path / "delta.safetensors"

    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    result = verify_sparse_delta_state_dicts(base, target, delta_path)
    assert result.ok
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        keys = set(delta.keys())
        qkv_idx_key = f"model.layers.0.self_attn.qkv_proj.weight{DELTA_INDEX_SUFFIX}"
        qkv_val_key = f"model.layers.0.self_attn.qkv_proj.weight{DELTA_VALUE_SUFFIX}"
        assert qkv_idx_key in keys
        assert qkv_val_key in keys
        assert f"model.layers.0.self_attn.q_proj.weight{DELTA_INDEX_SUFFIX}" not in keys

        qkv_values = delta.get_tensor(qkv_val_key)
        qkv_indices = decode_sparse_indices(delta.get_tensor(qkv_idx_key), qkv_values.numel())
        assert qkv_indices.tolist() == [3, 5, 6]
        assert qkv_values.tolist() == [10.0, 20.0, -1.0]


def test_tied_lm_head_is_skipped(tmp_path) -> None:
    embedding = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    base = {
        "model.embed_tokens.weight": embedding,
        "lm_head.weight": embedding.clone(),
        "model.norm.weight": torch.ones(2),
    }
    target_embedding = embedding + 1.0
    target = {
        "model.embed_tokens.weight": target_embedding,
        "lm_head.weight": target_embedding.clone(),
        "model.norm.weight": torch.ones(2),
    }
    delta_path = tmp_path / "delta.safetensors"

    ModelDeltaManager().extract_sparse_delta_from_state_dicts(base, target, delta_path)

    result = verify_sparse_delta_state_dicts(base, target, delta_path)
    assert result.ok
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        keys = set(delta.keys())
        assert f"model.embed_tokens.weight{DELTA_INDEX_SUFFIX}" in keys
        assert f"lm_head.weight{DELTA_INDEX_SUFFIX}" not in keys


def test_extract_delta_fused_sparse_file_round_trip_and_stats(tmp_path) -> None:
    base = {"model.layers.0.mlp.down_proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    target = {"model.layers.0.mlp.down_proj.weight": torch.tensor([[1.0, 5.0], [3.0, 8.0]])}
    base_path = tmp_path / "base.safetensors"
    target_path = tmp_path / "target.safetensors"
    delta_path = tmp_path / "delta.safetensors"
    save_file(base, base_path)
    save_file(target, target_path)

    summary = ModelDeltaManager().extract_delta_fused_sparse(
        base_model_path=base_path,
        finetuned_model_path=target_path,
        delta_output_path=delta_path,
        save_stats=True,
        index_encoding="naive",
    )

    assert summary["changed_params"] == 2
    assert verify_sparse_delta_file(base_path, target_path, delta_path).ok
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        idx = delta.get_tensor(f"model.layers.0.mlp.down_proj.weight{DELTA_INDEX_SUFFIX}")
        assert idx.dtype == torch.int32

    stats_path = tmp_path / "delta_stats.json"
    assert json.loads(stats_path.read_text(encoding="utf-8"))["summary"]["changed_params"] == 2
