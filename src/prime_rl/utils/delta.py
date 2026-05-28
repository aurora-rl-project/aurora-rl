from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

DELTA_INDEX_SUFFIX = ".__idx"
DELTA_VALUE_SUFFIX = ".__val"


@dataclass(frozen=True)
class DeltaStats:
    total_params: int
    changed_params: int
    change_percentage: float
    total_size_mb: float
    delta_size_mb: float
    compression_ratio: float


@dataclass(frozen=True)
class DeltaMismatch:
    name: str
    max_diff: float


@dataclass(frozen=True)
class DeltaVerificationResult:
    max_diff: float
    mismatches: tuple[DeltaMismatch, ...]

    @property
    def ok(self) -> bool:
        return len(self.mismatches) == 0


@dataclass(frozen=True)
class FusedPartRule:
    source_suffix: str
    fused_suffix: str
    order: int
    parts: int


@dataclass(frozen=True)
class MappedTensorName:
    name: str
    part_order: int | None
    parts: int | None


class TensorStore(Protocol):
    def keys(self) -> Iterable[str]: ...

    def get_tensor(self, name: str) -> torch.Tensor: ...


class StateDictTensorStore:
    def __init__(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        self._state_dict = state_dict

    def keys(self) -> Iterable[str]:
        return self._state_dict.keys()

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._state_dict[name]


FUSED_WEIGHT_RULES = (
    FusedPartRule(".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight", 0, 3),
    FusedPartRule(".self_attn.k_proj.weight", ".self_attn.qkv_proj.weight", 1, 3),
    FusedPartRule(".self_attn.v_proj.weight", ".self_attn.qkv_proj.weight", 2, 3),
    FusedPartRule(".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight", 0, 2),
    FusedPartRule(".mlp.up_proj.weight", ".mlp.gate_up_proj.weight", 1, 2),
)
FUSED_BIAS_RULES = (
    FusedPartRule(".self_attn.q_proj.bias", ".self_attn.qkv_proj.bias", 0, 3),
    FusedPartRule(".self_attn.k_proj.bias", ".self_attn.qkv_proj.bias", 1, 3),
    FusedPartRule(".self_attn.v_proj.bias", ".self_attn.qkv_proj.bias", 2, 3),
    FusedPartRule(".mlp.gate_proj.bias", ".mlp.gate_up_proj.bias", 0, 2),
    FusedPartRule(".mlp.up_proj.bias", ".mlp.gate_up_proj.bias", 1, 2),
)


def encode_varint_delta_indices(indices: torch.Tensor) -> torch.Tensor:
    indices = indices.reshape(-1).detach().to(torch.int64).cpu()
    if indices.numel() == 0:
        return torch.empty((0,), dtype=torch.uint8)
    if bool((indices < 0).any().item()):
        raise ValueError("sparse delta indices must be non-negative")
    if indices.numel() > 1 and bool((indices[1:] < indices[:-1]).any().item()):
        raise ValueError("varint delta encoding expects sorted indices")

    encoded = bytearray()
    previous = 0
    for current in indices.tolist():
        diff = int(current) - previous
        while diff >= 0x80:
            encoded.append((diff & 0x7F) | 0x80)
            diff >>= 7
        encoded.append(diff)
        previous = int(current)

    return torch.tensor(list(encoded), dtype=torch.uint8)


def decode_varint_delta_indices(index_bytes: torch.Tensor, expected_count: int) -> torch.Tensor:
    if index_bytes.dtype != torch.uint8:
        raise ValueError("varint delta indices must be stored as uint8")
    if expected_count == 0:
        if index_bytes.numel() != 0:
            raise ValueError("expected no indices, but varint stream is not empty")
        return torch.empty((0,), dtype=torch.int64)
    index_bytes = index_bytes.detach().cpu().reshape(-1)
    if index_bytes.numel() == expected_count and not bool((index_bytes & 0x80).any().item()):
        return index_bytes.to(torch.int64).cumsum(dim=0)

    return _decode_varint_delta_indices_numpy(index_bytes, expected_count)


def _decode_varint_delta_indices_numpy(index_bytes: torch.Tensor, expected_count: int) -> torch.Tensor:
    encoded = index_bytes.numpy()
    is_terminal = (encoded & 0x80) == 0
    end_positions = np.flatnonzero(is_terminal)

    if end_positions.size != expected_count:
        if encoded.size > 0 and not bool(is_terminal[-1]):
            raise ValueError("truncated varint stream")
        if end_positions.size > expected_count:
            raise ValueError("decoded more indices than expected")
        raise ValueError(f"decoded index count mismatch: {end_positions.size} vs {expected_count}")

    start_positions = np.empty_like(end_positions)
    start_positions[0] = 0
    start_positions[1:] = end_positions[:-1] + 1
    lengths = end_positions - start_positions + 1
    max_length = int(lengths.max(initial=0))
    if max_length > 9:
        raise ValueError("varint stream contains an index delta that does not fit int64")

    payload = encoded & 0x7F
    diffs = np.zeros(expected_count, dtype=np.int64)
    for offset in range(max_length):
        active = lengths > offset
        byte_positions = start_positions[active] + offset
        diffs[active] |= payload[byte_positions].astype(np.int64) << (7 * offset)

    indices = np.cumsum(diffs, dtype=np.int64)
    return torch.from_numpy(indices)


def _decode_varint_delta_indices_python(index_bytes: torch.Tensor, expected_count: int) -> torch.Tensor:
    if expected_count == 0:
        if index_bytes.numel() != 0:
            raise ValueError("expected no indices, but varint stream is not empty")
        return torch.empty((0,), dtype=torch.int64)

    out = torch.empty((expected_count,), dtype=torch.int64)
    out_index = 0
    current = 0
    diff = 0
    shift = 0

    for byte in memoryview(index_bytes.numpy()):
        diff |= (int(byte) & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue

        if out_index >= expected_count:
            raise ValueError("decoded more indices than expected")
        current += diff
        out[out_index] = current
        out_index += 1
        diff = 0
        shift = 0

    if shift != 0:
        raise ValueError("truncated varint stream")
    if out_index != expected_count:
        raise ValueError(f"decoded index count mismatch: {out_index} vs {expected_count}")
    return out


def encode_sparse_indices(indices: torch.Tensor, *, encoding: str = "optimized") -> torch.Tensor:
    indices = indices.reshape(-1).detach().to(torch.int64).cpu()
    if encoding == "optimized":
        return encode_varint_delta_indices(indices)
    if encoding == "naive":
        if indices.numel() == 0:
            return indices.to(torch.int32)
        return indices.to(torch.int32 if int(indices.max().item()) < 2**31 else torch.int64)
    raise ValueError(f"unsupported sparse index encoding: {encoding}")


def decode_sparse_indices(encoded_indices: torch.Tensor, value_count: int) -> torch.Tensor:
    if encoded_indices.dtype == torch.uint8:
        return decode_varint_delta_indices(encoded_indices, value_count)
    if encoded_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"unsupported sparse index dtype: {encoded_indices.dtype}")
    indices = encoded_indices.reshape(-1).to(torch.int64)
    if indices.numel() != value_count:
        raise ValueError(f"index/value length mismatch: {indices.numel()} vs {value_count}")
    return indices


def map_tensor_name_to_fused(name: str, *, include_bias: bool = False) -> MappedTensorName:
    for rule in FUSED_WEIGHT_RULES:
        if name.endswith(rule.source_suffix):
            return MappedTensorName(
                name=name.replace(rule.source_suffix, rule.fused_suffix),
                part_order=rule.order,
                parts=rule.parts,
            )
    if include_bias:
        for rule in FUSED_BIAS_RULES:
            if name.endswith(rule.source_suffix):
                return MappedTensorName(
                    name=name.replace(rule.source_suffix, rule.fused_suffix),
                    part_order=rule.order,
                    parts=rule.parts,
                )
    return MappedTensorName(name=name, part_order=None, parts=None)


def collect_tied_lm_head_skip_keys(store: TensorStore) -> set[str]:
    keys = set(store.keys())
    if "model.embed_tokens.weight" not in keys or "lm_head.weight" not in keys:
        return set()

    embedding = store.get_tensor("model.embed_tokens.weight")
    lm_head = store.get_tensor("lm_head.weight")
    if embedding.shape == lm_head.shape and torch.equal(embedding, lm_head):
        return {"lm_head.weight"}
    return set()


def build_fused_groups(
    keys: Iterable[str],
    *,
    include_bias: bool = False,
) -> tuple[dict[str, dict[int, str]], dict[str, int], set[str]]:
    fused_parts: dict[str, dict[int, str]] = defaultdict(dict)
    parts_expected: dict[str, int] = {}
    single_keys: set[str] = set()

    for key in keys:
        mapped = map_tensor_name_to_fused(key, include_bias=include_bias)
        if mapped.part_order is None:
            single_keys.add(key)
            continue

        if mapped.part_order in fused_parts[mapped.name]:
            raise ValueError(f"duplicate fused part {mapped.part_order} for {mapped.name}")
        fused_parts[mapped.name][mapped.part_order] = key
        parts_expected.setdefault(mapped.name, mapped.parts or 0)
        if parts_expected[mapped.name] != mapped.parts:
            raise ValueError(f"parts mismatch for {mapped.name}: {parts_expected[mapped.name]} vs {mapped.parts}")

    for fused_name, parts in list(fused_parts.items()):
        _validate_part_orders(fused_name, parts.keys(), parts_expected[fused_name])
        single_keys.discard(fused_name)

    return dict(fused_parts), parts_expected, single_keys


def load_fused_tensor(store: TensorStore, fused_name: str, parts: Mapping[int, str]) -> torch.Tensor:
    _validate_part_orders(fused_name, parts.keys(), len(parts))
    tensors = [store.get_tensor(parts[order]) for order in sorted(parts)]
    _validate_fused_shapes(fused_name, {order: tensor.shape for order, tensor in zip(sorted(parts), tensors)})
    return torch.cat(tensors, dim=0)


def reconstruct_sparse_delta_tensor(
    base_tensor: torch.Tensor,
    encoded_indices: torch.Tensor,
    values: torch.Tensor,
    *,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    indices = decode_sparse_indices(encoded_indices, values.numel())
    flat = base_tensor.reshape(-1).clone()
    if indices.numel() == 0:
        return flat.reshape_as(base_tensor)
    if int(indices.min().item()) < 0 or int(indices.max().item()) >= flat.numel():
        raise ValueError("sparse delta index out of range")

    indices = indices.to(device=flat.device)
    delta_values = values.reshape(-1).to(device=flat.device, dtype=base_tensor.dtype)
    if scale_factor != 1.0:
        delta_values = delta_values * scale_factor
    flat.index_add_(0, indices, delta_values)
    return flat.reshape_as(base_tensor)


def verify_sparse_delta_file(
    base_model_path: str | Path,
    target_model_path: str | Path,
    delta_path: str | Path,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    max_report: int = 10,
    include_bias: bool = False,
) -> DeltaVerificationResult:
    with (
        safe_open(base_model_path, framework="pt", device="cpu") as base,
        safe_open(target_model_path, framework="pt", device="cpu") as target,
        safe_open(delta_path, framework="pt", device="cpu") as delta,
    ):
        return verify_sparse_delta_stores(
            base,
            target,
            delta,
            atol=atol,
            rtol=rtol,
            max_report=max_report,
            include_bias=include_bias,
        )


def verify_sparse_delta_state_dicts(
    base_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    delta_path: str | Path,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    max_report: int = 10,
    include_bias: bool = False,
) -> DeltaVerificationResult:
    base = StateDictTensorStore(base_state)
    target = StateDictTensorStore(target_state)
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        return verify_sparse_delta_stores(
            base,
            target,
            delta,
            atol=atol,
            rtol=rtol,
            max_report=max_report,
            include_bias=include_bias,
        )


def verify_sparse_delta_stores(
    base: TensorStore,
    target: TensorStore,
    delta: TensorStore,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    max_report: int = 10,
    include_bias: bool = False,
) -> DeltaVerificationResult:
    base_keys = set(base.keys())
    target_keys = set(target.keys())
    delta_keys = set(delta.keys())

    if _looks_like_delta(base_keys):
        raise ValueError("--base looks like a sparse delta file")
    if _looks_like_delta(target_keys):
        raise ValueError("--target looks like a sparse delta file")
    if not _looks_like_delta(delta_keys):
        raise ValueError("--delta does not look like a sparse delta file")

    model_keys = _validated_model_keys(base, target, include_bias=include_bias)
    fused_parts, _parts_expected, single_keys = build_fused_groups(model_keys, include_bias=include_bias)
    all_names = set(single_keys) | set(fused_parts)

    idx_names = {key[: -len(DELTA_INDEX_SUFFIX)] for key in delta_keys if key.endswith(DELTA_INDEX_SUFFIX)}
    val_names = {key[: -len(DELTA_VALUE_SUFFIX)] for key in delta_keys if key.endswith(DELTA_VALUE_SUFFIX)}
    if idx_names != val_names:
        raise ValueError(f"delta index/value names mismatch: idx-only={idx_names - val_names}, val-only={val_names - idx_names}")
    if unexpected := idx_names - all_names:
        raise ValueError(f"delta has names not present in the model: {sorted(unexpected)[:10]}")

    max_diff = 0.0
    mismatches: list[DeltaMismatch] = []
    report_limit = max(max_report, 1)

    for name in sorted(all_names):
        if name in fused_parts:
            base_tensor = load_fused_tensor(base, name, fused_parts[name])
            target_tensor = load_fused_tensor(target, name, fused_parts[name])
        else:
            base_tensor = base.get_tensor(name)
            target_tensor = target.get_tensor(name)

        if base_tensor.shape != target_tensor.shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(base_tensor.shape)} vs {tuple(target_tensor.shape)}")

        idx_key = f"{name}{DELTA_INDEX_SUFFIX}"
        val_key = f"{name}{DELTA_VALUE_SUFFIX}"
        if name in idx_names:
            reconstructed = reconstruct_sparse_delta_tensor(
                base_tensor,
                delta.get_tensor(idx_key),
                delta.get_tensor(val_key),
            )
        else:
            reconstructed = base_tensor

        diff = (reconstructed - target_tensor).abs()
        diff_max = float(diff.max().item()) if diff.numel() else 0.0
        max_diff = max(max_diff, diff_max)
        if not torch.allclose(reconstructed, target_tensor, atol=atol, rtol=rtol) and len(mismatches) < report_limit:
            mismatches.append(DeltaMismatch(name=name, max_diff=diff_max))

    return DeltaVerificationResult(max_diff=max_diff, mismatches=tuple(mismatches))


def count_sparse_delta_values(delta_path: str | Path) -> int:
    total = 0
    with safe_open(delta_path, framework="pt", device="cpu") as delta:
        for key in delta.keys():
            if key.endswith(DELTA_VALUE_SUFFIX):
                total += delta.get_tensor(key).numel()
    return total


class ModelDeltaManager:
    def extract_sparse_delta(
        self,
        base_model_path: str | Path,
        finetuned_model_path: str | Path,
        delta_output_path: str | Path,
        *,
        index_encoding: str = "optimized",
        save_stats: bool = False,
    ) -> DeltaStats | None:
        with (
            safe_open(base_model_path, framework="pt", device="cpu") as base,
            safe_open(finetuned_model_path, framework="pt", device="cpu") as target,
        ):
            return self._extract_sparse_delta_from_stores(
                base,
                target,
                delta_output_path,
                index_encoding=index_encoding,
                include_bias=False,
                threshold=0.0,
                collect_stats=save_stats,
            )

    def extract_sparse_delta_from_state_dicts(
        self,
        base_state: Mapping[str, torch.Tensor],
        finetuned_state: Mapping[str, torch.Tensor],
        delta_output_path: str | Path,
        *,
        index_encoding: str = "optimized",
        save_stats: bool = False,
    ) -> DeltaStats | None:
        return self._extract_sparse_delta_from_stores(
            StateDictTensorStore(base_state),
            StateDictTensorStore(finetuned_state),
            delta_output_path,
            index_encoding=index_encoding,
            include_bias=False,
            threshold=0.0,
            collect_stats=save_stats,
        )

    def extract_delta_fused_sparse(
        self,
        base_model_path: str | Path,
        finetuned_model_path: str | Path,
        delta_output_path: str | Path,
        threshold: float = 0.0,
        include_bias: bool = False,
        save_stats: bool = False,
        save_delta_values: bool = False,
        plot_hist: bool = False,
        index_encoding: str = "optimized",
    ) -> dict[str, bool | float | int]:
        if save_delta_values or plot_hist:
            raise ValueError("delta value dumps and histograms are not part of the Phase 1 utility layer")

        with (
            safe_open(base_model_path, framework="pt", device="cpu") as base,
            safe_open(finetuned_model_path, framework="pt", device="cpu") as target,
        ):
            stats = self._extract_sparse_delta_from_stores(
                base,
                target,
                delta_output_path,
                index_encoding=index_encoding,
                include_bias=include_bias,
                threshold=threshold,
                collect_stats=save_stats,
            )

        summary: dict[str, bool | float | int] = {
            "threshold": float(threshold),
            "include_bias": bool(include_bias),
        }
        if stats is not None:
            summary.update(asdict(stats))
            stats_path = _stats_path(delta_output_path)
            stats_path.write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
        return summary

    def _extract_sparse_delta_from_stores(
        self,
        base: TensorStore,
        target: TensorStore,
        delta_output_path: str | Path,
        *,
        index_encoding: str,
        include_bias: bool,
        threshold: float,
        collect_stats: bool,
    ) -> DeltaStats | None:
        keys = _validated_model_keys(base, target, include_bias=include_bias)

        part_sizes: dict[str, dict[int, int]] = defaultdict(dict)
        part_shapes: dict[str, dict[int, torch.Size]] = defaultdict(dict)
        parts_expected: dict[str, int] = {}
        idx_parts: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
        val_parts: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
        idx_single: dict[str, list[torch.Tensor]] = defaultdict(list)
        val_single: dict[str, list[torch.Tensor]] = defaultdict(list)

        total_params = 0
        total_bytes = 0
        changed_params = 0

        for key in sorted(keys):
            base_tensor = base.get_tensor(key)
            target_tensor = target.get_tensor(key)
            if base_tensor.shape != target_tensor.shape:
                raise ValueError(
                    f"shape mismatch for {key}: {tuple(base_tensor.shape)} vs {tuple(target_tensor.shape)}"
                )

            if collect_stats:
                total_params += base_tensor.numel()
                total_bytes += base_tensor.numel() * base_tensor.element_size()

            mapped = map_tensor_name_to_fused(key, include_bias=include_bias)
            if mapped.part_order is not None:
                parts_expected.setdefault(mapped.name, mapped.parts or 0)
                if parts_expected[mapped.name] != mapped.parts:
                    raise ValueError(f"parts mismatch for {mapped.name}: {parts_expected[mapped.name]} vs {mapped.parts}")
                part_sizes[mapped.name][mapped.part_order] = int(base_tensor.numel())
                part_shapes[mapped.name][mapped.part_order] = base_tensor.shape

            flat_base = base_tensor.reshape(-1)
            flat_target = target_tensor.reshape(-1)
            delta = flat_target - flat_base
            mask = delta.abs() > threshold
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue

            values = delta.index_select(0, indices).contiguous().to(base_tensor.dtype)
            changed_params += int(indices.numel())

            if mapped.part_order is None:
                idx_single[mapped.name].append(indices)
                val_single[mapped.name].append(values)
            else:
                idx_parts[mapped.name][mapped.part_order].append(indices)
                val_parts[mapped.name][mapped.part_order].append(values)

        delta_tensors: dict[str, torch.Tensor] = {}

        for name in sorted(part_sizes):
            expected = parts_expected[name]
            _validate_part_orders(name, part_sizes[name].keys(), expected)
            _validate_fused_shapes(name, part_shapes[name])
            offsets = _fused_offsets(part_sizes[name], expected)

            idx_list: list[torch.Tensor] = []
            val_list: list[torch.Tensor] = []
            for order in range(expected):
                for indices, values in zip(idx_parts[name].get(order, []), val_parts[name].get(order, [])):
                    idx_list.append(indices + offsets[order])
                    val_list.append(values)

            if idx_list:
                _add_sparse_tensor(delta_tensors, name, idx_list, val_list, index_encoding=index_encoding)

        for name in sorted(idx_single):
            _add_sparse_tensor(delta_tensors, name, idx_single[name], val_single[name], index_encoding=index_encoding)

        save_file(delta_tensors, delta_output_path)
        if not collect_stats:
            return None

        delta_size_mb = os.path.getsize(delta_output_path) / (1024 * 1024)
        total_size_mb = total_bytes / (1024 * 1024)
        change_percentage = (changed_params / total_params * 100) if total_params else 0.0
        compression_ratio = total_size_mb / delta_size_mb if delta_size_mb > 0 else float("inf")
        return DeltaStats(
            total_params=total_params,
            changed_params=changed_params,
            change_percentage=change_percentage,
            total_size_mb=total_size_mb,
            delta_size_mb=delta_size_mb,
            compression_ratio=compression_ratio,
        )


def _add_sparse_tensor(
    delta_tensors: dict[str, torch.Tensor],
    name: str,
    index_chunks: list[torch.Tensor],
    value_chunks: list[torch.Tensor],
    *,
    index_encoding: str,
) -> None:
    if len(index_chunks) != len(value_chunks):
        raise ValueError(f"index/value chunks mismatch for {name}")

    indices = torch.cat(index_chunks, dim=0).to(torch.int64)
    values = torch.cat(value_chunks, dim=0)
    if indices.numel() > 1:
        order = torch.argsort(indices)
        indices = indices.index_select(0, order)
        values = values.index_select(0, order)

    delta_tensors[f"{name}{DELTA_INDEX_SUFFIX}"] = encode_sparse_indices(indices, encoding=index_encoding)
    delta_tensors[f"{name}{DELTA_VALUE_SUFFIX}"] = values


def _validated_model_keys(base: TensorStore, target: TensorStore, *, include_bias: bool) -> set[str]:
    base_keys = set(base.keys())
    target_keys = set(target.keys())
    skip_keys = collect_tied_lm_head_skip_keys(base)
    base_keys -= skip_keys
    target_keys -= skip_keys
    if not include_bias:
        base_keys = {key for key in base_keys if not key.endswith(".bias")}
        target_keys = {key for key in target_keys if not key.endswith(".bias")}
    if base_keys != target_keys:
        raise ValueError(f"keys mismatch: base-only={base_keys - target_keys}, target-only={target_keys - base_keys}")
    return base_keys


def _validate_part_orders(name: str, orders: Iterable[int], expected: int) -> None:
    actual = set(orders)
    wanted = set(range(expected))
    if actual != wanted:
        raise ValueError(f"missing fused parts for {name}: got {sorted(actual)} expected {sorted(wanted)}")


def _validate_fused_shapes(name: str, part_shapes: Mapping[int, torch.Size]) -> None:
    sorted_shapes = [part_shapes[order] for order in sorted(part_shapes)]
    if not sorted_shapes:
        return
    rank = len(sorted_shapes[0])
    trailing = tuple(sorted_shapes[0][1:])
    for shape in sorted_shapes:
        if len(shape) != rank or tuple(shape[1:]) != trailing:
            raise ValueError(f"incompatible fused part shapes for {name}: {[tuple(shape) for shape in sorted_shapes]}")


def _fused_offsets(part_sizes: Mapping[int, int], expected: int) -> dict[int, int]:
    _validate_part_orders("fused tensor", part_sizes.keys(), expected)
    offsets: dict[int, int] = {}
    running = 0
    for order in range(expected):
        offsets[order] = running
        running += part_sizes[order]
    return offsets


def _looks_like_delta(keys: Iterable[str]) -> bool:
    return any(key.endswith(DELTA_INDEX_SUFFIX) or key.endswith(DELTA_VALUE_SUFFIX) for key in keys)


def _stats_path(delta_output_path: str | Path) -> Path:
    path = Path(delta_output_path)
    suffix = "".join(path.suffixes)
    if suffix:
        return path.with_name(f"{path.name.removesuffix(suffix)}_stats.json")
    return path.with_name(f"{path.name}_stats.json")
