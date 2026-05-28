import shutil
import time
from pathlib import Path
from typing import Literal

import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig, LoRAConfig
from prime_rl.trainer.lora import save_lora_config
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import maybe_clean
from prime_rl.trainer.weights import (
    gather_weights_on_master,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
from prime_rl.utils.delta import ModelDeltaManager
from prime_rl.utils.utils import get_broadcast_dir, get_step_path


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via shared filesystem."""

    def __init__(
        self, output_dir: Path, config: FileSystemWeightBroadcastConfig, lora_config: LoRAConfig | None = None
    ):
        super().__init__(output_dir, lora_config)
        self.save_format: Literal["safetensors", "torch"] = config.save_format
        self.save_sharded = config.save_sharded if lora_config is None else False
        self.mode = config.mode
        self.delta_index_encoding = config.delta_index_encoding
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.delta_manager = ModelDeltaManager() if self.mode == "delta" else None
        self._initial_state: dict[str, Tensor] | None = None
        self._prev_state_by_run: dict[int, dict[str, Tensor]] = {}
        if self.mode == "delta" and lora_config is not None:
            raise ValueError("filesystem delta weight broadcast does not support LoRA yet")
        self.logger.debug(
            f"Filesystem broadcast initialized (mode={self.mode}, save_format={config.save_format}, "
            f"save_sharded={self.save_sharded})"
        )

    def initialize(self, model: nn.Module, step: int = 0) -> None:
        if self.mode != "delta":
            return

        state_dict = self._gather_hf_state(model)
        self._initial_state = state_dict
        for idx in self.multi_run_manager.used_idxs:
            self._prev_state_by_run[idx] = state_dict

            if step != 0:
                self.logger.warning(
                    f"Delta filesystem broadcast initialized at step {step}; using current weights as the delta base"
                )
                continue

            if self.world.is_master:
                save_dir = get_step_path(get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)), 0)
                save_dir.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Saving delta base weights for run {idx} to {save_dir}")
                save_state_dict(dict(state_dict), save_dir, self.save_format, self.save_sharded)
                self._notify_orchestrator(save_dir)

    def _gather_hf_state(self, model: nn.Module) -> dict[str, Tensor]:
        state_dict = gather_weights_on_master(model, is_master=self.world.is_master)
        if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
            model.convert_to_hf(state_dict)
        else:
            # For regular transformers models, revert internal format to original HF hub format
            from transformers.core_model_loading import revert_weight_conversion

            state_dict = revert_weight_conversion(model, state_dict)
        return state_dict

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via shared filesystem")
        start_time = time.perf_counter()
        adapter_only = self.lora_config is not None

        if self.mode == "delta" and adapter_only:
            raise ValueError("filesystem delta weight broadcast does not support LoRA yet")

        state_dict = {} if adapter_only else self._gather_hf_state(model)

        for idx in self.multi_run_manager.ready_to_update_idxs:
            self.logger.debug(
                f"Broadcasting weights for run {idx} (ready_to_update={self.multi_run_manager.ready_to_update[idx]})"
            )

            if adapter_only:
                # For adapter-only, MultiRunManager creates state dict directly for each run
                # All ranks must participate in DTensor gathering, but only master saves
                state_dict = self.multi_run_manager.get_state_dict_for_run(idx)
                for key, value in state_dict.items():
                    if isinstance(value, DTensor):
                        value = value.full_tensor()
                    if self.world.is_master:
                        state_dict[key] = value.to("cpu", non_blocking=False)

            # TODO: Broadcast ready to update in sync, then we dont need to gather on not ready
            if self.world.is_master:
                try:
                    save_dir = get_step_path(
                        get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                        self.multi_run_manager.progress[idx].step,
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)

                    if self.mode == "delta":
                        self._save_delta_for_run(idx, state_dict, save_dir)
                    else:
                        self.logger.debug(f"Saving weights for run {idx} to {save_dir}")
                        save_state_dict(
                            dict(state_dict), save_dir, self.save_format, self.save_sharded, adapter=adapter_only
                        )
                        if adapter_only:
                            orch_lora = self.multi_run_manager.config[idx].student.model.lora
                            save_lora_config(
                                model,
                                save_dir,
                                rank=orch_lora.rank,
                                alpha=orch_lora.alpha,
                                dropout=self.lora_config.dropout,
                            )

                    self._notify_orchestrator(save_dir)

                    # If the run is deleted, remove the run directory
                    # This is avoid the creation of zombie runs when the directory is deleted while we are broadcasting which recreates the directory
                    if self.multi_run_manager.get_orchestrator_config(self.multi_run_manager.idx_2_id[idx]) is None:
                        shutil.rmtree(self.multi_run_manager.get_run_dir(idx))

                except FileNotFoundError:
                    self.logger.warning(f"Run {idx} is deleted, skipping")
                except Exception:
                    self.logger.opt(exception=True).error(f"Error broadcasting weights for run {idx}")
                    raise
                finally:
                    self.multi_run_manager.ready_to_update[idx] = False

        if self.world.is_master:
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _save_delta_for_run(self, idx: int, state_dict: dict[str, Tensor], save_dir: Path) -> None:
        if self.delta_manager is None:
            raise RuntimeError("delta manager is not initialized")

        base_state = self._prev_state_by_run.get(idx)
        if base_state is None:
            if self._initial_state is None:
                raise RuntimeError("delta broadcast has no base state; initialize() must run before broadcast_weights()")
            self.logger.warning(f"Delta base for run {idx} was not registered during initialize(); using initial weights")
            base_state = self._initial_state

        delta_path = save_dir / "delta.safetensors"
        self.logger.debug(f"Saving sparse delta for run {idx} to {delta_path}")
        self.delta_manager.extract_sparse_delta_from_state_dicts(
            base_state,
            state_dict,
            delta_path,
            index_encoding=self.delta_index_encoding,
            save_stats=True,
        )
        self._prev_state_by_run[idx] = state_dict

    def _notify_orchestrator(self, save_dir: Path):
        """Notify the orchestrator that the weights have been broadcast by writing a 'STABLE' file to a shared filesystem."""
        stable_file = save_dir / "STABLE"
        stable_file.touch()

    def maybe_clean(self, interval_to_keep: int | None):
        for idx in self.multi_run_manager.used_idxs:
            maybe_clean(
                get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                self.multi_run_manager.progress[idx].step,
                interval_to_keep,
            )
