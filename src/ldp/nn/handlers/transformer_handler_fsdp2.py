from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

try:
    assert torch.__version__ >= "2.6.0", "FSDP2 requires PyTorch 2.6.0 or higher"
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        OffloadPolicy,
        fully_shard,
        register_fsdp_forward_method,
    )
except (ImportError, AssertionError) as e:
    raise ImportError(f"FSDP2 requires PyTorch 2.6.0 or higher: {e}") from e

from ldp.nn.lm_config import TorchDType

from .transformer_handler import (
    FSDPConfig,
    ParallelAsyncTransformer,
    ParallelTransformerHandler,
)

logger = logging.getLogger(__name__)


class FSDP2ParallelTransformerHandler(ParallelTransformerHandler):
    def _setup_fsdp(self):
        """Set up FSDP2 module wrapping."""
        if not dist.is_initialized():
            # For single device usage, just move to device directly
            device = self.config.lm_config.device
            self.module = self.module.to(device)
            return

        # Setup mixed precision policy if needed
        mp_policy = None
        logger.info(f"Setting up FSDP2 with dtype {self.config.lm_config.dtype}")
        if self.config.lm_config.dtype == TorchDType.bf16:
            logger.info("Setting up FSDP2 with bfloat16 dtype")
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
            )

        # Apply FSDP2 wrapping using new API
        fsdp_config = self.config.parallel_mode_config or FSDPConfig()
        offload_policy = (
            CPUOffloadPolicy(pin_memory=True)
            if fsdp_config.offload_cpu
            else OffloadPolicy()
        )

        self.module = fully_shard(
            self.module,
            mesh=None,  # Maybe we activate it later, see https://pytorch.org/docs/stable/distributed.html#torch.distributed.device_mesh.DeviceMesh
            reshard_after_forward=fsdp_config.reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )

        # Register model.generate as an FSDP forward method to handle generation correctly
        register_fsdp_forward_method(self.module, "generate")

    def load_checkpoint(self, ckpt: os.PathLike | str, **kwargs) -> None:
        logger.info(f'Loading checkpoint from "{ckpt}"')
        start_time = time.perf_counter()
        ckpt_path = Path(ckpt)
        if ckpt_path.is_dir():
            # Assume it's a directory containing sharded state dict
            state_dict = torch.load(
                ckpt_path / f"rank{self.local_rank}_checkpoint.pt", map_location="cpu"
            )
            self.module.load_state_dict(state_dict)
        else:
            # Assume it's a single file containing a full state dict
            state_dict = torch.load(ckpt, map_location="cpu")
            # Load the state dict - will automatically handle the sharding
            self.module.load_state_dict(state_dict)

        self.barrier()
        logger.info(f"Loading checkpoint took {time.perf_counter() - start_time:.2f}s")

    def save_checkpoint(self, ckpt: os.PathLike | str, **kwargs) -> None:
        start_time = time.perf_counter()
        ckpt_path = Path(ckpt)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        state_dict = self.module.state_dict()
        if dist.is_initialized():
            torch.save(state_dict, ckpt_path / f"rank{self.local_rank}_checkpoint.pt")
        else:
            torch.save(state_dict, ckpt_path / "checkpoint.pt")

        self.barrier()
        logger.info(f"Saving checkpoint took {time.perf_counter() - start_time:.2f}s")


class FSDP2ParallelAsyncTransformer(ParallelAsyncTransformer):
    def _get_parallel_transformer_handler_cls(self):
        return FSDP2ParallelTransformerHandler

    def state_dict(self, **kwargs) -> dict[str, torch.Tensor]:
        """Get consolidated state dict from all workers.

        With FSDP2, we need to manually consolidate the state dict
        """

        def state_dict_worker(
            handler: ParallelTransformerHandler,
        ) -> dict[str, torch.Tensor]:
            state_dict = handler.module.state_dict()
            # Convert DTensors to full tensors
            for key, tensor in state_dict.items():
                if hasattr(tensor, "full_tensor"):
                    state_dict[key] = tensor.full_tensor()
            return state_dict

        # Only need the state dict from rank 0
        state_dict = self._submit_and_gather(state_dict_worker, **kwargs)[0]
        return {k: v.cpu() for k, v in state_dict.items()}
