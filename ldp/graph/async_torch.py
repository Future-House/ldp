__all__ = ["AsyncTorchModule", "async_protect_torch_call"]

import asyncio
import operator
import time
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any
from uuid import UUID, uuid4

import torch
from torch import nn
from torch.nn.functional import pad
from torch.utils.data import default_collate
from transformers.generation.utils import GenerateDecoderOnlyOutput

_TORCH_LOCK = asyncio.Lock()


def _get_autocast_context(dtype: torch.dtype | None, device_type):
    return (
        nullcontext()
        if dtype is None
        else torch.autocast(dtype=dtype, device_type=device_type)
    )


def _get_grad_context(no_grad: bool):
    return torch.no_grad() if no_grad else nullcontext()


def async_protect_torch_call(
    module: nn.Module,
    module_call_fn: Callable = lambda m, *args, **kwargs: m(*args, **kwargs),
    no_grad: bool = False,
    autocast_dtype: torch.dtype | None = None,
    autocast_device_type=None,
) -> Callable:
    async def wrapped_call(*args, **kwargs):
        async with _TORCH_LOCK:
            with (
                _get_grad_context(no_grad),
                _get_autocast_context(autocast_dtype, autocast_device_type),
            ):
                return module_call_fn(module, *args, **kwargs)

    return wrapped_call


# TODO: make max_wait_interval adaptive. We can use a heuristic like
# half the average time for a single call. If it's not provided, enable
# adaptive mode.


class AsyncTorchModule:
    def __init__(
        self,
        module: nn.Module,
        batch_size: int,
        max_wait_interval: float,
        collate_fn: Callable = default_collate,
        decollate_fn: Callable = list,
        module_call_fn: Callable = lambda m, *args, **kwargs: m(*args, **kwargs),
    ):
        """A wrapper around a torch.nn.Module that allows for async calls.

        Usage:
        ```python
        my_model = nn.Linear(2, 2)
        async_model = AsyncTorchModule(my_model, batch_size=4, max_wait_interval=0.01)

        result = await asyncio.gather(*[
            async_model(input=torch.rand(2)) for _ in range(10)
        ])
        ```
        In the above example, note that we are making 10 calls with a batch size of 4.
        The first two groups of 4 will be batched and executed as they arrive. The last 2
        will wait for max_wait_interval and then execute.

        NOTE: This module is not thread-safe and currently always operates in no_grad() mode.
        It may be possible to relax the latter constraint.

        Args:
            module: The PyTorch module to wrap.
            batch_size: The target batch size to use when calling the module. As soon as
                batch_size calls are made, a forward pass is executed.
            max_wait_interval: The maximum time to wait for a batch to fill up before
                executing the calls we have buffered.
            collate_fn: A PyTorch collate function to use when batching inputs. Defaults to
                the PyTorch default_collate.
            decollate_fn: Kind of like the opposite of collate_fn. This function should take
                 the batched output and return an ordered list of outputs. Defaults to list.
            module_call_fn: Function that allows for customizing the call to the module.
        """
        self.module = module
        self.batch_size = batch_size
        self.timeout = max_wait_interval
        self.collate_fn = collate_fn
        self.decollate_fn = decollate_fn
        self.module_call_fn = module_call_fn

        self._work_buffer: list[tuple[float, UUID, dict[str, Any]]] = []
        self._result_buffer: dict[UUID, Any] = {}
        self._lock = asyncio.Lock()

    def _get_dtype_and_device(self) -> tuple[torch.dtype, torch.device]:
        param = next(self.module.parameters())
        return param.dtype, param.device

    async def __call__(self, **kwargs):
        request_id = uuid4()
        request_ts = time.time()

        async with self._lock:
            # Make sure only one coroutine is using the work buffer at a time
            self._work_buffer.append((request_ts, request_id, kwargs))

        while True:
            async with self._lock:
                # Only one coroutine allowed in here when:
                # - modifying the result buffer
                # - modifying the work buffer
                # - calling the module (handled by _TORCH_LOCK)

                if request_id in self._result_buffer:
                    # Our request was fulfilled by this or another coroutine!
                    return self._result_buffer.pop(request_id)

                # Try to run a batch
                await self._batched_call()

            # Sleep, to let another coroutine take over if it needs to
            await asyncio.sleep(0.0)

    async def _batched_call(self):
        now = time.time()

        # sort by oldest requests first
        self._work_buffer.sort(key=operator.itemgetter(0))

        if (
            len(self._work_buffer) >= self.batch_size
            or now - self._work_buffer[0][0] > self.timeout
        ):
            # if we're over batch size or have at least one input waiting for
            # more than timeout, pull out a batch to run
            batch = self._work_buffer[: self.batch_size]
            self._work_buffer = self._work_buffer[self.batch_size :]

            # Construct the batch tensors
            sample_kwargs = [x[2] for x in batch]
            batch_kwargs = self.collate_fn(sample_kwargs)

            # Wrap the forward call to be async-safe using the options we want
            dtype, device = self._get_dtype_and_device()
            protected_call = async_protect_torch_call(
                self.module,
                module_call_fn=self.module_call_fn,
                no_grad=True,
                autocast_dtype=dtype,
                autocast_device_type=device.type,
            )

            # Call the module and store results
            batched_results = await protected_call(
                **batch_kwargs,
            )
            request_ids = [x[1] for x in batch]
            results = self.decollate_fn(batched_results)
            self._result_buffer.update(zip(request_ids, results, strict=True))

    @staticmethod
    def collate_fn_transformers_model(
        samples: list[dict[str, torch.Tensor]], agg_keys: set[str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Collates and pads a batch of samples for input into a huggingface transformer model."""
        if agg_keys is None:
            agg_keys = {"input_ids", "attention_mask"}
        seq_lens = [inp["input_ids"].shape[1] for inp in samples]
        max_seq_len = max(seq_lens)
        n_pads = [max_seq_len - seq_len for seq_len in seq_lens]

        batch = {
            key: torch.cat(
                [
                    pad(inp[key], (0, n_pad), value=0)
                    for inp, n_pad in zip(samples, n_pads, strict=True)
                ],
                dim=0,
            )
            for key in agg_keys
        }

        # Treating other keys as constant kwargs params for the model
        other_keys = set(samples[0].keys()) - agg_keys
        for key in other_keys:
            for sample in samples:
                if key not in sample:
                    raise ValueError(f"Missing key {key} in sample.")
                if key in batch and batch[key] != sample[key]:
                    raise ValueError(
                        f"Constant kwarg key {key} has different values within batch."
                    )
                batch[key] = sample[key]

        return batch

    @staticmethod
    def decollate_fn_transformers_decoder(
        batched_output: GenerateDecoderOnlyOutput,
    ) -> list[GenerateDecoderOnlyOutput]:
        """Decollates a batched output from a huggingface transformer decoder."""
        batch_size = batched_output.sequences.size(0)

        return [
            GenerateDecoderOnlyOutput({
                "sequences": batched_output.sequences[i][None],
                "scores": [v[i][None] for v in batched_output.scores],
                # Ignore other keys for now
            })
            for i in range(batch_size)
        ]