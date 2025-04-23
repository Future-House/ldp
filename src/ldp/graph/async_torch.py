__all__ = ["AsyncTorchModule", "async_protect_torch_call"]

import asyncio
import operator
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any
from uuid import UUID, uuid4

try:
    import torch
    from torch import nn
    from torch.utils.data import default_collate
except ImportError:
    raise ImportError(
        "ldp.graph.async_torch requires PyTorch as a dependency. "
        "Please run `pip install ldp[nn]`."
    ) from None

_TORCH_LOCK = asyncio.Lock()

# Supported devices here: https://pytorch.org/docs/stable/amp.html#torch.autocast
_AUTOCAST_DEVICES = {"cpu", "cuda", "hpu", "xpu"}


def _get_autocast_context(dtype: torch.dtype | None, device_type: str):
    return (
        nullcontext()
        if dtype is None or device_type not in _AUTOCAST_DEVICES
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


class AsyncBufferedWorker(ABC):
    """Abstract class for a worker that buffers inputs and processes them in batches."""

    def __init__(
        self,
        batch_size: int,
        max_wait_interval: float,
        collate_fn: Callable = lambda x: x,
        decollate_fn: Callable = lambda x: x,
    ):
        """Initialize.

        Args:
            batch_size: The target batch size to use when calling the module. As soon as
                batch_size calls are made, a forward pass is executed.
            max_wait_interval: The maximum time (sec) to wait for a batch to fill up
                before executing the calls we have buffered.
            collate_fn: A function to pre-process a list of inputs into a batch. Defaults to a
                no-op.
            decollate_fn: Kind of like the opposite of collate_fn. This function should take
                 the batched output and return an ordered list of outputs. Defaults to no-op.
        """
        self.batch_size = batch_size
        self.timeout = max_wait_interval
        self.collate_fn = collate_fn
        self.decollate_fn = decollate_fn

        self._work_buffer: list[tuple[float, UUID, dict[str, Any]]] = []
        self._result_buffer: dict[UUID, Any] = {}
        self._lock = asyncio.Lock()
        self._exception_raised: Exception | None = None

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
                if request_id in self._result_buffer:
                    # Our request was fulfilled by this or another coroutine!
                    return self._result_buffer.pop(request_id)

                # Try to run a batch.
                try:
                    await self._maybe_process_batch()
                except Exception as e:
                    self._exception_raised = e
                    raise

            # Sleep, to let another coroutine take over if it needs to
            await asyncio.sleep(0.0)

    async def _maybe_process_batch(self) -> None:
        """If the buffer is >= batch size or we have been waiting long enough, process the oldest batch.

        If neither condition is met, do nothing.
        """
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

            # Construct the batch inputs
            sample_kwargs = [x[2] for x in batch]
            batch_kwargs = self.collate_fn(sample_kwargs)

            batched_results = await self._batched_call(batch_kwargs)
            request_ids = [x[1] for x in batch]
            results = self.decollate_fn(batched_results)
            self._result_buffer.update(zip(request_ids, results, strict=True))

    @abstractmethod
    async def _batched_call(self, batch_kwargs: dict[str, Any]) -> Any:
        """Logic to call the worker on a batch of inputs."""


class AsyncTorchModule(AsyncBufferedWorker):
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
        ])  # fmt: skip
        ```
        In the above example, note that we are making 10 calls with a batch size of 4.
        The first two groups of 4 will be batched and executed as they arrive. The last 2
        will wait for max_wait_interval and then execute.

        NOTE: This module is not thread-safe and currently always operates in no_grad() mode.
        It may be possible to relax the latter constraint.

        Args:
            module: The PyTorch module to wrap.
            batch_size: See parent class.
            max_wait_interval: See parent class.
            collate_fn: A PyTorch collate function to use when batching inputs. Defaults to
                the PyTorch default_collate.
            decollate_fn: Kind of like the opposite of collate_fn. This function should take
                 the batched output and return an ordered list of outputs. Defaults to list.
            module_call_fn: Function that allows for customizing the call to the module.
        """
        super().__init__(
            batch_size=batch_size,
            max_wait_interval=max_wait_interval,
            collate_fn=collate_fn,
            decollate_fn=decollate_fn,
        )
        self.module = module
        self.module_call_fn = module_call_fn

    def _get_dtype_and_device(self) -> tuple[torch.dtype, torch.device]:
        param = next(self.module.parameters())
        return param.dtype, param.device

    async def _batched_call(self, batch_kwargs: dict[str, Any]):
        # Call the module and store results
        # To be safe, set _TORCH_LOCK to prevent other
        # coroutines from messing with torch state while running.
        async with _TORCH_LOCK:
            dtype, device = self._get_dtype_and_device()
            with torch.no_grad(), _get_autocast_context(dtype, device.type):
                return self.module_call_fn(self.module, **batch_kwargs)
