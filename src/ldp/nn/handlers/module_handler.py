from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from functools import wraps
from pathlib import Path
from typing import Any, Concatenate, ParamSpec, TypeVar

import torch
from torch import nn

from ldp.graph.async_torch import AsyncTorchModule

TReturn = TypeVar("TReturn")
TParams = ParamSpec("TParams")


class ModuleExecutionInterface(ABC):
    """Defines a common interface for interacting with PyTorch modules.

    Subclasses should implement @wrap_func and @wrap_afunc such that calling code can do:

    ```python
    exec_interface = MyExecutionInterface(...)


    @exec_interface.wrap_func
    def my_func(exec_interface, arg1, arg2):
        return exec_interface.module(arg1, arg2)


    @exec_interface.wrap_afunc
    async def my_afunc(exec_interface, arg1, arg2):
        return await exec_interface.module(arg1, arg2)
    ```
    """

    def wrap_afunc(
        self,
        # NOTE: type-hinting the first argument as Any so that subclasses can specialize
        # without typing errors.
        func: Callable[Concatenate[Any, TParams], Awaitable[TReturn]],
        **kwargs,
    ) -> Callable[TParams, Awaitable[TReturn]]:
        """Wraps a coroutine whose first argument is a ModuleExecutionInterface.

        By default, the coroutine is returned with the first argument
        set to this object, equivalent to `partial(func, self)`.

        Args:
            func: The coroutine to wrap.
            kwargs: Additional arguments - discarded, but may be used by subclasses.
        """

        @wraps(func)
        async def wrapped_func(*args, **kwargs):
            return await func(self, *args, **kwargs)

        return wrapped_func

    def wrap_func(
        self, func: Callable[Concatenate[Any, TParams], TReturn], **kwargs
    ) -> Callable[TParams, TReturn]:
        """Wraps a function whose first argument is a ModuleExecutionInterface.

        By default, the function is returned with the first argument
        set to this object, equivalent to `partial(func, self)`.

        Args:
            func: The function to wrap.
            kwargs: Additional arguments - discarded, but may be used by subclasses.
        """

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(self, *args, **kwargs)

        return wrapped_func

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a provided state dict."""

    @abstractmethod
    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the current state dict."""

    @abstractmethod
    def load_checkpoint(self, ckpt: os.PathLike | str) -> None:
        """Load a checkpoint from the provided path."""

    @abstractmethod
    def save_checkpoint(self, ckpt: os.PathLike | str) -> None:
        """Save a checkpoint to the provided path."""

    def teardown(self):
        """Teardown any resources used by the interface."""


class ModuleHandler(ModuleExecutionInterface):
    """Base class for an execution interface that holds a PyTorch model.

    Subclasses can wrap calls to the model for specific use-cases, such as
    batching async calls, remote execution, etc.
    """

    def __init__(self, module: nn.Module):
        self.module: nn.Module = module

        # A container for objects that are local to the current process.
        # Can be used for something like an optimizer, if self.module is
        # sharded. The handler has to own these objects because they may
        # be created remotely (e.g. w/ Dask) and cannot be pickled.
        self.process_local_objects: dict[str, Any] = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state_dict)

    def save_checkpoint(self, ckpt: os.PathLike | str, **kwargs) -> None:
        ckpt = Path(ckpt)
        if ckpt.is_dir():
            ckpt /= "model.pt"
        torch.save(self.state_dict(), ckpt)

    def load_checkpoint(self, ckpt: os.PathLike | str) -> None:
        ckpt = Path(ckpt)
        if ckpt.is_dir():
            ckpt /= "model.pt"
        self.load_state_dict(torch.load(ckpt))


class AsyncModuleHandler(ModuleHandler):
    def __init__(self, module: nn.Module, **kwargs):
        super().__init__(module)
        self.async_module = AsyncTorchModule(module, **kwargs)

    async def __call__(self, **kwargs):
        return await self.async_module(**kwargs)
