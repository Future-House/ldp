from __future__ import annotations

import asyncio
import logging
import random
import typing
from collections import UserList
from collections.abc import Awaitable, Callable, Iterator
from enum import StrEnum, auto
from itertools import chain
from typing import Any, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm

from ldp.graph import eval_mode
from ldp.graph.async_torch import AsyncTorchModule

logger = logging.getLogger(__name__)


class ReplayBuffer(UserList[dict]):
    """A base replay buffer that only allows adding and sampling."""

    @staticmethod
    def _batched_iter(
        data: list[dict],
        batch_size: int,
        shuffle: bool = True,
        infinite: bool = False,
    ) -> Iterator[dict]:
        while True:
            indices = list(range(len(data)))
            if shuffle:
                random.shuffle(indices)

            for i in range(0, len(data), batch_size):
                keys = data[0].keys()

                batch: dict[str, list] = {k: [] for k in keys}
                for j in indices[i : i + batch_size]:
                    if data[j].keys() != keys:
                        raise RuntimeError(
                            "Found buffer element with inconsistent keys"
                        )

                    for k in keys:
                        batch[k].append(data[j][k])

                yield batch

            if not infinite:
                break

    def batched_iter(
        self,
        batch_size: int,
        shuffle: bool = True,
        infinite: bool = False,
    ) -> Iterator[dict]:
        return self._batched_iter(self.data, batch_size, shuffle, infinite)

    def resize(self, size: int) -> None:
        """Optional method for the buffer to resize itself."""

    async def prepare_for_sampling(self) -> None:
        """Optional method for the buffer to prepare itself before sampling."""

    @staticmethod
    def sample_from(*buffers: ReplayBuffer, **kwargs) -> Iterator[dict]:
        """Helper method to uniformly sample from multiple buffers."""
        if any(isinstance(b, PrioritizedReplayBuffer) for b in buffers):
            # This is because PrioritizedReplayBuffer determines samples inside
            # batched_iter, so we cannot rely on buffer.data.
            raise RuntimeError(
                "sample_from does not support prioritized replay buffers"
            )

        all_buffers = list(chain.from_iterable(b.data for b in buffers))
        return ReplayBuffer._batched_iter(data=all_buffers, **kwargs)


class CircularReplayBuffer(ReplayBuffer):
    def resize(self, size: int):
        if len(self) > size:
            self.data = self.data[-size:]


class RandomizedReplayBuffer(ReplayBuffer):
    def resize(self, size: int):
        if len(self) > size:
            self.data = random.sample(self.data, size)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Implements a variant of https://arxiv.org/abs/1511.05952.

    Instead of updating the TD error on the fly, we compute it for all samples
    in the buffer before `update``. This allows us to efficiently
    batch prioritization and lets us sample w/o replacement.

    Also note that we define the TD error using the MC return, not the one-step
    return, for now. One-step may be possible using `next_state_action_cands`.

    As we expect our buffers to be O(100k) at most, we can afford to skip
    the binary heap implementation and just do a linear scan.
    """

    def __init__(
        self, alpha: float, ranked: bool, q_function: Callable[..., Awaitable]
    ):
        super().__init__()
        self.alpha = alpha
        self.ranked = ranked
        self.q_function = q_function

        self.buf_size: int | None = None

    def resize(self, size: int):
        self.buf_size = size

    @staticmethod
    async def _call_q(
        q_function: Callable[..., Awaitable], pbar: tqdm, *args, **kwargs
    ) -> float:
        # TODO: clean up this branching and force user to specify a Callable[..., Awaitable[float]]
        if isinstance(q_function, AsyncTorchModule):
            _, result = await q_function(*args, **kwargs)
        else:
            result = await q_function(*args, **kwargs)

        if isinstance(result, torch.Tensor):
            result = result.item()

        pbar.update()
        return cast("float", result)

    async def prepare_for_sampling(self):
        if self.buf_size is None:
            return

        pbar = tqdm(total=len(self.data), desc="Computing TD errors", ncols=0)
        async with eval_mode():
            values = await asyncio.gather(*[
                self._call_q(
                    self.q_function, pbar, *el["input_args"], **el["input_kwargs"]
                )
                for el in self.data
            ])
        for el, v in zip(self.data, values, strict=True):
            el["td_error"] = el["discounted_return"] - v

    def batched_iter(
        self,
        batch_size: int,
        shuffle: bool = True,
        infinite: bool = False,
    ) -> Iterator[dict]:
        if self.buf_size is None or (len(self.data) <= self.buf_size):
            # resize hasn't been called yet or we haven't hit the limit, so
            # use all samples
            buffer = self.data

        else:
            # roughly following Algo 1
            try:
                abs_tde = np.abs(
                    np.array([el["td_error"] for el in self.data])
                )  # L11-12
            except KeyError:
                raise RuntimeError(
                    "TD errors not available for all samples in the buffer. "
                    "Make sure to call prepare_for_update() after adding all samples "
                    "and before sampling."
                ) from None

            if self.ranked:
                ranks = np.argsort(abs_tde)
                prio = 1 / (ranks + 1)
            else:
                prio = abs_tde
            exp_prio = prio**self.alpha  # L9
            prob = exp_prio / exp_prio.sum()  # L9

            idcs = np.arange(len(self.data))
            sampled_idcs = np.random.choice(  # noqa: NPY002  # TODO: fix
                idcs, size=self.buf_size, p=prob, replace=False
            )
            buffer = [self.data[i] for i in sampled_idcs]

        return self._batched_iter(buffer, batch_size, shuffle, infinite)


class ReplayBufferType(StrEnum):
    # Maps to different buffer classes
    APPEND_ONLY = auto()
    CIRCULAR = auto()
    RANDOMIZED = auto()
    PRIORITIZED = auto()


class ReplayBufferConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    buf_type: ReplayBufferType = Field(
        default=ReplayBufferType.CIRCULAR, description="Circular is a good default."
    )
    size: int | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    def make_buffer(self, **kwargs) -> ReplayBuffer:
        # kwargs are only for prioritized case
        if self.buf_type == ReplayBufferType.APPEND_ONLY:
            return ReplayBuffer()
        if self.buf_type == ReplayBufferType.CIRCULAR:
            return CircularReplayBuffer()
        if self.buf_type == ReplayBufferType.RANDOMIZED:
            return RandomizedReplayBuffer()
        if self.buf_type == ReplayBufferType.PRIORITIZED:
            kwargs = self.kwargs | kwargs
            return PrioritizedReplayBuffer(**kwargs)
        typing.assert_never(self.buf_type)
