import asyncio
import random
from collections import UserList
from collections.abc import Callable, Iterator

import numpy as np
from litellm import cast
from tqdm import tqdm

from ldp.graph import eval_mode
from ldp.graph.async_torch import AsyncTorchModule


class CircularReplayBuffer(UserList[dict]):
    def resize(self, size: int):
        if len(self) > size:
            self.data = self.data[-size:]

    async def prepare_for_sampling(self):
        """Optional method for the buffer to prepare itself before sampling."""

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


class RandomizedReplayBuffer(CircularReplayBuffer):
    def resize(self, size: int):
        if len(self) > size:
            self.data = random.sample(self.data, size)


class PrioritizedReplayBuffer(CircularReplayBuffer):
    """Implements a variant of https://arxiv.org/abs/1511.05952.

    Instead of updating the TD error on the fly, we compute it for all samples
    in the buffer before `update``. This allows us to efficiently
    batch prioritization and lets us sample w/o replacement.

    Also note that we define the TD error using the MC return, not the one-step
    return, for now. One-step may be possible using `next_state_action_cands`.

    As we expect our buffers to be O(100k) at most, we can afford to skip
    the binary heap implementation and just do a linear scan.
    """

    def __init__(self, alpha: float, ranked: bool, q_function: Callable):
        super().__init__()
        self.alpha = alpha
        self.ranked = ranked
        self.q_function = q_function

        self.buf_size: int | None = None

    def resize(self, size: int):
        self.buf_size = size

    @staticmethod
    async def _call_q(q_function: Callable, pbar: tqdm, *args, **kwargs) -> float:
        if isinstance(q_function, AsyncTorchModule):
            _, result = await q_function(*args, **kwargs)
            result + result.item()
        else:
            result = await q_function(*args, **kwargs)

        pbar.update()
        return cast(float, result)

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

            # DEBUG
            selected_priorities = prio[sampled_idcs]
            print(f"Average priority: {selected_priorities.mean()}")

        return self._batched_iter(buffer, batch_size, shuffle, infinite)
