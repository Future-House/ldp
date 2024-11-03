from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable

from ldp.data_structures import Trajectory
from ldp.shims import tqdm

logger = logging.getLogger(__name__)


# Registry for all optimizers
_OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {}


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init_subclass__(cls) -> None:
        # Register each optimizer subclass
        _OPTIMIZER_REGISTRY[cls.__name__] = cls
        return super().__init_subclass__()

    def aggregate(
        self, trajectories: Iterable[Trajectory], show_pbar: bool = False, **kwargs
    ) -> None:
        """Aggregate trajectories to construct training samples."""
        trajectories_with_pbar = tqdm(
            trajectories,
            desc="Aggregating trajectories",
            ncols=0,
            mininterval=1,
            disable=not show_pbar,
        )
        for trajectory in trajectories_with_pbar:
            self.aggregate_trajectory(trajectory, **kwargs)

    @abstractmethod
    def aggregate_trajectory(self, trajectory: Trajectory) -> None:
        """Aggregate transitions from a single trajectory to construct training samples."""

    @abstractmethod
    async def update(self) -> None:
        """Update the model based on the aggregated samples."""


class ChainedOptimizer(Optimizer):
    """An optimizer that runs a sequence of sub-optimizers in the order they are provided."""

    def __init__(self, *optimizers: Optimizer):
        self.optimizers = optimizers

    def aggregate(
        self, trajectories: Iterable[Trajectory], show_pbar: bool = False, **kwargs
    ) -> None:
        for optimizer in self.optimizers:
            optimizer.aggregate(trajectories, show_pbar=show_pbar, **kwargs)

    async def update(self) -> None:
        for optimizer in self.optimizers:
            await optimizer.update()
