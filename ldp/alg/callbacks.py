import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import aiofiles
from aviary.env import Environment, TaskDataset
from aviary.message import Message
from aviary.tools import MessagesAdapter, ToolRequestMessage

from ldp.agent import Agent
from ldp.data_structures import Trajectory, Transition
from ldp.graph.ops import OpResult

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class Callback:
    """Base class for callbacks used by RolloutManager/Evaluator/OnlineTrainer.

    Pseudocode to demonstrate how callback methods are invoked (marked as *):

    RolloutManager.sample_trajectories():
        while not done:
            callback.before_transition() *
            agent.get_asv()
            callback.after_agent_get_asv() *
            env.step()
            callback.after_env_step() *
            callback.after_transition() *

    Evaluator.evaluate / OnlineTrainer._eval_loop():
        callback.before_eval_loop() *
        for batch in eval_dataset:
            rollout_manager.sample_trajectories()
            callback.after_eval_step() *
        callback.after_eval_loop() *

    OfflineTrainer / OnlineTrainer.train():
        for batch in train_dataset:
            rollout_manager.sample_trajectories() # if online
            optimizer.aggregate()
            if updating_optimizer:
                optimizer.update()
                callback.after_update() *
            callback.after_train_step() *
    """

    async def before_transition(
        self,
        traj_id: str,
        agent: Agent,
        env: Environment,
        agent_state: Any,
        obs: list[Message],
    ) -> None:
        """Invoked by RolloutManager before each transition."""

    async def after_agent_get_asv(
        self,
        traj_id: str,
        action: OpResult[ToolRequestMessage],
        next_agent_state: Any,
        value: float,
    ):
        """Invoked by RolloutManager after agent.get_asv()."""

    async def after_env_step(
        self, traj_id: str, obs: list[Message], reward: float, done: bool, trunc: bool
    ):
        """Invoked by RolloutManager after env.step()."""

    async def after_transition(
        self, traj_id: str, agent: Agent, env: Environment, transition: Transition
    ) -> None:
        """Invoked by RolloutManager after each transition."""

    async def after_train_step(self, trajectories: Sequence[Trajectory]) -> None:
        """Invoked by OnlineTrainer after each training step."""

    async def before_eval_loop(self) -> None:
        """Invoked by Evaluator and OnlineTrainer before the evaluation loop."""

    async def after_eval_step(self, trajectories: Sequence[Trajectory]) -> None:
        """Invoked by Evaluator and OnlineTrainer after each evaluation step."""

    async def after_eval_loop(self) -> None:
        """Invoked by Evaluator and OnlineTrainer after the evaluation loop."""

    async def after_update(self) -> None:
        """Invoked by OnlineTrainer after each optimizer.update() call."""


class TrajectoryFileCallback(Callback):
    """Callback that writes trajectories to a file."""

    def __init__(self, output_dir: os.PathLike | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.out_files: dict[str, Path] = {}
        self.trajs: dict[str, Trajectory] = defaultdict(Trajectory)

    def _make_filename(self, traj_id: str, env: Environment) -> str:
        """Create the filename for the output file."""
        return f"{traj_id}.jsonl"

    async def before_transition(
        self,
        traj_id: str,
        agent: Agent,
        env: Environment,
        agent_state: Any,
        obs: list[Message],
    ) -> None:
        if traj_id not in self.out_files:
            self.out_files[traj_id] = self.output_dir / self._make_filename(
                traj_id, env
            )

    async def after_transition(
        self, traj_id: str, agent: Agent, env: Environment, transition: Transition
    ) -> None:
        assert traj_id in self.out_files
        traj = self.trajs[traj_id]
        traj.steps.append(transition)
        # TODO: make this async?
        traj.to_jsonl(self.out_files[traj_id])

    def cleanup(self) -> None:
        for out_file in self.out_files.values():
            if out_file.exists():
                out_file.unlink()


class RolloutDebugDumpCallback(Callback):
    """Writes rollout debug info to an output directory."""

    def __init__(self, output_dir: os.PathLike | str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.out_files: dict[str, Path] = {}

    def _get_out_file(self, traj_id: str) -> Path:
        if traj_id not in self.out_files:
            self.out_files[traj_id] = self.output_dir / f"{traj_id}.jsonl"
        return self.out_files[traj_id]

    async def before_transition(
        self,
        traj_id: str,
        agent: Agent,
        env: Environment,
        agent_state,
        obs: list[Message],
    ) -> None:
        self.start = time.time()

    def _get_time_elapsed(self) -> float:
        elapsed = time.time() - self.start
        self.start = time.time()
        return elapsed

    async def after_agent_get_asv(
        self,
        traj_id: str,
        action: OpResult[ToolRequestMessage],
        next_agent_state: Any,
        value: float,
    ) -> None:
        log = {
            "event": "AGENT_GET_ASV",
            "elapsed": self._get_time_elapsed(),
            "action": action.value.model_dump(),
            "value": value,
        }
        async with aiofiles.open(self._get_out_file(traj_id), "a") as f:
            await f.write(json.dumps(log) + "\n")

    async def after_env_step(
        self,
        traj_id: str,
        obs: list[Message],
        reward: float,
        done: bool,
        trunc: bool,
    ):
        log = {
            "event": "ENV_STEP",
            "elapsed": self._get_time_elapsed(),
            "obs": MessagesAdapter.dump_python(obs),
            "reward": reward,
            "done": done,
            "truncated": trunc,
        }
        async with aiofiles.open(self._get_out_file(traj_id), "a") as f:
            await f.write(json.dumps(log) + "\n")


class ComputeTrajectoryMetricsMixin:
    """Mixin for TaskDataset classes to enable them to compute metrics."""

    def compute_trajectory_metrics(
        self,
        trajectories: Sequence[Trajectory],
    ) -> dict[str, list[float]]:
        return {
            "reward": [
                sum(step.reward for step in traj.steps) for traj in trajectories
            ],
            "truncation_rate": [
                sum(step.truncated for step in traj.steps) for traj in trajectories
            ],
            "avg_value": [
                sum(step.value for step in traj.steps) / len(traj.steps)
                for traj in trajectories
            ],
            "num_steps": [len(traj.steps) for traj in trajectories],
        }


class TrajectoryMetricsCallback(Callback):
    """
    Compute metrics that are defined by task datasets.

    NOTE: evaluation portion's after_eval_step/loop() is not concurrency safe because
    trajectories should be stored in the order of after_eval_step() calls.
    """

    def __init__(
        self,
        train_dataset: TaskDataset | None = None,
        eval_dataset: TaskDataset | None = None,
        train_metrics_transform: Callable[[dict[str, list[float]]], Any] = lambda x: x,
        eval_metrics_transform: Callable[
            [list[dict[str, list[float]]]], Any
        ] = lambda x: x,
    ):
        for ds in (train_dataset, eval_dataset):
            if ds and not isinstance(ds, ComputeTrajectoryMetricsMixin):
                raise ValueError(
                    f"Dataset {ds} didn't implement"
                    f" {ComputeTrajectoryMetricsMixin.__name__}, which is required for"
                    " this callback."
                )
        self._train_metrics_fn = (
            train_dataset.compute_trajectory_metrics if train_dataset else None  # type: ignore[attr-defined]
        )
        self._eval_metrics_fn = (
            eval_dataset.compute_trajectory_metrics if eval_dataset else None  # type: ignore[attr-defined]
        )
        self._train_metrics_transform = train_metrics_transform
        self._eval_metrics_transform = eval_metrics_transform
        self._eval_trajectories: list[Sequence[Trajectory]] = []

    async def after_train_step(self, trajectories: Sequence[Trajectory]) -> None:
        if not self._train_metrics_fn:
            return
        self._train_metrics_transform(self._train_metrics_fn(trajectories))

    async def after_eval_step(self, trajectories: Sequence[Trajectory]) -> None:
        if not self._eval_metrics_fn:
            return
        self._eval_trajectories.append(trajectories)

    async def after_eval_loop(self) -> None:
        if not self._eval_metrics_fn:
            return
        self._eval_metrics_transform([
            self._eval_metrics_fn(ts) for ts in self._eval_trajectories
        ])
        self._eval_trajectories.clear()


class MeanMetricsCallback(TrajectoryMetricsCallback):
    """Take a mean of all metrics."""

    def __init__(
        self,
        train_dataset: TaskDataset | None = None,
        eval_dataset: TaskDataset | None = None,
    ):
        super().__init__(
            train_dataset,
            eval_dataset,
            train_metrics_transform=partial(self._compute_means, "_train_means"),
            eval_metrics_transform=partial(self._compute_means, "_eval_means"),
        )
        self._train_means: dict[str, float] | None = None
        self._eval_means: dict[str, float] | None = None

    def _compute_means(
        self, attr: str, metrics: dict[str, list[float]] | list[dict[str, list[float]]]
    ) -> None:
        if isinstance(metrics, list):  # We need to flatten
            buckets: dict[str, list[float]] = defaultdict(list)
            for m in metrics:
                for k, v in m.items():
                    buckets[k].extend(v)
        else:
            buckets = metrics
        setattr(self, attr, {k: sum(v) / len(v) for k, v in buckets.items()})

    @property
    def train_means(self) -> dict[str, float]:
        if self._train_means is None:
            raise RuntimeError(
                "Training means are only available after this callback is invoked."
            )
        return self._train_means

    @property
    def eval_means(self) -> dict[str, float]:
        if self._eval_means is None:
            raise RuntimeError(
                "Evaluation means are only available after this callback is invoked."
            )
        return self._eval_means


class WandBLoggingCallback(TrajectoryMetricsCallback):
    def __init__(
        self,
        train_dataset: TaskDataset | None = None,
        eval_dataset: TaskDataset | None = None,
    ):
        if wandb is None:
            raise ImportError(
                f"{type(self).__name__} processing requires the 'monitor' extra for"
                " 'wandb'. Please: `pip install aviary-internal[monitor]`."
            )
        super().__init__(
            train_dataset,
            eval_dataset,
            train_metrics_transform=self._train_log,
            eval_metrics_transform=self._eval_log,
        )

        self._num_train_step = 0

    async def after_train_step(self, trajectories: Sequence[Trajectory]) -> None:
        self._num_train_step += 1
        return await super().after_train_step(trajectories)

    def _train_log(self, metrics: dict[str, list[float]]) -> None:
        # Each wandb.log() increments the wandb step by 1. Log the training step here
        # so we can use it as an x-axis for training metrics that are logged by different
        # wandb.log() calls.
        wandb.log(
            {
                f"train/{key}_mean": sum(vals) / len(vals)
                for key, vals in metrics.items()
            }
            | {"train/step": self._num_train_step}
        )

    @staticmethod
    def _eval_log(metrics: list[dict[str, list[float]]]) -> None:
        flattened_metrics = defaultdict(list)
        for m in metrics:
            for k, v in m.items():
                flattened_metrics[k].extend(v)
        wandb.log({
            f"eval/{key}_mean": sum(vals) / len(vals)
            for key, vals in flattened_metrics.items()
        })
