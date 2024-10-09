import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path
from typing import Any

import aiofiles
from aviary.env import Environment, TaskDataset
from aviary.message import Message
from aviary.tools import MessagesAdapter, Tool, ToolRequestMessage

from ldp.agent import Agent
from ldp.data_structures import Trajectory, Transition
from ldp.graph.ops import OpCtx, OpResult

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class Callback:
    """Base class for callbacks used by RolloutManager/Evaluator/OnlineTrainer.

    Pseudocode to demonstrate how callback methods are invoked (marked as *):

    RolloutManager.sample_trajectories():
        env.reset()
        callback.after_env_reset() *
        agent.init_state()
        callback.after_agent_init_state() *
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
        """Invoked by runners before each transition and after agent and env reset."""

    async def after_agent_init_state(self, traj_id: str, init_state: Any) -> None:
        """Invoked by runners after agent.init_state()."""

    async def after_agent_get_asv(
        self,
        traj_id: str,
        action: OpResult[ToolRequestMessage],
        next_agent_state: Any,
        value: float,
    ) -> None:
        """Invoked by runners after agent.get_asv()."""

    async def after_env_reset(
        self, traj_id: str, obs: list[Message], tools: list[Tool]
    ) -> None:
        """Invoked by runners after env.reset()."""

    async def after_env_step(
        self, traj_id: str, obs: list[Message], reward: float, done: bool, trunc: bool
    ) -> None:
        """Invoked by runners after env.step()."""

    async def after_transition(
        self, traj_id: str, agent: Agent, env: Environment, transition: Transition
    ) -> None:
        """Invoked by runners after each transition."""

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
    """Dump JSONL files for each agent and environment step to a directory."""

    def __init__(self, output_dir: os.PathLike | str):
        """Initialize.

        Args:
            output_dir: Directory to place JSONL files.
        """
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

    def _get_elapsed_time(self, reset: bool = True) -> float:
        elapsed = time.time() - self.start
        if reset:
            self.start = time.time()
        return elapsed

    async def after_agent_get_asv(
        self,
        traj_id: str,
        action: OpResult[ToolRequestMessage],
        next_agent_state: Any,
        value: float,
    ) -> None:
        log_jsonl = json.dumps({
            "event": "AGENT_GET_ASV",
            "elapsed": self._get_elapsed_time(),
            "action": action.value.model_dump(),
            "value": value,
        })
        async with aiofiles.open(self._get_out_file(traj_id), "a") as f:
            await f.write(log_jsonl + "\n")

    async def after_env_step(
        self, traj_id: str, obs: list[Message], reward: float, done: bool, trunc: bool
    ) -> None:
        log_jsonl = json.dumps({
            "event": "ENV_STEP",
            "elapsed": self._get_elapsed_time(),
            "obs": MessagesAdapter.dump_python(obs),
            "reward": reward,
            "done": done,
            "truncated": trunc,
        })
        async with aiofiles.open(self._get_out_file(traj_id), "a") as f:
            await f.write(log_jsonl + "\n")


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
            "failures": [traj.failed for traj in trajectories],
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

        self._train_metrics: dict[str, list[float]] | None = None
        self._eval_metrics: dict[str, list[float]] = {}

    async def after_train_step(self, trajectories: Sequence[Trajectory]) -> None:
        if self._train_metrics_fn is not None:
            self._train_metrics = self._train_metrics_fn(trajectories)

    async def after_eval_step(self, trajectories: Sequence[Trajectory]) -> None:
        if self._eval_metrics_fn is not None:
            for k, v in self._eval_metrics_fn(trajectories).items():
                if k not in self._eval_metrics:
                    # Don't use defaultdict - error prone in user code
                    self._eval_metrics[k] = []
                self._eval_metrics[k].extend(v)

    async def after_eval_loop(self) -> None:
        self._eval_metrics.clear()


class MeanMetricsCallback(TrajectoryMetricsCallback):
    """Take a mean of all metrics."""

    def __init__(
        self,
        train_dataset: TaskDataset | None = None,
        eval_dataset: TaskDataset | None = None,
    ):
        super().__init__(train_dataset, eval_dataset)
        self._train_means: dict[str, float] | None = None
        self._eval_means: dict[str, float] | None = None

    async def after_train_step(self, trajectories: Sequence[Trajectory]) -> None:
        await super().after_train_step(trajectories)
        if self._train_metrics is not None:
            # may be None if train_dataset was not provided
            self._train_means = self._compute_means(self._train_metrics)

    async def after_eval_loop(self) -> None:
        if self._eval_metrics:
            # may be empty if eval_dataset was not provided
            self._eval_means = self._compute_means(self._eval_metrics)
        await super().after_eval_loop()

    @staticmethod
    def _compute_means(metrics: dict[str, list[float]]) -> dict[str, float]:
        return {k: sum(v) / len(v) for k, v in metrics.items()}

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
        super().__init__(train_dataset, eval_dataset)

        self._num_train_step = 0

    async def after_train_step(self, trajectories: Sequence[Trajectory]) -> None:
        await super().after_train_step(trajectories)
        self._num_train_step += 1

        if self._train_metrics is None:
            return

        # Each wandb.log() increments the wandb step by 1. Log the training step here
        # so we can use it as an x-axis for training metrics that are logged by different
        # wandb.log() calls.
        wandb.log(
            {
                f"train/{key}_mean": sum(vals) / len(vals)
                for key, vals in self._train_metrics.items()
            }
            | {"train/step": self._num_train_step}
        )

    async def after_eval_loop(self) -> None:
        if not self._eval_metrics:
            return

        wandb.log({
            f"eval/{key}_mean": (sum(vals) / len(vals) if vals else None)
            for key, vals in self._eval_metrics.items()
        })

        await super().after_eval_loop()


class ClearContextCallback(Callback):
    def __init__(self, op_names: Iterable[str] | None = None):
        self._op_names = op_names

    async def after_eval_step(self, trajectories: Sequence[Trajectory]) -> None:
        OpCtx.clear_contexts(self._op_names)

    async def after_update(self) -> None:
        OpCtx.clear_contexts(self._op_names)


class LoggingCallback(MeanMetricsCallback):
    """Custom callback for logging filtered metrics (e.g., pass rates) to the console.

    This callback extends the `MeanMetricsCallback` and allows logging of user-specified metrics
    after each training step and after the evaluation loop. It calculates the specified metrics
    (e.g., pass rates) from the trajectories and logs the results.
    """

    def __init__(
        self,
        train_dataset: TaskDataset | None = None,
        eval_dataset: TaskDataset | None = None,
        metrics_to_log: Collection[str] | None = None,
    ):
        """Initialize the callback with a list of metric keys to log.

        Args:
            train_dataset: The training dataset for computing metrics.
            eval_dataset: The evaluation dataset for computing metrics.
            metrics_to_log: Optional metric names (e.g., ["pass"]) to log.
                            If left as default of None, all metrics will be logged.
        """
        super().__init__(train_dataset, eval_dataset)
        self.metrics_to_log = (
            metrics_to_log or set()
        )  # If no metrics provided, log all by default

    def _log_filtered_metrics(self, metrics: dict[str, float], step_type: str) -> None:
        """Helper function to log only the specified metrics.

        Args:
            metrics: Dictionary of calculated means for the current step (e.g., train or eval).
            step_type: The type of step (e.g., "Train" or "Eval") for logging purposes.
        """
        if self.metrics_to_log:
            for metric in self.metrics_to_log:
                if metric in metrics:
                    logger.info(
                        f"{metric.upper()} RATE ({step_type}): {metrics[metric]:.5f}"
                    )
        else:
            # Log all metrics if no specific ones are provided
            logger.info(f"{step_type} Metrics: {metrics}")

    async def after_train_step(self, trajectories: Sequence[Trajectory]) -> None:
        """Log metrics and pass rate after each training step.

        This method is called after every training step, calculating and logging
        the training metrics and pass rate.

        Args:
            trajectories: A sequence of trajectories from the training step.
        """
        await super().after_train_step(trajectories)  # Call the parent to compute means
        if self.train_means:
            self._log_filtered_metrics(self.train_means, step_type="Train")

    async def after_eval_loop(self) -> None:
        """Log metrics and pass rate after the evaluation loop.

        This method is called after the evaluation loop finishes, calculating and logging
        the evaluation metrics and pass rate.
        """
        await super().after_eval_loop()  # Call the parent to compute means
        if self.eval_means:
            self._log_filtered_metrics(self.eval_means, step_type="Eval")
