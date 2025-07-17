import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import aiofiles
from aviary.core import (
    Environment,
    Message,
    MessagesAdapter,
    TaskDataset,
    Tool,
    ToolRequestMessage,
)

from ldp.agent import Agent
from ldp.data_structures import Trajectory, Transition
from ldp.graph import OpCtx, OpResult

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from ldp.alg.optimizer.replay_buffers import ReplayBuffer

logger = logging.getLogger(__name__)


class Callback:
    """Base class for callbacks used by RolloutManager/Evaluator/OnlineTrainer.

    Pseudocode to demonstrate how callback methods are invoked (marked as *):

    RolloutManager.sample_trajectories():
        callback.before_rollout() *
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

    async def before_rollout(self, traj_id: str, env: Environment) -> None:
        """Invoked by runners when each rollout starts (at trajectory start)."""

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


class StoreTrajectoriesCallback(Callback):
    """Simple callback that stores train/eval trajectories in an in-memory list."""

    def __init__(self):
        self.train_trajectories = []
        self.eval_trajectories = []

    async def after_train_step(self, trajectories: Sequence[Trajectory]) -> None:
        self.train_trajectories.extend(trajectories)

    async def after_eval_step(self, trajectories: Sequence[Trajectory]) -> None:
        self.eval_trajectories.extend(trajectories)


class TrajectoryFileCallback(Callback):
    """Callback that writes trajectories to a file."""

    @staticmethod
    def default_serialize_env(env: Environment, transition: Transition) -> str | None:
        """Export a JSON-serialized Frame if the transition is done."""
        if not transition.done:
            return None
        try:
            frame = env.export_frame()
        except NotImplementedError:
            return None  # Allow for envs that didn't implement export_frame()
        return frame.model_dump_json(exclude={"state"}, indent=2)

    def __init__(
        self,
        output_dir: str | os.PathLike,
        serialize_env: Callable[
            [Environment, Transition], str | None
        ] = default_serialize_env,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.traj_files: dict[str, Path] = {}
        self.env_files: dict[str, Path] = {}
        self.trajs: dict[str, Trajectory] = defaultdict(Trajectory)

        self.serialize_env = serialize_env

    def _make_traj_filename(self, traj_id: str, env: Environment) -> str:
        """Create the filename for the output file."""
        return f"{traj_id}.jsonl"

    async def after_transition(
        self, traj_id: str, agent: Agent, env: Environment, transition: Transition
    ) -> None:
        if traj_id not in self.traj_files:
            self.traj_files[traj_id] = self.output_dir / self._make_traj_filename(
                traj_id, env
            )
            self.env_files[traj_id] = self.output_dir / f"{traj_id}_env.json"

        traj = self.trajs[traj_id]
        traj.steps.append(transition)

        async def possibly_dump_env() -> None:
            async with aiofiles.open(self.env_files[traj_id], "w") as f:
                env_or_none = self.serialize_env(env, transition)
                if env_or_none is not None:
                    await f.write(env_or_none)

        await asyncio.gather(
            possibly_dump_env(), traj.to_jsonl(self.traj_files[traj_id])
        )


class StoreEnvironmentsCallback(Callback):
    """Callback to store the environment underlying each trajectory."""

    def __init__(self):
        self.traj_id_to_envs: dict[str, Environment] = {}

    async def before_rollout(self, traj_id: str, env: Environment) -> None:
        self.traj_id_to_envs[traj_id] = env


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

    # Tools or tool names to include in trajectory metrics
    tools_to_track: Collection[str | Tool] = set()

    def compute_trajectory_metrics(
        self,
        trajectories: Sequence[Trajectory],
    ) -> dict[str, list[float]]:
        metrics: dict[str, list[float]] = {
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
        for tool in self.tools_to_track:  # Default of empty set means this is not run
            if isinstance(tool, Tool):
                tool = tool.info.name
            metrics[f"tool_{tool}"] = [
                sum(
                    sum(tc.function.name == tool for tc in s.action.value.tool_calls)
                    for s in traj.steps
                    if isinstance(s.action, OpResult)
                )
                for traj in trajectories
            ]
        return metrics


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
        track_tool_usage: bool = False,
    ):
        self._datasets = train_dataset, eval_dataset
        self._track_tool_usage = track_tool_usage
        for ds in self._datasets:
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

    async def after_env_reset(
        self, traj_id: str, obs: list[Message], tools: list[Tool]
    ) -> None:
        for ds in (ds for ds in self._datasets if ds):
            if self._track_tool_usage:
                cast("ComputeTrajectoryMetricsMixin", ds).tools_to_track = {
                    t.info.name for t in tools
                }

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
    def __init__(self, *args, **kwargs):
        if wandb is None:
            raise ImportError(
                f"{type(self).__name__} processing requires the 'monitor' extra for"
                " 'wandb'. Please: `pip install aviary-internal[monitor]`."
            )
        super().__init__(*args, **kwargs)

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
            f"eval/{key}_mean": sum(vals) / len(vals) if vals else None
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


class TerminalPrintingCallback(Callback):
    """Callback that prints action, observation, and timing information to the terminal."""

    def __init__(self):
        self.start_time = None
        # try now, rather than start running and die
        try:
            from rich.pretty import pprint  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"rich is required for {type(self).__name__}. Please install it with"
                " `pip install rich`."
            ) from e

    async def before_transition(
        self,
        traj_id: str,
        agent: Agent,
        env: Environment,
        agent_state: Any,
        obs: list[Message],
    ) -> None:
        """Start the timer before each transition."""
        self.start_time = time.time()

    async def after_agent_get_asv(
        self,
        traj_id: str,
        action: OpResult[ToolRequestMessage],
        next_agent_state: Any,
        value: float,
    ) -> None:
        from rich.pretty import pprint

        print("\nAction:")
        pprint(action.value, expand_all=True)

    async def after_env_step(
        self, traj_id: str, obs: list[Message], reward: float, done: bool, trunc: bool
    ) -> None:
        from rich.pretty import pprint

        # Compute elapsed time
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.start_time = None  # Reset timer
        else:
            elapsed_time = 0.0
        print("\nObservation:")
        pprint(obs, expand_all=True)
        print(f"Elapsed time: {elapsed_time:.2f} seconds")


class ClearOptimizerBuffersCallback(Callback):
    """Invoke the clear method on buffer(s) after each optimizer update."""

    def __init__(self, *buffers: "ReplayBuffer"):
        self._buffers = list(buffers)

    async def after_update(self) -> None:
        for b in self._buffers:
            b.clear()
