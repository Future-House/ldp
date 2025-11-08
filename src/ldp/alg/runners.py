from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Sequence
from typing import Any, cast

from aviary.core import Environment, TaskDataset
from pydantic import BaseModel, ConfigDict, Field

from ldp.agent import Agent
from ldp.alg.optimizer import Optimizer
from ldp.data_structures import Trajectory
from ldp.graph import OpResult, eval_mode, train_mode
from ldp.shims import tqdm

from .callbacks import Callback, ClearContextCallback
from .rollout import RolloutManager


async def _run_eval_loop(
    dataset: TaskDataset,
    rollout_manager: RolloutManager,
    batch_size: int,
    num_iterations: int | None,
    max_rollout_steps: int | None,
    callbacks: Sequence[Callback],
    shuffle: bool = False,
    num_rollouts_per_env: int = 1,
) -> None:
    await asyncio.gather(*[callback.before_eval_loop() for callback in callbacks])

    if num_iterations is None:
        try:
            num_iterations = math.ceil(len(dataset) / batch_size)
        except TypeError:
            raise ValueError(
                "If num_iterations is not provided, the "
                "dataset must be finite and implement __len__."
            ) from None
    if not num_iterations:
        return

    with tqdm(
        desc="Evaluation Iterations", ncols=0, total=num_iterations, leave=False
    ) as pbar:
        # We use pbar.n as a counter for the number of training steps
        while pbar.n < num_iterations:
            for batch in dataset.iter_batches(batch_size, shuffle=shuffle):
                all_trajectories: list[Trajectory] = []

                for _ in range(num_rollouts_per_env):
                    trajectories = await rollout_manager.sample_trajectories(
                        environments=batch, max_steps=max_rollout_steps
                    )
                    all_trajectories.extend(trajectories)

                await asyncio.gather(*[
                    callback.after_eval_step(all_trajectories) for callback in callbacks
                ])
                pbar.update()

                if pbar.n == num_iterations:
                    break

    await asyncio.gather(*[callback.after_eval_loop() for callback in callbacks])


class EvaluatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 1
    num_eval_iterations: int | None = Field(
        default=None,
        description=(
            "Number of eval iterations. "
            "If not provided, will exhaust the dataset. "
            "If 0, will not run the eval loop. "
        ),
    )
    num_rollouts_per_env: int = Field(
        default=1,
        description=(
            "Number of rollouts to execute for each environment in the dataset."
        ),
    )
    max_rollout_steps: int | None = None
    catch_agent_failures: bool = True
    catch_env_failures: bool = True
    clear_ctx_at_each_iter: bool = False
    shuffle: bool = Field(default=False, description="Shuffles the evaluation dataset.")

    def make_rollout_manager(
        self, agent: Agent, callbacks: Sequence[Callback]
    ) -> RolloutManager:
        return RolloutManager(
            agent=agent,
            callbacks=callbacks,
            catch_agent_failures=self.catch_agent_failures,
            catch_env_failures=self.catch_env_failures,
        )


class Evaluator:
    def __init__(
        self,
        config: EvaluatorConfig,
        agent: Agent,
        dataset: TaskDataset,
        callbacks: Sequence[Callback] | None = None,
    ):
        self.config = config
        self.agent = agent
        self.dataset = dataset
        self.callbacks = callbacks or []
        if self.config.clear_ctx_at_each_iter:
            clear_cb = ClearContextCallback()
            self.callbacks = [*self.callbacks, clear_cb] if callbacks else [clear_cb]
        self.rollout_manager = self.config.make_rollout_manager(agent, self.callbacks)

    @eval_mode()
    async def evaluate(self, **kwargs) -> None:
        """Run the agent over the provided dataset in eval mode."""
        return await self.run(**kwargs)

    async def run(self, **kwargs) -> None:
        """Run the agent over the provided dataset.

        **kwargs can be used to override config settings.

        This method does not set training mode, so it can be used to collect
        trajectories for offline training.
        """
        eval_kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "num_iterations": self.config.num_eval_iterations,
            "max_rollout_steps": self.config.max_rollout_steps,
            "shuffle": self.config.shuffle,
            "num_rollouts_per_env": self.config.num_rollouts_per_env,
        }
        await _run_eval_loop(
            dataset=self.dataset,
            rollout_manager=self.rollout_manager,
            callbacks=self.callbacks,
            **(eval_kwargs | kwargs),
        )


class OnlineTrainerConfig(EvaluatorConfig):
    batch_size: int
    num_train_iterations: int = Field(
        ge=0,
        description=(
            "Number of iterations (at one batch per iteration) to process during"
            " training, and setting to 0 skips training."
        ),
    )
    update_every: int = Field(
        default=1,
        description="Number of training iterations between optimizer update calls.",
        ge=1,
    )
    eval_every: int | None = Field(
        None,
        description=(
            "If set, will repeatedly evaluate on the validation set after this many"
            " iterations. If unset (default), no evaluation is performed."
        ),
    )
    eval_before: bool = Field(
        default=True,
        description=(
            "If True (default), evaluate on the validation set before kicking off"
            " training."
        ),
    )
    clear_ctx_at_each_iter: bool = False


class OnlineTrainer:
    def __init__(
        self,
        config: OnlineTrainerConfig,
        agent: Agent,
        optimizer: Optimizer,
        train_dataset: TaskDataset,
        eval_dataset: TaskDataset | None = None,
        callbacks: Sequence[Callback] | None = None,
    ):
        if config.eval_every is not None and eval_dataset is None:
            raise ValueError("Must specify eval_dataset if eval_every is set")

        self.config = config
        self.agent = agent
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        if self.config.clear_ctx_at_each_iter:
            clear_cb = ClearContextCallback()
            self.callbacks = [*self.callbacks, clear_cb] if callbacks else [clear_cb]
        self.rollout_manager = self.config.make_rollout_manager(
            agent=agent, callbacks=self.callbacks
        )

    async def train(self) -> None:
        if self.config.eval_before:
            await self.evaluate()

        with tqdm(
            desc="Training Iterations", ncols=0, total=self.config.num_train_iterations
        ) as pbar:
            # We use pbar.n as a counter for the number of training steps
            while pbar.n < self.config.num_train_iterations:
                for batch in self.train_dataset.iter_batches(
                    self.config.batch_size, shuffle=True
                ):
                    await self._training_step(pbar.n, batch)
                    pbar.update()  # Increment pbar.n by 1

                    if (
                        self.config.eval_every is not None
                        and pbar.n % self.config.eval_every == 0
                    ):
                        await self.evaluate()

                    if pbar.n == self.config.num_train_iterations:
                        break  # Will also break out of the outer while loop

        await self.evaluate()

    @eval_mode()
    async def evaluate(self, **kwargs) -> None:
        await _run_eval_loop(
            dataset=cast("TaskDataset", self.eval_dataset),
            rollout_manager=self.rollout_manager,
            batch_size=self.config.batch_size,
            num_iterations=self.config.num_eval_iterations,
            max_rollout_steps=self.config.max_rollout_steps,
            callbacks=self.callbacks,
            **kwargs,
        )

    @train_mode()
    async def _training_step(self, i_iter: int, envs: Sequence[Environment]) -> None:
        training_batch: list[Trajectory] = []
        all_trajectories: list[Trajectory] = []

        for _ in range(self.config.num_rollouts_per_env):
            trajectories = await self.rollout_manager.sample_trajectories(
                environments=envs, max_steps=self.config.max_rollout_steps
            )

            training_batch.extend(traj for traj in trajectories if not traj.failed)
            all_trajectories.extend(trajectories)

        await self._optimizer_step(i_iter, training_batch)

        await asyncio.gather(*[
            callback.after_train_step(all_trajectories) for callback in self.callbacks
        ])

    async def _optimizer_step(
        self, i_iter: int, training_batch: Sequence[Trajectory]
    ) -> None:
        for traj in training_batch:
            for step in traj.steps:
                # TODO: make this async
                # step.action is not None because we checked traj.failed above
                cast("OpResult", step.action).compute_grads()

        self.optimizer.aggregate(training_batch)

        if (i_iter + 1) % self.config.update_every == 0:
            await self.optimizer.update()

            await asyncio.gather(*[
                callback.after_update() for callback in self.callbacks
            ])


class OfflineTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(ge=1)
    update_every: int = Field(
        default=1,
        description="Number of training iterations to run before updating the model.",
        ge=0,
    )
    clear_ctx_at_each_iter: bool = False
    num_epochs: int = Field(default=1, ge=0)


class OfflineTrainer:
    def __init__(
        self,
        config: OfflineTrainerConfig,
        agent: Agent,
        optimizer: Optimizer,
        train_trajectories: list[Trajectory],
        callbacks: Sequence[Callback] | None = None,
    ):
        self.config = config
        self.agent = agent
        self.optimizer = optimizer
        # copy so we can shuffle
        self.train_trajectories = train_trajectories.copy()
        self.callbacks = callbacks or []
        if self.config.clear_ctx_at_each_iter:
            clear_cb = ClearContextCallback()
            self.callbacks = [*self.callbacks, clear_cb] if callbacks else [clear_cb]

    async def train(self) -> None:
        random.shuffle(self.train_trajectories)

        full_batch = len(self.train_trajectories) <= self.config.batch_size
        num_total_steps = self.config.num_epochs * (
            1
            if full_batch
            else (len(self.train_trajectories) // self.config.batch_size)
        )

        with tqdm(total=num_total_steps, desc="Training Iterations", ncols=0) as pbar:
            for _ in range(self.config.num_epochs):
                if full_batch:
                    # Separating out the full batch case lets the user run a single update()
                    # step even if train_trajectories is empty. This can be useful if the
                    # optimizer is pre-populated with offline training data, for example.
                    batch = self.train_trajectories

                    self.optimizer.aggregate(batch, show_pbar=True)
                    await self.optimizer.update()

                    await asyncio.gather(*[
                        callback.after_update() for callback in self.callbacks
                    ])

                    await asyncio.gather(*[
                        callback.after_train_step(batch) for callback in self.callbacks
                    ])
                    pbar.update()

                else:
                    for training_step, i_batch_start in enumerate(
                        range(0, len(self.train_trajectories), self.config.batch_size)
                    ):
                        batch = self.train_trajectories[
                            i_batch_start : i_batch_start + self.config.batch_size
                        ]

                        self.optimizer.aggregate(batch)

                        if (training_step + 1) % self.config.update_every == 0:
                            await self.optimizer.update()
                            await asyncio.gather(*[
                                callback.after_update() for callback in self.callbacks
                            ])

                        await asyncio.gather(*[
                            callback.after_train_step(batch)
                            for callback in self.callbacks
                        ])
                        pbar.update()
