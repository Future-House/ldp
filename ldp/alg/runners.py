from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Sequence
from typing import cast

from aviary.env import Environment, TaskDataset
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm, trange

from ldp.agent import Agent
from ldp.alg.callbacks import Callback
from ldp.alg.optimizer import Optimizer
from ldp.alg.rollout import RolloutManager
from ldp.data_structures import Trajectory
from ldp.graph.op_utils import eval_mode, train_mode
from ldp.graph.ops import OpResult


async def _run_eval_loop(
    dataset: TaskDataset,
    rollout_manager: RolloutManager,
    batch_size: int,
    num_iterations: int | None,
    max_rollout_steps: int | None,
    callbacks: Sequence[Callback],
    shuffle: bool = False,
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

    for i_iter, envs in tqdm(
        enumerate(dataset.iter_batches(batch_size, shuffle=shuffle)),
        desc="Evaluation Iterations",
        ncols=0,
        leave=False,
        total=num_iterations,
    ):
        trajectories = await rollout_manager.sample_trajectories(
            environments=envs, max_steps=max_rollout_steps
        )

        # Close the environment after we have sampled from it,
        # in case it needs to tear down resources.
        await asyncio.gather(*(env.close() for env in envs))

        await asyncio.gather(*[
            callback.after_eval_step(trajectories) for callback in callbacks
        ])

        if i_iter == num_iterations - 1:
            break

    await asyncio.gather(*[callback.after_eval_loop() for callback in callbacks])


class EvaluatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 1
    num_eval_iterations: int | None = Field(
        None,
        description="Number of eval iterations. "
        "If not provided, will exhaust the dataset. "
        "If 0, will not run the eval loop. ",
    )
    max_rollout_steps: int | None = None
    catch_agent_failures: bool = True
    catch_env_failures: bool = True

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
        self.rollout_manager = self.config.make_rollout_manager(agent, self.callbacks)

    @eval_mode()
    async def evaluate(self, **kwargs) -> None:
        """Run the agent over the provided dataset in eval mode."""
        return await self.run(**kwargs)

    async def run(self, **kwargs) -> None:
        """Run the agent over the provided dataset.

        This method does not set training mode, so it can be used to collect
        trajectories for offline training.
        """
        await _run_eval_loop(
            dataset=self.dataset,
            rollout_manager=self.rollout_manager,
            batch_size=self.config.batch_size,
            num_iterations=self.config.num_eval_iterations,
            max_rollout_steps=self.config.max_rollout_steps,
            callbacks=self.callbacks,
            **kwargs,
        )


class OnlineTrainerConfig(EvaluatorConfig):
    batch_size: int
    num_train_iterations: int
    num_rollouts_per_env: int = Field(
        1,
        description="Number of rollouts to execute for each "
        "environment per training iteration.",
    )
    update_every: int = Field(
        1,
        description="Number of training iterations to run before updating the model.",
    )
    eval_every: int | None = Field(
        None,
        description=(
            "If set, will repeatedly evaluate on the validation set after this many"
            " iterations. If unset (default), no evaluation is performed."
        ),
    )
    eval_before: bool = Field(
        True,  # noqa: FBT003
        description="If True (default), run an evaluation loop before training.",
    )


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
        self.rollout_manager = self.config.make_rollout_manager(
            agent=agent, callbacks=self.callbacks
        )

    async def train(self) -> None:
        if self.config.eval_before:
            await self._eval_loop()

        pbar = tqdm(
            desc="Training Iterations", ncols=0, total=self.config.num_train_iterations
        )

        while pbar.n < self.config.num_train_iterations:
            for batch in self.train_dataset.iter_batches(
                self.config.batch_size, shuffle=True
            ):
                await self._training_step(pbar.n, batch)
                pbar.update()

                if (
                    self.config.eval_every is not None
                    and pbar.n % self.config.eval_every == 0
                ):
                    await self._eval_loop()

                if pbar.n == self.config.num_train_iterations:
                    break

        pbar.close()

        await self._eval_loop()

    @eval_mode()
    async def _eval_loop(self, **kwargs) -> None:
        if self.config.num_eval_iterations == 0:
            return

        await _run_eval_loop(
            dataset=cast(TaskDataset, self.eval_dataset),
            rollout_manager=self.rollout_manager,
            batch_size=self.config.batch_size,
            num_iterations=self.config.num_eval_iterations,
            max_rollout_steps=self.config.max_rollout_steps,
            callbacks=self.callbacks,
            **kwargs,
        )

    @train_mode()
    async def _training_step(
        self, training_step: int, envs: Sequence[Environment]
    ) -> None:
        training_batch: list[Trajectory] = []

        for _ in range(self.config.num_rollouts_per_env):
            trajectories = await self.rollout_manager.sample_trajectories(
                environments=envs, max_steps=self.config.max_rollout_steps
            )

            # Close the environments after we have sampled from them, in case they need to tear down resources.
            await asyncio.gather(*[env.close() for env in envs])

            training_batch.extend(traj for traj in trajectories if not traj.failed)

        await self._optimizer_step(training_step, training_batch)

        await asyncio.gather(*[
            callback.after_train_step(trajectories) for callback in self.callbacks
        ])

    async def _optimizer_step(
        self, training_step: int, training_batch: Sequence[Trajectory]
    ) -> None:
        for traj in training_batch:
            for step in traj.steps:
                # TODO: make this async
                # step.action is not None because we checked traj.failed above
                cast(OpResult, step.action).compute_grads()

        self.optimizer.aggregate(training_batch)

        if (training_step + 1) % self.config.update_every == 0:
            await self.optimizer.update()

            await asyncio.gather(*[
                callback.after_update() for callback in self.callbacks
            ])


class OfflineTrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int
    update_every: int = Field(
        1,
        description="Number of training iterations to run before updating the model.",
    )
    # TODO: add some concept of eval loops


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

    async def train(self) -> None:
        random.shuffle(self.train_trajectories)

        for training_step, i_batch_start in enumerate(
            trange(
                0,
                len(self.train_trajectories),
                self.config.batch_size,
                desc="Training iterations",
                ncols=0,
            )
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
                callback.after_train_step(batch) for callback in self.callbacks
            ])
