from collections.abc import Sequence
from unittest.mock import patch

import pytest
from aviary.env import DummyEnv, TaskDataset

from ldp.agent import MemoryAgent, SimpleAgent
from ldp.alg.callbacks import Callback, MeanMetricsCallback
from ldp.alg.datasets import (  # noqa: F401  # Force TASK_DATASET_REGISTRY update
    DummyTaskDataset,
)
from ldp.alg.optimizer import default_optimizer_factory
from ldp.alg.runners import (
    Evaluator,
    EvaluatorConfig,
    OfflineTrainer,
    OfflineTrainerConfig,
    OnlineTrainer,
    OnlineTrainerConfig,
)
from ldp.data_structures import Trajectory


@pytest.mark.asyncio
async def test_online_trainer():
    agent = MemoryAgent()
    opt = default_optimizer_factory(agent)
    dataset = TaskDataset.from_name("dummy")
    callback = DummyCallback()

    train_conf = OnlineTrainerConfig(
        batch_size=1,
        num_train_iterations=1,
        max_rollout_steps=1,
        num_eval_iterations=1,
        eval_every=1,
    )
    trainer = OnlineTrainer(
        config=train_conf,
        agent=agent,
        optimizer=opt,
        train_dataset=dataset,
        eval_dataset=dataset,
        callbacks=[callback],
    )
    await trainer.train()

    for k, v in callback.fn_invocations.items():
        # eval is run 3 times: before training, during training, after training
        assert v == (3 if "eval" in k else 1)


@pytest.mark.asyncio
async def test_evaluator() -> None:
    agent = SimpleAgent()
    dataset = TaskDataset.from_name("dummy")
    metrics_callback = MeanMetricsCallback(eval_dataset=dataset)
    count_callback = DummyCallback()

    eval_conf = EvaluatorConfig(num_eval_iterations=1)
    evaluator = Evaluator(
        config=eval_conf,
        agent=agent,
        dataset=dataset,
        callbacks=[metrics_callback, count_callback],
    )
    with patch.object(DummyEnv, "close") as mock_close:
        await evaluator.evaluate()

    mock_close.assert_awaited_once(), "Env should be closed"
    assert isinstance(metrics_callback.eval_means["reward"], float)

    for k, v in count_callback.fn_invocations.items():
        assert v == (1 if "eval" in k else 0)


@pytest.mark.asyncio
async def test_offline_trainer():
    # This is kind of a system test of getting trajectories from the evaluator
    # and then training on them "offline"
    agent = MemoryAgent()
    opt = default_optimizer_factory(agent)
    dataset = TaskDataset.from_name("dummy")
    traj_callback = StoreTrajectoriesCallback()

    evaluator = Evaluator(
        config=EvaluatorConfig(num_eval_iterations=1),
        agent=agent,
        dataset=dataset,
        callbacks=[traj_callback],
    )
    await evaluator.evaluate()
    assert len(traj_callback.trajectories) == 1

    count_callback = DummyCallback()
    train_conf = OfflineTrainerConfig(batch_size=1)
    trainer = OfflineTrainer(
        config=train_conf,
        agent=agent,
        optimizer=opt,
        train_trajectories=traj_callback.trajectories,
        callbacks=[count_callback],
    )
    await trainer.train()

    assert count_callback.fn_invocations == {
        "after_train_step": 1,
        "after_eval_step": 0,
        "after_eval_loop": 0,
        "after_update": 1,
    }


class StoreTrajectoriesCallback(Callback):
    def __init__(self):
        self.trajectories = []

    async def after_eval_step(self, trajectories: Sequence[Trajectory]) -> None:
        self.trajectories.extend(trajectories)


class DummyCallback(Callback):
    def __init__(self):
        self.fn_invocations = {
            "after_train_step": 0,
            "after_eval_step": 0,
            "after_eval_loop": 0,
            "after_update": 0,
        }

    async def after_train_step(self, trajectories: Sequence[Trajectory]) -> None:
        self.fn_invocations["after_train_step"] += 1

    async def after_eval_step(self, trajectories: Sequence[Trajectory]) -> None:
        self.fn_invocations["after_eval_step"] += 1

    async def after_eval_loop(self) -> None:
        self.fn_invocations["after_eval_loop"] += 1

    async def after_update(self) -> None:
        self.fn_invocations["after_update"] += 1
