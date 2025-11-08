from collections.abc import Sequence
from unittest.mock import patch

import litellm
import pytest
from aviary.core import DummyEnv, TaskDataset

from ldp.agent import MemoryAgent, SimpleAgent
from ldp.alg import (
    Callback,
    Evaluator,
    EvaluatorConfig,
    MeanMetricsCallback,
    OfflineTrainer,
    OfflineTrainerConfig,
    OnlineTrainer,
    OnlineTrainerConfig,
    StoreTrajectoriesCallback,
)
from ldp.alg.datasets import (  # noqa: F401  # Force TASK_DATASET_REGISTRY update
    DummyTaskDataset,
)
from ldp.alg.optimizer import default_optimizer_factory
from ldp.data_structures import Trajectory
from ldp.graph import OpCtx


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("clear_ctx_at_each_iter", [True, False])
async def test_online_trainer(clear_ctx_at_each_iter: bool) -> None:
    agent = MemoryAgent()
    opt = default_optimizer_factory(agent)
    dataset = TaskDataset.from_name("dummy")
    dummy_callback = DummyCallback()
    metrics_callback = MeanMetricsCallback(train_dataset=dataset, track_tool_usage=True)

    train_conf = OnlineTrainerConfig(
        batch_size=1,
        num_train_iterations=1,
        max_rollout_steps=1,
        num_eval_iterations=1,
        num_rollouts_per_env=2,
        eval_every=1,
        clear_ctx_at_each_iter=clear_ctx_at_each_iter,
    )
    trainer = OnlineTrainer(
        config=train_conf,
        agent=agent,
        optimizer=opt,
        train_dataset=dataset,
        eval_dataset=dataset,
        callbacks=[dummy_callback, metrics_callback],
    )
    await trainer.train()

    for k, v in dummy_callback.counters.items():
        if k == "num_train_trajectories":
            # because num_rollouts_per_env==2
            assert v == 2
        elif "eval" in k:
            # eval is run 3 times: before training, during training, after training
            assert v == 3
        else:
            # after_{train_step,update} should be called once each
            assert v == 1
    assert metrics_callback.train_means["failures"] < 1, "Training should work"
    assert "tool_print_story" in metrics_callback.train_means

    if clear_ctx_at_each_iter:
        all(not ctx_data.data for ctx_data in OpCtx._CTX_REGISTRY.values())
    else:
        any(ctx_data.data for ctx_data in OpCtx._CTX_REGISTRY.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("clear_ctx_at_each_iter", [True, False])
async def test_evaluator(clear_ctx_at_each_iter) -> None:
    agent = SimpleAgent()
    dataset = TaskDataset.from_name("dummy")
    metrics_callback = MeanMetricsCallback(eval_dataset=dataset)
    count_callback = DummyCallback()

    eval_conf = EvaluatorConfig(
        num_eval_iterations=1,
        clear_ctx_at_each_iter=clear_ctx_at_each_iter,
    )
    evaluator = Evaluator(
        config=eval_conf,
        agent=agent,
        dataset=dataset,
        callbacks=[metrics_callback, count_callback],
    )
    with patch.object(DummyEnv, "close", autospec=True) as spy_close:
        await evaluator.evaluate()

    spy_close.assert_awaited_once()
    assert isinstance(metrics_callback.eval_means["reward"], float)
    assert "tool_print_story" not in metrics_callback.eval_means

    for k, v in count_callback.counters.items():
        assert v == (1 if "eval" in k else 0)

    if clear_ctx_at_each_iter:
        all(not ctx_data.data for ctx_data in OpCtx._CTX_REGISTRY.values())
    else:
        any(ctx_data.data for ctx_data in OpCtx._CTX_REGISTRY.values())


@pytest.mark.asyncio
async def test_can_measure_evaluation_failure_rate() -> None:
    dataset = TaskDataset.from_name("dummy")
    metrics_callback = MeanMetricsCallback(eval_dataset=dataset)

    evaluator = Evaluator(
        config=EvaluatorConfig(
            batch_size=1, num_eval_iterations=1, max_rollout_steps=2
        ),
        agent=SimpleAgent(),
        dataset=dataset,
        callbacks=[metrics_callback],
    )
    with (
        patch.object(
            type(evaluator.agent), "get_asv", side_effect=litellm.APIError
        ) as mock_get_asv,
        patch.object(DummyEnv, "close", autospec=True) as spy_close,
    ):
        await evaluator.evaluate()  # Confirm this does not crash
    mock_get_asv.assert_awaited_once()
    assert metrics_callback.eval_means["failures"] == 1.0
    spy_close.assert_awaited_once()


@pytest.mark.vcr
@pytest.mark.asyncio
@pytest.mark.parametrize("clear_ctx_at_each_iter", [True, False])
async def test_offline_trainer(clear_ctx_at_each_iter: bool) -> None:
    # This is kind of a system test of getting trajectories from the evaluator
    # and then training on them "offline"
    agent = MemoryAgent()
    opt = default_optimizer_factory(agent)
    dataset = TaskDataset.from_name("dummy")
    traj_callback = StoreTrajectoriesCallback()

    evaluator = Evaluator(
        config=EvaluatorConfig(
            num_eval_iterations=1,
            clear_ctx_at_each_iter=clear_ctx_at_each_iter,
        ),
        agent=agent,
        dataset=dataset,
        callbacks=[traj_callback],
    )
    await evaluator.run()
    assert len(traj_callback.eval_trajectories) == 1

    count_callback = DummyCallback()
    metrics_callback = MeanMetricsCallback(train_dataset=dataset)
    trainer = OfflineTrainer(
        config=OfflineTrainerConfig(
            batch_size=1,
            clear_ctx_at_each_iter=clear_ctx_at_each_iter,
            num_epochs=2,
        ),
        agent=agent,
        optimizer=opt,
        train_trajectories=traj_callback.eval_trajectories,
        callbacks=[count_callback, metrics_callback],
    )
    await trainer.train()

    assert count_callback.counters == {
        "after_train_step": 2,
        "after_eval_step": 0,
        "after_eval_loop": 0,
        "after_update": 2,
        "num_train_trajectories": 2,
    }
    assert metrics_callback.train_means["failures"] < 1, "Training should work"

    if clear_ctx_at_each_iter:
        all(not ctx_data.data for ctx_data in OpCtx._CTX_REGISTRY.values())
    else:
        any(ctx_data.data for ctx_data in OpCtx._CTX_REGISTRY.values())


class DummyCallback(Callback):
    def __init__(self):
        self.counters = {
            "after_train_step": 0,
            "after_eval_step": 0,
            "after_eval_loop": 0,
            "after_update": 0,
            "num_train_trajectories": 0,
        }

    async def after_train_step(self, trajectories: Sequence[Trajectory]) -> None:
        self.counters["after_train_step"] += 1
        self.counters["num_train_trajectories"] += len(trajectories)

    async def after_eval_step(self, trajectories: Sequence[Trajectory]) -> None:
        self.counters["after_eval_step"] += 1

    async def after_eval_loop(self) -> None:
        self.counters["after_eval_loop"] += 1

    async def after_update(self) -> None:
        self.counters["after_update"] += 1
