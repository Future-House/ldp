import itertools
import random
import tempfile
from copy import deepcopy
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from aviary.core import Environment, Frame, Message, Tool, ToolRequestMessage
from litellm import AuthenticationError
from pydantic import BaseModel

from ldp.agent import Agent, SimpleAgent, SimpleAgentState
from ldp.alg import (
    BeamSearchRollout,
    Callback,
    RolloutManager,
    StoreEnvironmentsCallback,
    TreeSearchRollout,
)
from ldp.data_structures import Trajectory, Transition
from ldp.graph import FxnOp, OpResult, compute_graph, set_training_mode


class DummyEnv(Environment[None]):
    def __init__(self, instance_id: int | None = None):
        self.tools = [Tool.from_function(self.talk)]
        self._instance_id = instance_id
        self.close_mock = AsyncMock()

    async def get_id(self) -> str:
        if self._instance_id is None:
            raise ValueError("No instance ID was configured.")
        return str(self._instance_id)

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        return [Message(content="Hello!")], self.tools

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        if action.tool_calls:
            responses = cast("list[Message]", await self.exec_tool_calls(action))
        else:
            responses = [Message(content="Use the 'talk' tool to speak.")]

        return responses, 0.0, False, False

    async def talk(self, message: str) -> str:
        """Say something to me.

        Args:
            message (str): what you want to say

        Returns:
            str: my response
        """
        return message

    def export_frame(self) -> Frame:
        return Frame()

    async def close(self) -> None:
        await self.close_mock()


async def count_exclamations(traj: Trajectory) -> float:  # noqa: RUF029
    last_step = traj.steps[-1]
    agent_state = cast("SimpleAgentState", last_step.next_agent_state)
    return float(
        sum(m.content.count("!") for m in agent_state.messages if m.content is not None)
    )


@pytest.mark.vcr
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.asyncio
async def test_rollout(training: bool) -> None:
    agent = SimpleAgent()
    counter_callback = DummyCallback()
    env_storage_callback = StoreEnvironmentsCallback()
    set_training_mode(training)
    rollout_manager = RolloutManager(
        agent,
        catch_agent_failures=False,
        catch_env_failures=False,
        callbacks=[counter_callback, env_storage_callback],
    )
    envs = [DummyEnv(instance_id=1), DummyEnv()]
    trajs = await rollout_manager.sample_trajectories(environments=envs, max_steps=1)
    first_traj, second_traj = trajs
    assert first_traj.traj_id
    assert (
        first_traj.metadata.get("env_id")
        == "1"
        == await env_storage_callback.traj_id_to_envs[first_traj.traj_id].get_id()
    )
    assert second_traj.metadata.get("env_id") is None

    for env in envs:
        env.close_mock.assert_awaited_once_with()

    assert all(
        tx.metadata.get(f"time_elapsed_{fn}") is not None
        for fn in (
            "before_transition",
            "agent_get_asv",
            "after_agent_get_asv",
            "env_step",
            "after_env_step",
        )
        for traj in trajs
        for tx in traj.steps
    ), "All transitions should have timing metadata"

    # Let's check we can serialize and deserialize the trajectories
    for traj in trajs:
        assert traj.traj_id
        assert traj.traj_id in env_storage_callback.traj_id_to_envs
        with tempfile.NamedTemporaryFile(suffix=".jsonl") as f:
            await traj.to_jsonl(filename=f.name)
            rehydrated_traj = Trajectory.from_jsonl(f.name)
            assert traj.traj_id == rehydrated_traj.traj_id

    assert all(v == 2 for v in counter_callback.fn_invocations.values())


@pytest.mark.vcr
@pytest.mark.parametrize("fallback", [True, False])
@pytest.mark.asyncio
async def test_fallbacks_working(fallback: bool) -> None:
    AGENT_MODEL_LIST = [
        {
            "model_name": "openai/gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "abc123"},
        },
        {
            "model_name": "openai/gpt-4o",
            "litellm_params": {
                "model": "openai/gpt-4o",
            },
        },
    ]
    AGENT_ROUTER_KWARGS: dict[str, bool | list[dict[str, list[str]]]] = {
        "set_verbose": True,
    }
    if fallback:
        AGENT_ROUTER_KWARGS["fallbacks"] = [
            {
                "openai/gpt-4o-mini": [
                    "openai/gpt-4o",
                ]
            }
        ]

    AGENT_CONFIG = {
        "llm_model": {
            "name": "openai/gpt-4o-mini",
            "config": {
                "model_list": AGENT_MODEL_LIST,
                "router_kwargs": AGENT_ROUTER_KWARGS,
            },
        }
    }
    if fallback:
        AGENT_CONFIG["llm_model"]["config"]["fallbacks"] = [  # type: ignore[index]
            {
                "openai/gpt-4o-mini": [
                    "openai/gpt-4o",
                ]
            }
        ]
    agent = SimpleAgent(**AGENT_CONFIG)
    callback = DummyCallback()

    rollout_manager = RolloutManager(
        agent,
        catch_agent_failures=fallback,
        callbacks=[callback],
    )
    env = DummyEnv()
    if fallback:
        assert await rollout_manager.sample_trajectories(
            environments=[env], max_steps=2
        )
    else:
        with pytest.raises(AuthenticationError):
            await rollout_manager.sample_trajectories(environments=[env], max_steps=2)
    env.close_mock.assert_awaited_once_with()


async def adeepcopy(x):  # noqa: RUF029
    return deepcopy(x)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_beam_search() -> None:
    agent = SimpleAgent()
    callback = DummyCallback()
    beam_search = BeamSearchRollout(
        agent,
        beam_width=1,  # keep these numbers small to speed up test
        samples_per_beam=1,
        env_clone_fn=adeepcopy,
        agent_clone_fn=deepcopy,
        scoring_fn=count_exclamations,
        catch_agent_failures=False,
        catch_env_failures=False,
        callbacks=[callback],
    )

    trajs = await beam_search.sample_trajectories(
        environments=[DummyEnv(), DummyEnv()], max_steps=1
    )
    assert len(trajs) == 2

    assert all(
        v == 2
        for k, v in callback.fn_invocations.items()
        if k != "after_agent_init_state"  # TODO: support this callback too
    )


class DummyCallback(Callback):
    def __init__(self):
        # NOTE: don't use collections.defaultdict here because it can lead to
        # test aliasing for a callback being missed altogether
        self.fn_invocations = {
            "before_rollout": 0,
            "before_transition": 0,
            "after_agent_init_state": 0,
            "after_agent_get_asv": 0,
            "after_env_reset": 0,
            "after_env_step": 0,
            "after_transition": 0,
        }

    async def before_rollout(self, traj_id: str, env: Environment) -> None:
        self.fn_invocations["before_rollout"] += 1

    async def before_transition(
        self,
        traj_id: str,
        agent: Agent,
        env: Environment,
        agent_state: Any,
        obs: list[Message],
    ) -> None:
        self.fn_invocations["before_transition"] += 1

    async def after_agent_init_state(self, traj_id: str, init_state: Any) -> None:
        self.fn_invocations["after_agent_init_state"] += 1

    async def after_agent_get_asv(
        self,
        traj_id: str,
        action: OpResult[ToolRequestMessage],
        next_agent_state: Any,
        value: float,
    ):
        self.fn_invocations["after_agent_get_asv"] += 1

    async def after_env_reset(
        self, traj_id: str, obs: list[Message], tools: list[Tool]
    ) -> None:
        self.fn_invocations["after_env_reset"] += 1

    async def after_env_step(
        self,
        traj_id: str,
        obs: list[Message],
        reward: float,
        done: bool,
        trunc: bool,
    ):
        self.fn_invocations["after_env_step"] += 1

    async def after_transition(
        self,
        traj_id: str,
        agent: Agent,
        env: Environment,
        transition: Transition,
    ) -> None:
        self.fn_invocations["after_transition"] += 1


class CountingAgentState(BaseModel):
    count: float = 0.0


class CountingAgent(Agent[CountingAgentState]):
    def __init__(self):
        self.op = FxnOp[ToolRequestMessage](lambda: ToolRequestMessage(tool_calls=[]))

    async def init_state(self, tools: list[Tool]) -> CountingAgentState:
        return CountingAgentState()

    @compute_graph()
    async def get_asv(
        self, agent_state: CountingAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], CountingAgentState, float]:
        new_state = CountingAgentState(count=float(cast("str", obs[0].content)) + 1)
        action = await self.op()
        return action, new_state, 0.0


class CountingEnv(Environment[float]):
    def __init__(self, state: float = 0.0):
        self.state = state

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        return [Message(content=str(self.state))], []

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        self.state += 1
        return [Message(content=str(self.state))], 0.0, self.state >= 3, False

    def export_frame(self) -> Frame:
        return Frame()


@pytest.mark.asyncio
async def test_deterministic_rollout():
    agent = CountingAgent()
    env = CountingEnv()

    rollout_manager = RolloutManager(agent)
    traj, *_ = await rollout_manager.sample_trajectories(environments=[env])

    assert len(traj.steps) == 3
    for i_step, step in enumerate(traj.steps):
        f_step = float(i_step)
        # check that we didn't clobber any agent or env states
        assert step.agent_state.count == f_step
        assert step.next_agent_state.count == f_step + 1
        assert step.observation[0].content == str(f_step)
        assert step.next_observation[0].content == str(f_step + 1)


class NoisyCountingEnv(CountingEnv):
    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        self.state += 1 + random.uniform(-0.01, 0.01)
        return [Message(content=str(self.state))], 1.0, self.state >= 3, False


class TestTreeSearch:
    @pytest.mark.asyncio
    async def test_tree_search(self) -> None:
        agent = CountingAgent()
        # Use a slightly stochastic env so we can distinguish branches
        env = NoisyCountingEnv()

        callback = DummyCallback()
        rollout_manager = TreeSearchRollout(
            agent,
            branching_factor=2,
            env_clone_fn=deepcopy,
            concurrency_limit=1,
            callbacks=[callback],
        )
        tree = await rollout_manager.sample_tree(env, max_depth=3)
        trajs = tree.get_trajectories()
        assert len(trajs) == 8

        traj_ids_wo_root: set[str] = {
            cast("str", traj.traj_id).replace(tree.root_id, "").lstrip(":")
            for traj in trajs
        }
        # IDs should be 0:0:0, 0:0:1, ... 1:1:1 (order doesn't matter)
        assert traj_ids_wo_root == {
            ":".join(x) for x in itertools.product("01", repeat=3)
        }

        observations: dict[tuple[str, ...], str] = {}
        for traj in trajs:
            branch_path = tuple(cast("str", traj.traj_id).split(":")[1:])

            prev_step: Transition | None = None
            for i_step, step in enumerate(traj.steps):
                if prev_step is not None:
                    # Check that the child node started at the state emitted at the parent node
                    assert prev_step.next_agent_state == step.agent_state

                # Steps that started at the same node in the tree should have the same observation
                node_id = branch_path[: i_step + 1]
                assert len(step.observation) == 1
                assert step.observation[0].content
                if node_id in observations:
                    assert observations[node_id] == step.observation[0].content
                else:
                    observations[node_id] = step.observation[0].content

                prev_step = step

        for callback_fn, num_calls in callback.fn_invocations.items():
            if callback_fn in {
                "before_rollout",
                "after_agent_init_state",
                "after_env_reset",
            }:
                assert num_calls == 1, "These should be invoked once at the start"
            else:
                # We expect sum_{i=1}^3 2^i = 2^4 - 2 = 14 transitions:
                # - branching factor = 2, depth = 3
                # - root node isn't sampled, so no i=0 term in sum
                assert num_calls == 14

    @pytest.mark.asyncio
    async def test_early_stopping(self) -> None:
        agent = CountingAgent()
        # Use a slightly stochastic env so we can distinguish branches
        env = NoisyCountingEnv()

        callback = DummyCallback()
        rollout_manager = TreeSearchRollout(
            agent,
            branching_factor=2,
            env_clone_fn=deepcopy,
            concurrency_limit=1,
            callbacks=[callback],
            target_reward=0.5,
        )
        trajs = (await rollout_manager.sample_tree(env, max_depth=3)).get_trajectories()
        assert len(trajs) < 8  # should have exited early
        for traj in trajs:
            # should have hit target reward immediately
            assert len(traj.steps) == 1
