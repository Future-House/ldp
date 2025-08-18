"""Test file for code examples extracted from the README.md file with a no-extra install."""

from dataclasses import dataclass, field
from datetime import datetime

import pytest

# =============================================================================
# Running an Agent on an Aviary Environment
# =============================================================================


@pytest.mark.asyncio
async def test_agent_on_aviary():
    """Test minimal example of running a language agent on an Aviary environment."""
    from aviary.core import DummyEnv

    from ldp.agent import SimpleAgent

    env = DummyEnv()
    agent = SimpleAgent()

    obs, tools = await env.reset()
    agent_state = await agent.init_state(tools=tools)

    # Run one step
    action, agent_state, _ = await agent.get_asv(agent_state, obs)
    obs, reward, done, truncated = await env.step(action.value)

    # Basic assertions
    assert action is not None
    assert agent_state is not None
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)


# =============================================================================
# SimpleAgent Definition
# =============================================================================


def test_simple_agent_definition():
    """Test defining LDP's SimpleAgent."""
    from ldp.agent import Agent
    from ldp.graph import LLMCallOp

    class AgentState:
        def __init__(self, messages, tools):
            self.messages = messages
            self.tools = tools

    class TestSimpleAgent(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.llm_call_op = LLMCallOp()

        async def init_state(self, tools):
            return AgentState([], tools)

        async def get_asv(self, agent_state, obs):
            # For testing purposes, create a mock action
            from aviary.core import ToolRequestMessage

            action = await Agent.wrap_action(ToolRequestMessage(tool_calls=[]))
            new_state = AgentState(
                messages=agent_state.messages + obs + [action], tools=agent_state.tools
            )
            return action, new_state, 0.0

    # Test that the class can be instantiated
    agent = TestSimpleAgent()
    assert agent is not None
    assert hasattr(agent, "llm_call_op")


# =============================================================================
# NoThinkAgent - Plain Python Agent
# =============================================================================


def test_no_think_agent():
    """Test minimal example of a deterministic Agent using plain Python."""
    from aviary.core import ToolCall, ToolRequestMessage

    from ldp.agent import Agent

    class NoThinkAgent(Agent):
        async def init_state(self, tools):
            return None

        async def get_asv(self, agent_state, obs):
            tool_call = ToolCall.from_name("specific_tool_call", arg1="foo")
            action = ToolRequestMessage(tool_calls=[tool_call])
            return await Agent.wrap_action(action), None, 0.0

    # Test that the class can be instantiated
    agent = NoThinkAgent()
    assert agent is not None


# =============================================================================
# Stochastic Computation Graph (SCG) - Basic Example
# =============================================================================


@pytest.mark.asyncio
async def test_basic_scg():
    """Test basic example of using stochastic computation graph."""
    from ldp.graph import FxnOp, compute_graph

    op_a = FxnOp(lambda x: 2 * x)

    async with compute_graph():
        op_result = await op_a(3)

    # Test that we got a result
    assert op_result is not None


# =============================================================================
# Complex SCG Example - Agent with Memory (Conceptual)
# =============================================================================


def test_memory_agent_structure():
    """Test the structure of the MemoryAgent class (conceptual only)."""

    class MemoryAgent:
        """Example agent that possesses memory using SCG."""

        def __init__(self):
            # These would be defined in the actual implementation
            self._query_factory_op = None
            self._memory_op = None
            self._format_memory_op = None
            self._prompt_op = None
            self._package_op = None
            self._config_op = None
            self._llm_call_op = None
            self.num_memories = 5
            self.memory_prompt = "Memory context:"

        def has_memory_components(self):
            """Check if the agent has all expected memory components."""
            return hasattr(self, "num_memories") and hasattr(self, "memory_prompt")

    # Test that the agent structure is correct
    memory_agent = MemoryAgent()
    assert memory_agent.has_memory_components()
    assert memory_agent.num_memories == 5
    assert memory_agent.memory_prompt == "Memory context:"


# =============================================================================
# Generic Support - Custom State Type
# =============================================================================


def test_custom_state_agent():
    """Test using Python generics with custom state type."""
    from ldp.agent import Agent

    @dataclass
    class MyComplexState:
        vector: list[float]
        timestamp: datetime = field(default_factory=datetime.now)

    class MyAgent(Agent[MyComplexState]):
        """Some agent who is now type checked to match the custom state."""

        async def init_state(self, tools):
            return MyComplexState(vector=[1.0, 2.0, 3.0])

        async def get_asv(self, agent_state, obs):
            from aviary.core import ToolRequestMessage

            action = await Agent.wrap_action(ToolRequestMessage(tool_calls=[]))
            return action, agent_state, 0.0

    # Test that the classes are defined correctly
    assert MyComplexState is not None
    assert MyAgent is not None

    # Create an instance of the state
    state_instance = MyComplexState(vector=[1.0, 2.0, 3.0])
    assert state_instance.vector == [1.0, 2.0, 3.0]
    assert isinstance(state_instance.timestamp, datetime)

    # Test that agent can be instantiated
    agent = MyAgent()
    assert agent is not None


# =============================================================================
# Rollout - Basic Example
# =============================================================================
@pytest.mark.asyncio
async def test_rollout():
    """Test basic example of running a rollout."""
    from aviary.core import DummyEnv

    from ldp.agent import SimpleAgent
    from ldp.alg import RolloutManager

    env = DummyEnv()
    agent = SimpleAgent()

    rollout_manager = RolloutManager(agent)
    traj, *_ = await rollout_manager.sample_trajectories(environments=[env])

    assert traj is not None
