from enum import StrEnum


class DefaultLLMModelNames(StrEnum):
    """Defaults for LLM models, pin exact versions for performance stability."""

    OPENAI = "gpt-4o-2024-08-06"  # Cheap, fast, and decent


# ruff: noqa: E402  # Avoid circular imports

from .agent import Agent, AgentConfig
from .agent_client import HTTPAgentClient, make_simple_agent_server
from .dqn_agent import DQNAgent, MultipleCompletionLLMCallOp
from .memory_agent import MemoryAgent
from .react_agent import ReActAgent
from .simple_agent import SimpleAgent, SimpleAgentState
from .tree_of_thoughts_agent import TreeofThoughtsAgent

__all__ = [
    "Agent",
    "AgentConfig",
    "DQNAgent",
    "DefaultLLMModelNames",
    "HTTPAgentClient",
    "MemoryAgent",
    "MultipleCompletionLLMCallOp",
    "ReActAgent",
    "SimpleAgent",
    "SimpleAgentState",
    "TreeofThoughtsAgent",
    "make_simple_agent_server",
]