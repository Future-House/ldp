from enum import StrEnum


class DefaultLLMModelNames(StrEnum):
    """Defaults for LLM models, pin exact versions for performance stability."""

    OPENAI = "gpt-4o-2024-08-06"  # Cheap, fast, and decent


# Lower than LiteLLM's 10-min default: https://github.com/BerriAI/litellm/blob/v1.48.10/litellm/main.py#L859
DEFAULT_LLM_COMPLETION_TIMEOUT = 120  # seconds

# ruff: noqa: E402  # Avoid circular imports

from .agent import Agent, AgentConfig
from .agent_client import HTTPAgentClient, make_simple_agent_server
from .memory_agent import MemoryAgent
from .react_agent import ReActAgent
from .simple_agent import SimpleAgent, SimpleAgentState
from .tree_of_thoughts_agent import TreeofThoughtsAgent

__all__ = [
    "Agent",
    "AgentConfig",
    "DefaultLLMModelNames",
    "HTTPAgentClient",
    "MemoryAgent",
    "ReActAgent",
    "SimpleAgent",
    "SimpleAgentState",
    "TreeofThoughtsAgent",
    "make_simple_agent_server",
]
