"""
A module is a subgraph of a compute graph that can be exposed like a single node/op.

An analogous entity in PyTorch is torch.nn.Module.
"""

from .llm_call import ParsedLLMCallModule
from .react import (
    MalformedMessageError,
    ReActModule,
    ToolDescriptionMethods,
    parse_message,
)
from .reflect import ReflectModule, ReflectModuleConfig
from .thought import ThoughtModule
from .value_function import DQNOp, DQNPolicyModule, EmbeddingDQNOp, EpsilonGreedyOp

__all__ = [
    "DQNOp",
    "DQNPolicyModule",
    "EmbeddingDQNOp",
    "EpsilonGreedyOp",
    "MalformedMessageError",
    "ParsedLLMCallModule",
    "ReActModule",
    "ReflectModule",
    "ReflectModuleConfig",
    "ThoughtModule",
    "ToolDescriptionMethods",
    "parse_message",
]
