"""
A module is a subgraph of a compute graph that can be exposed like a single node/op.

An analogous entity in PyTorch is torch.nn.Module.
"""

from .llm_call import ParsedLLMCallModule
from .react import (
    ReActModule,
    ReActModuleSinglePrompt,
    ToolDescriptionMethods,
    parse_message,
)
from .reflect import ReflectModule, ReflectModuleConfig
from .thought import ThoughtModule

__all__ = [
    "ParsedLLMCallModule",
    "ReActModule",
    "ReActModuleSinglePrompt",
    "ReflectModule",
    "ReflectModuleConfig",
    "ThoughtModule",
    "ToolDescriptionMethods",
    "parse_message",
]
