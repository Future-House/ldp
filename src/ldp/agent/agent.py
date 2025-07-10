from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any, Generic, TypeVar

import numpy as np
from aviary.core import Message, Tool, ToolRequestMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ldp.graph import IdentityOp, Op, OpResult
from ldp.graph.ops import ResultOrValue

try:
    # So we can skip torch objects when looking for Ops
    import torch
except ImportError:
    # If torch is not available, then it won't be used in an Agent anyway
    torch = None  # type: ignore[assignment]

TAgentState = TypeVar("TAgentState")


# A global registry of all Agent subclasses, so we can look them up by name
_AGENT_REGISTRY: dict[str, type[Agent]] = {}


class Agent(ABC, Generic[TAgentState]):
    def __init_subclass__(cls, **kwargs):
        """Ensure Ops have unique names and subclasses are in _AGENT_REGISTRY."""
        super().__init_subclass__(**kwargs)

        original_init = cls.__init__

        def init_with_op_naming(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            # loop through Ops and give them proper names
            for name, op in _find_ops(self):
                op.set_name(name)

        cls.__init__ = init_with_op_naming  # type: ignore[method-assign]

        # Register the Agent subclass.
        _AGENT_REGISTRY[cls.__name__] = cls

    @abstractmethod
    async def get_asv(
        self, agent_state: TAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], TAgentState, float]:
        """
        Get new action, state, and value given state and observation messages.

        NOTE: the method's name has action listed before state to help you realize it's
        a new state.

        Args:
            agent_state: Optional current agent state, pass None if irrelevant.
                This can be something like agent memory.
            obs: Most recent list of observation messages from the environment's steps.
                If more observations than the most recent list are necessary, track them
                in the agent state.

        Returns:
            Three-tuple of new action, new agent state, and estimated value. The
                agent_state is returned as a copy so that you can safely mutate it
                without affecting the original. The estimated value is the agent's
                estimate of the future rewards given the input state and observations,
                and is used for RL training.  If estimated value doesn't matter, just
                return 0. The value could also come from a Q-value evaluated at the
                action chosen by the agent.
        """

    @abstractmethod
    async def init_state(self, tools: list[Tool]) -> TAgentState:
        """Initializes the first agent state with the provided tools."""

    def named_ops(self) -> Iterable[tuple[str, Op]]:
        """Analogous to torch.nn.Module.named_parameters()."""
        return _find_ops(self)

    @classmethod
    def from_name(cls, name: str, **kwargs) -> Agent:
        return _AGENT_REGISTRY[name](**kwargs)

    @classmethod
    async def wrap_action(
        cls, action: ResultOrValue[ToolRequestMessage]
    ) -> OpResult[ToolRequestMessage]:
        """Wraps the action in an OpResult, if it isn't already."""
        if isinstance(action, OpResult):
            return action
        return await IdentityOp[ToolRequestMessage]()(action)


class AgentConfig(BaseModel):
    """Configuration for specifying the type of agent i.e. the subclass of Agent above."""

    model_config = ConfigDict(extra="forbid")

    agent_type: str = Field(
        description=(
            "The type of agent to be used. This should be a subclass of Agent above."
        ),
    )
    agent_kwargs: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Keyword arguments to pass to the agent's constructor.",
    )

    def construct_agent(self) -> Agent:
        return Agent.from_name(self.agent_type, **self.agent_kwargs)

    def __hash__(self) -> int:
        return hash(self.agent_type + json.dumps(self.agent_kwargs, sort_keys=True))


def _find_ops(  # noqa: C901
    root: object, root_name: str = "", visited: set[int] | None = None
) -> Iterable[tuple[str, Op]]:
    """Recursive function to find children that are Ops and the attr chain to reach them.

    E.g. if root.module.op is an Op, then we will yield ("module.op", root.module.op).
    These are not fully qualified names, but more like "locally qualified names". In the above
    example, "root." + "module.op" is the fully qualified name.
    This is an internal function - Agent.named_ops() should usually suffice.

    Args:
        root: Any object that might hold Ops.
        root_name: The name of the root object. Defaults to empty string and is passed as an arg to
            make this method recursive.
        visited: a set of visited node IDs to avoid loops. Defaults to None.

    Yields:
        Two-tuple of (locally qualified name, Op) pairs
    """
    # Recursive function to find children that are Ops and the
    # attribute chain to reach them.
    if visited is None:
        visited = set()

    if isinstance(root, Op):
        yield root_name, root
        # Assume an Op may not have sub-Ops. I think this is sound, since
        # we wouldn't be tracking the compute graph properly if it did.
        return

    if "__pydantic_parent_namespace__" in root_name:
        # Skip Pydantic internals
        return

    # Don't recurse into PyTorch objects because they won't contain Ops
    if torch is not None and (  # type: ignore[redundant-expr]
        isinstance(root, torch.Tensor | torch.nn.Module)
    ):
        return

    # Similarly for numpy
    if isinstance(root, np.ndarray):
        return

    # loop through 3 types of containers: dicts, collections, and objects
    if isinstance(root, Mapping):
        named_attrs: Any = root.items()
    elif isinstance(root, Sequence | Collection) and not isinstance(root, str | bytes):
        named_attrs = enumerate(root)
    elif hasattr(root, "__dict__"):
        # object?
        named_attrs = root.__dict__.items()
    else:
        # couldn't descend
        return

    for k, v in named_attrs:
        id_v = id(v)
        if id_v not in visited:
            # only visit each object once - avoid loops, etc.
            visited.add(id_v)
            if root_name:
                k = f"{root_name}.{k}"
            yield from _find_ops(v, root_name=k, visited=visited)
