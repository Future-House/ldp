from __future__ import annotations

import json
import logging
import os
from typing import Any, ClassVar, Self

from aviary.message import Message
from aviary.tools import ToolRequestMessage, ToolResponseMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ldp.alg.algorithms import discounted_returns
from ldp.graph.ops import OpResult

logger = logging.getLogger(__name__)


class Transition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Sentinel value for missing observation, as opposed to empty observation
    # Only used for tests; a user should never use this.
    NO_OBSERVATION: ClassVar[list[Message]] = []

    timestep: int = Field(description="Zero-indexed MDP timestep t.")

    agent_state: Any = Field(
        description=(
            "Agent.get_asv's input. This is `s_t` in RL terms. Note that `s_0` comes"
            " from `Agent.init_state()`"
        )
    )
    next_agent_state: Any = Field(
        description="Agent.get_asv's output. This is s_t+1 in RL terms."
    )

    observation: list[ToolResponseMessage | Message] = Field(
        description="Agent.get_asv's input. This is o_t in RL terms."
    )
    next_observation: list[ToolResponseMessage | Message] = Field(
        description="Environment.step output. This is o_t+1 in RL terms."
    )

    # Non-ToolRequestMessage OpResult is here to allow for testing hacks
    action: OpResult[ToolRequestMessage] | OpResult | None = Field(
        default=None, description="Agent.get_asv output. This is a_t in RL terms."
    )

    reward: float = Field(
        default=0.0, description="Environment.step output. This is r_t in RL terms."
    )

    truncated: bool = Field(
        default=False, description="timestep t's Environment.step output."
    )
    done: bool = Field(
        default=False, description="timestep t's Environment.step output."
    )
    value: float = Field(
        default=0.0,
        description=(
            "Value estimate output from timestep t's Agent.get_asv. This is v(s_t)"
            " [state value function] or q(s_t, a_t) [state-action value]."
        ),
    )
    # JsonValue so we can serialize
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """Get if an exception was encountered during rollout, for convenience.

        If True, this transition should not be trained on.
        Failed transitions are for debugging purposes.
        """
        return bool(self.metadata.get("exception"))


class Trajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    traj_id: str | None = None
    steps: list[Transition] = Field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(step.failed for step in self.steps)

    @property
    def done(self) -> bool:
        if not self.steps:
            return False
        return self.steps[-1].done

    def to_jsonl(self, filename: str | os.PathLike) -> None:
        with open(filename, "w") as f:
            f.write(json.dumps(self.traj_id) + "\n")
            f.writelines(s.model_dump_json() + "\n" for s in self.steps)

    @classmethod
    def from_jsonl(cls, filename: str | os.PathLike) -> Self:
        with open(filename) as f:
            reader = iter(f)
            traj = cls(traj_id=json.loads(next(reader)))
            for json_line in reader:
                traj.steps.append(Transition(**json.loads(json_line)))
        return traj

    def compute_discounted_returns(self, discount: float = 1.0) -> list[float]:
        return discounted_returns(
            rewards=[step.reward for step in self.steps],
            terminated=[step.truncated for step in self.steps],
            discount=discount,
        )
