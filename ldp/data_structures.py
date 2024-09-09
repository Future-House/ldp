from __future__ import annotations

import json
import logging
import os
from typing import Any, ClassVar, Self, cast

import networkx as nx
from aviary.message import Message
from aviary.tools import ToolRequestMessage, ToolResponseMessage
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

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

    action: OpResult[ToolRequestMessage] | None = Field(
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

    @field_validator("action", mode="before")
    @classmethod
    def construct_action(
        cls, action: OpResult[ToolRequestMessage] | dict | None
    ) -> OpResult[ToolRequestMessage] | None:
        if isinstance(action, dict):
            return OpResult.from_dict(ToolRequestMessage, action)
        return action

    @property
    def failed(self) -> bool:
        """Get if an exception was encountered during rollout, for convenience.

        If True, this transition should not be trained on.
        Failed transitions are for debugging purposes.
        """
        return bool(self.metadata.get("exception"))

    def model_dump_json(self, *, indent: int | None = None, **kwargs) -> str:
        # The kwargs for model_dump are the same as super().model_dump_json,
        # with the exception of indent.
        dump = self.model_dump(**kwargs)
        if self.action is not None:
            dump["action"] = self.action.to_dict()
        return json.dumps(dump, indent=indent)


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


class TransitionTree:
    def __init__(self, root_id: str):
        """A tree of transitions.

        If A->B is an edge in this tree, then A and B are consecutive
        transitions in an LDP. Any path from the root node to a terminal
        node constitutes a complete LDP.

        Args:
            root_id: A unique identifier for the root node of the tree.
                All IDs of transitions added to this tree must begin with
                the same root_id.
        """
        self.root_id = root_id

        self.tree = nx.DiGraph()  # the actual tree
        self.rev_tree = nx.DiGraph()  # the same as self.tree, but with reversed edges

        self._add_node(root_id, transition=None)

    def _add_node(self, step_id: str, transition: Transition | None):
        self.tree.add_node(step_id, transition=transition)
        self.rev_tree.add_node(step_id)

    def _add_edge(self, parent_step_id: str, child_step_id: str):
        self.tree.add_edge(parent_step_id, child_step_id)
        self.rev_tree.add_edge(child_step_id, parent_step_id)

    def get_transition(self, step_id: str) -> Transition:
        if step_id == self.root_id:
            raise ValueError("Root node has no transition.")

        return cast(Transition, self.tree.nodes[step_id]["transition"])

    def add_transition(self, step_id: str, step: Transition):
        """Add a transition to the tree.

        Args:
            step_id: A unique identifier for the root node of the tree.
                The expected form of the step ID is "{parent step ID}:{step index}".
            step: The transition to add.
        """
        root_id, *step_ids = step_id.split(":")
        assert (
            root_id == self.root_id
        ), f"Step ID {step_id} does not start with root ID {self.root_id}"
        assert step_ids, "Step ID cannot be the same as the root ID."
        # TODO: maybe this should be warning?
        assert (
            step_id not in self.tree
        ), f"Step ID {step_id} already exists in the tree."

        self._add_node(step_id, transition=step)

        parent_id = ":".join([root_id, *step_ids[:-1]])
        if parent_id in self.tree:
            self._add_edge(parent_id, step_id)

    def get_trajectories(self) -> list[Trajectory]:
        """Return a list of trajectories.

        Since each path from the root node to a terminal node defines
        a unique trajectory, N(terminal node) trajectories will be returned.
        The trajectory ID will be set to the ID of the terminal step.

        Note that we include failed and truncated trajectories; it is up to the
        caller to decide what to do them.

        Returns:
            All trajectories in this tree.
        """
        trajs = []
        step: Transition | None

        for step_id, step in self.tree.nodes(data="transition"):
            if not step:
                # root node
                continue

            is_terminal = (
                # check terminal conditions in increasing order of expense
                step.done
                or step.truncated
                or step.failed
                or self.tree.out_degree(step_id) == 0
            )

            if not is_terminal:
                continue

            # set the ID to the terminal node, which uniquely identifies the path
            traj = Trajectory(traj_id=step_id)
            # Build the trajectory up from a terminal node
            current_step: Transition | None = step
            current_step_id = step_id

            # Walk backwards towards the root (current_step=None)
            while current_step:
                traj.steps.append(current_step)

                parent_step_id, *extra = list(self.rev_tree.successors(current_step_id))
                assert not extra, f"Expected a single parent, but got {len(extra) + 1}"

                current_step_id = parent_step_id
                current_step = self.tree.nodes[parent_step_id]["transition"]

            # would've added things in reverse order, so fix that here
            traj.steps.sort(key=lambda x: x.timestep)
            trajs.append(traj)

        return trajs

    def assign_mc_value_estimates(self, discount_factor: float = 1.0):
        """Assign Monte Carlo state-action value estimates to each transition (in-place).

        Args:
            discount_factor: The discount factor to use when computing cumulative
                future rewards.
        """
        for step_id in nx.topological_sort(self.rev_tree):
            step: Transition | None = self.tree.nodes[step_id]["transition"]
            if step is None:
                continue

            if children := list(self.tree.successors(step_id)):
                # V_{t+1}(s') = sum_{a'} p(a'|s') * Q_{t+1}(s', a')
                # Here we assume p(a'|s') is uniform.
                # TODO: don't make that assumption where a logprob is available
                v_tp1 = sum(
                    self.get_transition(child_id).value for child_id in children
                ) / len(children)
            else:
                v_tp1 = 0.0

            # Q_t(s_t, a_t) = r_{t+1} + gamma * V_{t+1}(s_{t+1})
            # (we are assuming the environment is deterministic)
            step.value = step.reward + discount_factor * v_tp1
