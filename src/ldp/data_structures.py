from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Hashable, Iterable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast
from uuid import UUID

import aiofiles
from aviary.core import Message, ToolRequestMessage, ToolResponseMessage, join
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from ldp.graph import OpResult
from ldp.graph.op_utils import _lazy_import_networkx
from ldp.utils import discounted_returns

if TYPE_CHECKING:
    from aviary.core import Environment


logger = logging.getLogger(__name__)


class Transition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Sentinel value for missing observation, as opposed to empty observation
    # Only used for tests; a user should never use this.
    NO_OBSERVATION: ClassVar[list[Message]] = []

    timestep: int = Field(description="Zero-indexed MDP timestep t.")

    agent_state: Any = Field(
        description=(
            "Agent.get_asv's input. This is `s_t` in the POMDP. Note that `s_0` comes"
            " from `Agent.init_state()`"
        )
    )
    next_agent_state: Any = Field(
        description="Agent.get_asv's output. This is s_t+1 in the POMDP."
    )

    observation: list[ToolResponseMessage | Message] = Field(
        description="Agent.get_asv's input. This is o_t in the POMDP."
    )
    next_observation: list[ToolResponseMessage | Message] = Field(
        description="Environment.step output. This is o_t+1 in the POMDP."
    )

    action: OpResult[ToolRequestMessage] | None = Field(
        default=None, description="Agent.get_asv output. This is a_t in the POMDP."
    )

    reward: float = Field(
        default=0.0, description="Environment.step output. This is r_t in the POMDP."
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
        # TODO: decide if we should override Transition.model_dump() instead.
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
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description=(
            "Optional JSON metadata on the trajectory, for example the ID of a task"
            " this trajectory was run on."
        ),
    )

    @property
    def failed(self) -> bool:
        return any(step.failed for step in self.steps)

    @property
    def done(self) -> bool:
        if not self.steps:
            return False
        return self.steps[-1].done

    async def to_jsonl(self, filename: str | os.PathLike) -> None:
        async with aiofiles.open(filename, "w") as f:
            await f.write(json.dumps(self.traj_id) + "\n")
            for s in self.steps:
                await f.write(s.model_dump_json() + "\n")

    @classmethod
    async def from_env(cls, env: Environment, **kwargs) -> Self:
        """Create a Trajectory while propagating environment ID."""
        traj_metadata = kwargs.pop("metadata", {})
        with suppress(NotImplementedError, ValueError):
            traj_metadata["env_id"] = await env.get_id()
        return cls(metadata=traj_metadata, **kwargs)

    @classmethod
    def from_jsonl(cls, filename: str | os.PathLike) -> Self:
        with open(filename, encoding="utf-8") as f:
            reader = iter(f)
            traj = cls(traj_id=json.loads(next(reader)))
            for json_line in reader:
                data = json.loads(json_line)
                # logprob may have been serialized, but cannot be passed to
                # OpResult, so remove it here.
                with suppress(KeyError):
                    data["action"].pop("logprob")
                traj.steps.append(Transition(**data))
        return traj

    def compute_discounted_returns(self, discount: float = 1.0) -> list[float]:
        """Compute the discounted returns for each step in the trajectory."""
        return discounted_returns(
            rewards=[step.reward for step in self.steps],
            terminated=[step.truncated for step in self.steps],
            discount=discount,
        )


class TransitionTree:
    def __init__(self, root_id: str | UUID):
        """A tree of transitions.

        If A->B is an edge in this tree, then A and B are consecutive
        transitions in an LDP. Any path from the root node to a terminal
        node constitutes a complete LDP.

        A node may be assigned an arbitrary weight, which will be treated
        as a relative probability of sampling that node. For example, if
        A(weight=1) and B(w=2) are both children of the same node,
        then we treat B as twice as likely as A.

        Args:
            root_id: A unique identifier for the root node of the tree.
                All IDs of transitions added to this tree must begin with
                the same root_id.
        """
        self.root_id = str(root_id)

        nx = _lazy_import_networkx()
        self.tree = nx.DiGraph()  # the actual tree
        self.rev_tree = nx.DiGraph()  # the same as self.tree, but with reversed edges

        self._add_node(self.root_id, transition=None, weight=1.0)

    def _add_node(
        self, step_id: str, transition: Transition | None, weight: float
    ) -> None:
        self.tree.add_node(step_id, transition=transition, weight=weight)
        self.rev_tree.add_node(step_id)

    def _add_edge(self, parent_step_id: str, child_step_id: str) -> None:
        self.tree.add_edge(parent_step_id, child_step_id)
        self.rev_tree.add_edge(child_step_id, parent_step_id)

    def get_transition(self, step_id: str) -> Transition:
        if step_id == self.root_id:
            raise ValueError("Root node has no transition.")

        return cast("Transition", self.tree.nodes[step_id]["transition"])

    def get_weight(self, step_id: str) -> float:
        return cast("float", self.tree.nodes[step_id]["weight"])

    def add_transition(
        self, step_id: str, step: Transition, weight: float = 1.0
    ) -> None:
        """Add a transition to the tree.

        Args:
            step_id: A unique identifier for this node in the tree.
                The expected form of the step ID is "{parent step ID}:{step index}".
            step: The transition to add.
            weight: Weight of the transition. Defaults to 1.0.
        """
        root_id, *step_ids = step_id.split(":")
        assert root_id == self.root_id, (
            f"Step ID {step_id} does not start with root ID {self.root_id}"
        )
        assert step_ids, "Step ID cannot be the same as the root ID."
        # TODO: maybe this should be warning?
        assert step_id not in self.tree, (
            f"Step ID {step_id} already exists in the tree."
        )

        self._add_node(step_id, transition=step, weight=weight)

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

    def assign_mc_value_estimates(self, discount_factor: float = 1.0) -> None:
        """Assign Monte Carlo state-action value estimates to each transition (in-place).

        Args:
            discount_factor: The discount factor to use when computing cumulative
                future rewards.
        """
        for step_id in _lazy_import_networkx().topological_sort(self.rev_tree):
            step: Transition | None = self.tree.nodes[step_id]["transition"]
            if step is None:
                continue

            if children := list(self.tree.successors(step_id)):
                # V_{t+1}(s') = sum_{a'} p(a'|s') * Q_{t+1}(s', a')
                # Here we assume p(a'|s') is uniform over the sampled actions.
                # TODO: don't make that assumption where a logprob is available
                weights = [self.get_weight(child_id) for child_id in children]
                steps = [self.get_transition(child_id) for child_id in children]
                v_tp1 = sum(
                    w * step.value for w, step in zip(weights, steps, strict=True)
                ) / sum(weights)
            else:
                v_tp1 = 0.0

            # Q_t(s_t, a_t) = r_{t+1} + gamma * V_{t+1}(s_{t+1})
            # (we are assuming the environment is deterministic)
            step.value = step.reward + discount_factor * v_tp1

    def compute_advantages(self) -> None:
        """Replace Transition.value with an advantage (in-place).

        A(s, a) = Q(s, a) - V(s), where V(s) is estimated as the
        average of Q(s, a') over all a' sampled at s.

        TODO: put this in Transition.metadata['advantage']. Not doing
        this right now due to implementation details in an optimizer.
        """
        state_values: dict[str, float] = {}

        for step_id in cast(
            "Iterable[str]", _lazy_import_networkx().topological_sort(self.tree)
        ):
            # topological sort means we will update a parent node in-place before
            # descending to its children

            step: Transition | None = self.tree.nodes[step_id]["transition"]
            if step is None:
                state_values[step_id] = 0.0
                continue

            # First, update V_t so that we can compute A_{t+1} for children
            children = [
                self.tree.nodes[child_id] for child_id in self.tree.successors(step_id)
            ]
            if children:
                state_action_values = [child["transition"].value for child in children]
                weights = [child["weight"] for child in children]
                state_values[step_id] = sum(
                    w * v for w, v in zip(weights, state_action_values, strict=True)
                ) / sum(weights)

            # Now compute A_t and replace Q_t with it in-place
            # Note that we are guaranteed at least one parent, since the `step is None`
            # check above should have caught the root node.
            parent_id, *extra = list(self.rev_tree.successors(step_id))
            assert not extra, "self.tree is not a tree!"
            step.value -= state_values[parent_id]
            # TODO: switch to the following, instead of overwriting step.value.
            # See docstring for explanation.
            # step.metadata["advantage"] = step.value - state_values[parent_id]

    def remove_nonterminal_branches(self) -> TransitionTree:
        """Creates a new tree with only branches that end in terminal states (done=True).

        TODO: refactor this to not use trajectories. See the note in merge_identical_nodes
        for reasoning.
        """
        new_tree = TransitionTree(self.root_id)
        for trajectory in self.get_trajectories():
            if not trajectory.done:
                continue

            traj_id_parts = cast("str", trajectory.traj_id).split(":")

            for step in trajectory.steps:
                step_id = ":".join(traj_id_parts[: step.timestep + 2])
                if step_id not in new_tree.tree:
                    # Traversing the tree by traversing trajectories means we may
                    # visit early nodes multiple times. Only add if we haven't visited
                    # already.
                    new_tree.add_transition(
                        step_id=step_id,
                        step=step,
                        weight=self.get_weight(step_id),
                    )

        return new_tree

    def merge_identical_nodes(
        self,
        agent_state_hash_fn: Callable[[Any], Hashable],
        observation_hash_fn: Callable[
            [list[ToolResponseMessage | Message]], Hashable
        ] = join,
        next_observation_hash_fn: Callable[
            [list[ToolResponseMessage | Message]], Hashable
        ] = join,
    ) -> TransitionTree:
        """Merge nodes with identical (state, observation, action)s. Returns a new tree.

        NOTE: the step IDs of nodes will lose their lineage after merging nodes. For example,
        the parent of ROOT:0:0 may not be ROOT:0 if ROOT:0 got merged with ROOT:1. Algorithms
        that rely on step IDs (like remove_nonterminal_branches) will not work as expected.

        Args:
            agent_state_hash_fn: A function that returns a hashable representation
                of the agent state of a transition.
            observation_hash_fn: A function that returns a hashable representation
                of the observation messages of a transition.
            next_observation_hash_fn: A function that returns a hashable representation
                of the next observation messages of a transition.
        """
        new_tree = TransitionTree(self.root_id)

        # step hash -> step ID
        seen_hash_to_step_id: dict[int, str] = {}
        # old step ID -> new step ID
        node_remap: dict[str, str] = {self.root_id: self.root_id}

        for step_id in _lazy_import_networkx().topological_sort(self.tree):
            step: Transition | None = self.tree.nodes[step_id]["transition"]
            if step is None:
                continue

            state_hash = agent_state_hash_fn(step.agent_state)

            if step.action is not None:
                tool_request_msg = step.action.value
                # NOTE: we are ignoring tool call ID in the comparison of tool requests.
                # Thus, the tool call ID is excluded from the hash, so we can't just
                # simply call str(tool_request_msg)
                action_str = (tool_request_msg.content or "") + " ".join(
                    str(tc) for tc in tool_request_msg.tool_calls
                )
            else:
                action_str = ""

            step_hash = hash((
                state_hash,
                action_str,
                # (s, a, o): works for deterministic envs
                observation_hash_fn(step.observation),
                # (s, a, o, o'): works for both deterministic and stochastic envs
                next_observation_hash_fn(step.next_observation),
            ))
            step_weight = self.get_weight(step_id)

            if step_hash in seen_hash_to_step_id:  # Seen: merge
                merged_step_id = node_remap[step_id] = seen_hash_to_step_id[step_hash]
                # Not sure if this is the fastest way to do this
                new_tree.tree.nodes[merged_step_id]["weight"] += step_weight
            else:  # Unseen: don't merge
                node_remap[step_id] = seen_hash_to_step_id[step_hash] = step_id
                parent_id = node_remap[":".join(step_id.split(":")[:-1])]

                # manually add transitions, since the step_id substring relationship
                # will be broken
                new_tree._add_node(step_id, transition=step, weight=step_weight)
                new_tree._add_edge(parent_id, step_id)

        return new_tree
