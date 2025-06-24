import networkx as nx
import pytest
from aviary.core import Message, ToolResponseMessage, join

from ldp.data_structures import Transition, TransitionTree


def test_tree_mc_value():
    root_id = "dummy"
    tree = TransitionTree(root_id=root_id)

    kw = {
        "agent_state": None,
        "next_agent_state": None,
        "observation": Transition.NO_OBSERVATION,
        "next_observation": Transition.NO_OBSERVATION,
        "action": None,
    }

    # Construct a tree with some rewards scattered about
    tree.add_transition(f"{root_id}:0", Transition(timestep=0, reward=0.0, **kw))

    tree.add_transition(f"{root_id}:0:0", Transition(timestep=1, reward=1.0, **kw))
    for i in range(3):
        tree.add_transition(
            f"{root_id}:0:0:{i}",
            Transition(timestep=2, reward=float(i), done=True, **kw),
        )

    tree.add_transition(
        f"{root_id}:0:1", Transition(timestep=1, reward=-1.0, done=True, **kw)
    )

    tree.assign_mc_value_estimates(discount_factor=0.9)

    # Now make sure the value estimates are as expected
    # First, check the terminal nodes: Q==reward
    for i in range(3):
        assert tree.get_transition(f"{root_id}:0:0:{i}").value == float(i)
    assert tree.get_transition(f"{root_id}:0:1").value == -1.0

    # Then go up the tree
    assert tree.get_transition(f"{root_id}:0:0").value == pytest.approx(
        1 + 0.9 * ((0 + 1 + 2) / 3), rel=0.001
    )
    assert tree.get_transition(f"{root_id}:0").value == pytest.approx(
        0.0 + 0.9 * ((1.9 - 1) / 2), rel=0.001
    )

    # Check we can compute advantages w/o crashing for now. TODO: test the assigned
    # advantages. Will do so after the TODO in compute_advantages() is resolved.
    tree.compute_advantages()


def test_tree_node_merging() -> None:
    root_id = "dummy"
    tree = TransitionTree(root_id=root_id)

    kw = {
        "next_agent_state": None,
        "observation": Transition.NO_OBSERVATION,
        "next_observation": Transition.NO_OBSERVATION,
        "action": None,
    }

    # Construct a tree with two identical nodes
    for child_id in ("0", "1"):
        tree.add_transition(
            f"{root_id}:{child_id}",
            Transition(timestep=0, reward=0.0, agent_state=0, **kw),
        )
    # Add some identical nodes from a stochastic environment
    for child_id, next_o_content in zip(("2", "3"), ("stub", "stub 123"), strict=True):
        tree.add_transition(
            f"{root_id}:{child_id}",
            Transition(
                timestep=0,
                reward=0.0,
                agent_state=0,
                **(kw | {"next_observation": [Message(content=next_o_content)]}),
            ),
        )

    # Now some children nodes
    for parent in ("0", "1"):
        tree.add_transition(  # This tests merge of child nodes too
            f"{root_id}:{parent}:0",
            Transition(timestep=1, reward=0.0, agent_state=1, **kw),
        )
    for parent in ("2", "3"):
        tree.add_transition(
            f"{root_id}:{parent}:0",
            Transition(timestep=1, reward=0.0, agent_state=9, **kw),
        )

    # Tree at this stage is:
    # ROOT -> 0 -> 0:0; ROOT -> 1 -> 1:0; ROOT -> 2 -> 2:0; ROOT -> 3 -> 3:0

    def custom_next_obs_join(obs: list[ToolResponseMessage | Message]) -> str:
        return join(
            type(o)(
                content=(o.content or "").split(" ", maxsplit=1)[0],
                **o.model_dump(exclude={"content"}),
            )
            for o in obs
        )

    merged_tree = tree.merge_identical_nodes(
        lambda state: state, next_observation_hash_fn=custom_next_obs_join
    )
    # Tree should now be ROOT -> 0/1 -> 0:0/1:0; ROOT -> 2/3 -> 2:0

    assert {
        step_id: tree.get_weight(step_id) for step_id in nx.topological_sort(tree.tree)
    } == {
        "dummy": 1.0,  # Root
        "dummy:0": 1.0,
        "dummy:1": 1.0,
        "dummy:2": 1.0,
        "dummy:3": 1.0,
        "dummy:0:0": 1.0,
        "dummy:1:0": 1.0,
        "dummy:2:0": 1.0,
        "dummy:3:0": 1.0,
    }
    assert {
        step_id: merged_tree.get_weight(step_id)
        for step_id in nx.topological_sort(merged_tree.tree)
    } == {
        "dummy": 1.0,  # Root
        "dummy:0": 2.0,  # Middle 0/1
        "dummy:2": 2.0,  # Middle 2/3
        "dummy:0:0": 2.0,  # Bottom 0 (from 0/1)
        "dummy:2:0": 2.0,  # Bottom 0 (from 2/3)
    }
