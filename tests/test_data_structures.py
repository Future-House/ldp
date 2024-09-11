import networkx as nx
import pytest

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


def test_tree_node_merging():
    root_id = "dummy"
    tree = TransitionTree(root_id=root_id)

    kw = {
        "next_agent_state": None,
        "observation": Transition.NO_OBSERVATION,
        "next_observation": Transition.NO_OBSERVATION,
        "action": None,
    }

    # Construct a tree with two identical nodes
    tree.add_transition(
        f"{root_id}:0", Transition(timestep=0, reward=0.0, agent_state=0, **kw)
    )
    tree.add_transition(
        f"{root_id}:1", Transition(timestep=0, reward=0.0, agent_state=0, **kw)
    )

    # Now add a child to each of the nodes
    for parent in ("0", "1"):
        tree.add_transition(
            f"{root_id}:{parent}:0",
            Transition(timestep=1, reward=0.0, agent_state=1, **kw),
        )

    # Tree at this stage is ROOT -> 0 -> 0:0; ROOT -> 1 -> 1:0

    merged_tree = tree.merge_identical_nodes(lambda state: state)
    # Tree should now be ROOT -> 0/1 -> 0:0/1:0

    assert len(tree.tree.nodes) == 5
    assert len(merged_tree.tree.nodes) == 3

    node_weights = [
        merged_tree.get_weight(step_id)
        for step_id in nx.topological_sort(merged_tree.tree)
    ]
    assert node_weights == [1, 2, 2]  # 1 for the root, 2 for the others
