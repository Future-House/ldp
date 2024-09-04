import itertools
from collections.abc import Sequence
from typing import Any

import networkx as nx
from aviary.message import Message, join
from aviary.tools import Tool, ToolRequestMessage

from ldp.graph.ops import GradOutType, OpResult


def discounted_returns(
    rewards: list[float], terminated: list[bool], discount: float = 1.0
) -> list[float]:
    r"""
    Calculate the discounted returns for a list of rewards, considering termination flags and a discount factor.

    The discounted return represents the future discounted rewards from each time step onwards, taking into account
    whether an episode has terminated at each step.

    The discounted return \( G_t \) is given by:

    .. math::
        G_t = \sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k}

        where:
        - \( G_t \) is the discounted return starting from time step \( t \).
        - \( \gamma \) is the discount factor.
        - \( R_{t+k} \) is the reward received at time step \( t+k \).


    Args:
        rewards: A list of rewards at each time step.
        terminated: A list of boolean flags indicating whether the episode terminated at each time step.
        discount: Discount factor to apply to future rewards. Defaults to 1.0 which means no discounting is applied.

    Returns:
        A list of discounted returns (rewards to go), with each element representing the
            total discounted reward from that step onwards.

    Example:
        >>> rewards = [1.0, 2.0, 3.0]
        >>> terminated = [False, False, True]
        >>> discounted_returns(rewards, terminated, discount=0.9)
        [5.23, 4.7, 3.0]
    """
    returns = []
    r = 0.0
    for reward, term in zip(reversed(rewards), reversed(terminated), strict=False):
        # 1 - term is 0 if the episode has terminated
        r = reward + discount * r * (1 - term)
        returns.append(r)
    returns.reverse()
    return returns


def to_network(  # noqa: C901
    op_result: OpResult,
    max_label_height: int | None = None,
    max_label_width: int | None = None,
    G: "nx.MultiDiGraph | None" = None,
) -> "nx.MultiDiGraph":
    """
    Populate a NetworkX graph from the input op result's computation graph.

    How to export Graphviz .dot file: nx.drawing.nx_pydot.write_dot(G, "file.dot")
    How to render with Graphviz: nx.drawing.nx_pydot.to_pydot(G).write_png("file.png")
    Online Graphviz renderer: https://dreampuf.github.io/GraphvizOnline/

    Args:
        op_result: Starting op result to recurse parent op calls and results.
        max_label_height: Optional max label height (lines).
        max_label_width: Optional max label width (chars).
        G: Optional graph to add nodes/edges to. Allows this to be a recursive function.

    Returns:
        Populated a NetworkX multi-edge directed graph.
    """

    def gvizify(x: Any) -> str:
        """Stringify and then escape colons for Graphviz labels."""
        if isinstance(x, OpResult):
            x = x.value
        if isinstance(x, Sequence):
            if isinstance(x[0], Message):
                x = join(x)
            elif isinstance(x[0], Tool):
                x = "\n".join(f"Tool {t.info.name}" for t in x)
        elif isinstance(x, ToolRequestMessage):
            # reformatting tool calls to make them easier to read
            x = str(x).split(" for tool calls: ")
            x = "\n".join(x).replace("; ", "\n")
        result = (
            "\n".join(
                # Replace double quotes since they can interfere with colon escapes
                # Strip here to avoid trailing spaces in the labels
                x_line[:max_label_width].replace('"', "'").strip()
                for i, x_line in enumerate(str(x).split("\n"))
                if not max_label_height or i < max_label_height
            )
        ).strip()  # Remove trailing newlines
        return result if ":" not in result else f'"{result}"'  # Escape colons

    call_id = op_result.call_id
    assert (
        call_id is not None
    ), "to_network currently assumes a compute graph is available"
    ctx = op_result.ctx

    op_result_str = gvizify(op_result)
    op_result_node = gvizify(f"{op_result_str}\n{call_id.fwd_id}")
    if G is None:
        # TODO: figure out a way to use OpResult.get_compute_graph(), which builds
        # a nx.DiGraph.
        G = nx.MultiDiGraph()

    op_call_str = gvizify(f"{ctx.op_name}:{call_id.fwd_id}")
    if op_call_str in G:
        # We have already visited this node - can skip.
        return G

    G.add_node(op_result_node, style="dotted", label=op_result_str)
    G.add_edge(op_call_str, op_result_node)

    if (
        result_grad := ctx.get(key="grad_output", call_id=call_id, default=None)
    ) is not None:
        G.add_edge(
            op_result_node,
            op_call_str,
            label=gvizify(result_grad),
            style="dotted",
        )

    input_args, input_kwargs = op_result.inputs
    grads = ctx.get(key="grad_input", call_id=call_id, default=None)
    if grads is None:
        arg_grads: list[GradOutType | None] = [None] * len(input_args)
        kwarg_grads: dict[str, GradOutType | None] = dict.fromkeys(input_kwargs)
    else:
        arg_grads, kwarg_grads = grads

    args_and_grads = itertools.chain(
        zip(input_args, arg_grads, strict=True),
        ((arg, kwarg_grads[key]) for key, arg in input_kwargs.items()),
    )

    for arg, grad in args_and_grads:
        arg_str = gvizify(arg)

        if not isinstance(arg, OpResult):
            G.add_node(arg_str, style="dotted")

        else:
            arg_str = gvizify(f"{arg_str}\n{arg.call_id.fwd_id}")
            G = to_network(
                arg,
                max_label_height=max_label_height,
                max_label_width=max_label_width,
                G=G,
            )

        G.add_edge(arg_str, op_call_str)
        if grad is not None:
            G.add_edge(op_call_str, arg_str, label=gvizify(grad), style="dotted")

    return G
