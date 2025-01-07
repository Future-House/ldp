import asyncio
import collections
import itertools
import random
from collections.abc import Awaitable, Callable, Hashable, Iterable, Sequence
from typing import Any, Literal, TypeVar, cast

import networkx as nx
import numpy as np
from aviary.core import Message, Tool, ToolRequestMessage, is_coroutine_callable, join

from ldp.graph import OpResult
from ldp.graph.ops import GradOutType


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
    assert call_id is not None, (
        "to_network currently assumes a compute graph is available"
    )
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


TData = TypeVar("TData")
TGroupKey = TypeVar("TGroupKey", bound=Hashable)
TAnswer = TypeVar("TAnswer")
NO_IDEAL_ANSWER_FN: Literal["NO_IDEAL_ANSWER_FN"] = "NO_IDEAL_ANSWER_FN"  # Sentinel


async def evaluate_consensus(
    data: Iterable[TData],
    grouping_fn: Callable[[TData], TGroupKey],
    extract_answer_fn: Callable[[TData], TAnswer | Awaitable[TAnswer]],
    ideal_answer_fn: (
        Callable[[TData], TAnswer] | Literal["NO_IDEAL_ANSWER_FN"]
    ) = NO_IDEAL_ANSWER_FN,
    num_samples: int = 1,
    seed: int | None = None,
) -> tuple[dict[TGroupKey, list[tuple[TAnswer, int]]], float]:
    """
    Create consensus groups and evaluate the consensus accuracy for each one.

    Args:
        data: Data to evaluate consensus upon, length is called N.
        grouping_fn: Function to extract the group key from a datum.
        extract_answer_fn: Function to extract the actual answer from a datum. It can
            be async so this can be done using a LLM call.
        ideal_answer_fn: Optional function to extract the ideal answer from a datum to
            compute accuracy with, or pass NO_IDEAL_ANSWER to skip this calculation.
        num_samples: Number of samples to choose from the N total.
        seed: Optional seed for sampling.

    Returns:
        Two-tuple of consensus list generated by collections.Counter.most_common and
            the proportion of groups for which the consensus matches the ideal.
    """
    groups = collections.defaultdict(list)
    for x in data:
        groups[grouping_fn(x)].append(x)

    ideal_count = 0
    grouped_consensus: dict[TGroupKey, list[tuple[TAnswer, int]]] = {}
    rand = random.Random(seed) if seed is not None else random
    for group_key, group in groups.items():
        if len(group) < num_samples:  # Too few items, sample with replacement
            sampled = [rand.choice(group) for _ in range(num_samples)]
        else:  # Sample without replacement
            sampled = rand.sample(group, num_samples)

        # Get answers for the sampled data
        if is_coroutine_callable(extract_answer_fn):
            extract_answer_fn = cast(
                Callable[[TData], Awaitable[TAnswer]], extract_answer_fn
            )
            answers = await asyncio.gather(*(extract_answer_fn(x) for x in sampled))
        else:
            extract_answer_fn = cast(Callable[[TData], TAnswer], extract_answer_fn)
            answers = [extract_answer_fn(x) for x in sampled]

        # Compute consensus: mode of the sampled answers
        grouped_consensus[group_key] = collections.Counter(answers).most_common()
        # NOTE: If there are multiple modes, just use the first one
        consensus: TAnswer = grouped_consensus[group_key][0][0]
        if ideal_answer_fn != NO_IDEAL_ANSWER_FN:
            # Assume all items in the group have the same ideal answer
            ideal_count += consensus == ideal_answer_fn(group[0])

    return grouped_consensus, ideal_count / len(groups) if groups else 0.0


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute an unbiased estimation for 'pass @ k'.

    Source: https://doi.org/10.48550/arXiv.2107.03374's figure 3.

    If there's multiple tasks, an aggregation used is averaging pass @ k across tasks,
    per https://doi.org/10.48550/arXiv.2407.21787's equation 1.

    Args:
        n: Total number of samples.
        c: Number of correct (pass a verifier) samples.
        k: k term (number of attempts) used in pass @ k.

    Returns:
        Unbiased estimation for pass @ k, the probability of getting at least one
            successful outcome in k attempts.
    """
    if n - c < k:
        return 1.0

    # Using logarithms for a more efficient and numerically stable product calculation
    log_product = sum(np.log(1.0 - k / i) for i in range(n - c + 1, n + 1))
    return 1.0 - np.exp(log_product)
