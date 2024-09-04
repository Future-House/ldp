"""This module defines various gradient estimators that can be patched in during backward passes."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import tree

from ldp.graph.op_utils import CallID
from ldp.graph.ops import GradInType, OpCtx, OpResult, ResultOrValue

logger = logging.getLogger(__name__)


def assign_constant_grads(
    input_args: list[ResultOrValue],
    input_kwargs: dict[str, ResultOrValue],
    value: Any,
    descend: bool = True,
):
    if not descend:
        return [value] * len(input_args), dict.fromkeys(input_kwargs, value)

    # descend into nested objects
    arg_grads = [
        tree.map_structure(lambda _: value, OpResult.unwrap_value(arg))
        for arg in input_args
    ]
    kwarg_grads = {
        k: tree.map_structure(lambda _: value, OpResult.unwrap_value(v))
        for k, v in input_kwargs.items()
    }
    return arg_grads, kwarg_grads


def straight_through_estimator(
    ctx: OpCtx,  # noqa: ARG001
    input_args: list[ResultOrValue],
    input_kwargs: dict[str, ResultOrValue],
    grad_output: tree.Structure,
    call_id: CallID,  # noqa: ARG001
    descend: bool = True,
) -> GradInType:
    return assign_constant_grads(input_args, input_kwargs, grad_output, descend=descend)


def stop_grad(
    ctx: OpCtx,  # noqa: ARG001
    input_args: list[ResultOrValue],
    input_kwargs: dict[str, ResultOrValue],
    grad_output: tree.Structure,  # noqa: ARG001
    call_id: CallID,  # noqa: ARG001
) -> GradInType:
    # don't descend - want gradients to stop at the OpResult level
    return assign_constant_grads(input_args, input_kwargs, None, descend=False)


def zero_estimator(
    ctx: OpCtx,  # noqa: ARG001
    input_args: list[ResultOrValue],
    input_kwargs: dict[str, ResultOrValue],
    grad_output: tree.Structure,  # noqa: ARG001
    call_id: CallID,  # noqa: ARG001
) -> GradInType:
    """Sets the gradient of all inputs to zero.

    Note that this is not the same as truncating the compute graph (stop_grad),
    since upstream nodes can still optimize their logprobs. The zero estimator
    the unbiased choice if we have no information about the gradient.
    """
    return assign_constant_grads(input_args, input_kwargs, 0.0)


def llm_straight_through_estimator(
    ctx: OpCtx,  # noqa: ARG001
    input_args: list[ResultOrValue],
    input_kwargs: dict[str, ResultOrValue],
    grad_output: tree.Structure,
    call_id: CallID,  # noqa: ARG001
) -> GradInType:
    """Straight-through for an LLM: descend into the config, but not msgs/tools/tool_calls.

    See LLMCallOp.backward() for more details on this choice.
    Don't bother checking that input_args/input_kwargs have the right structure,
    since compute_grads() will raise if not.
    """
    config_grad = tree.map_structure(
        lambda _: grad_output, OpResult.unwrap_value(input_kwargs["config"])
    )
    grad_args = [grad_output] * len(input_args)
    grad_kwargs = {"config": config_grad}
    if "msgs" in input_kwargs:
        grad_kwargs["msgs"] = grad_output
    if "tools" in input_kwargs:
        grad_kwargs["tools"] = grad_output

    return grad_args, grad_kwargs


def assign_default_grads(
    input_grads: GradInType,
    input_args: list[ResultOrValue],
    input_kwargs: dict[str, ResultOrValue],
    default_grad_val: float = 0.0,
) -> GradInType:
    """Sets a default value of default_grad_val for every element in input_grads.

    Example:
    - input_kwargs = {"a": {"b": 1, "c": 2}},
    - input_grad_kwargs = {"a": {"b": 0.1}}
    Output: input_grads[1] = {"a": {"b": 0.1, "c": default_grad_val}}

    Returns:
        GradInType: A tuple containing the updated input_grad_args and
            input_grad_kwargs with default values assigned where necessary.
    """

    def get_nested_value(data: tree.Structure, path: list) -> Any:
        """Traverse given path over data and return the value at the end of the path."""
        try:
            current_value = data
            for key in path:
                current_value = current_value[key]
        except (KeyError, IndexError):
            return None  # If path not found, return None (than default_grad_val will be assigned)
        else:
            return current_value

    def assign_default_gradients(
        input_grads: tree.Structure, path: list, _value: Any
    ) -> Any:
        """Assign default_grad_val where grads are missing."""
        return get_nested_value(input_grads, path) or default_grad_val

    input_args_kwargs = (input_args, input_kwargs)
    input_grads = tree.map_structure_with_path(
        partial(assign_default_gradients, input_grads),
        input_args_kwargs,
    )

    tree.assert_same_structure(input_grads, input_args_kwargs)
    return input_grads
