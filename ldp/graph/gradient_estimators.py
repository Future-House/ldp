"""This module defines various gradient estimators that can be patched in during backward passes."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import tree

from .op_utils import CallID
from .ops import GradInType, OpCtx, OpResult, ResultOrValue

try:
    import torch

    from .torch_ops import TorchOp

except ImportError:
    torch = None  # type: ignore[assignment]

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
    for arg in ("msgs", "tools", "tool_choice"):
        if arg in input_kwargs:
            grad_kwargs[arg] = grad_output

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


class TorchParamBackwardEstimator:
    """
    Gradient estimator for `TorchOp` internal parameters.

    This estimator computes gradients with respect to the internal parameters of a
    `torch.nn.Module` by calling the `backward` method of the estimator instead of the default
    `backward` method of `TorchOp`. Computed gradients are stored in the context of the operation
    under the key `"grad_params"`.

    Examples:
        >>> torch_module = torch.nn.Sequential(
        ...     torch.nn.Linear(4, 4),
        ...     torch.nn.Linear(4, 1),
        ... )
        >>> torch_op = TorchOp(torch_module)
        >>> estimator = TorchParamBackwardEstimator(torch_module)
        >>> result = await torch_op(torch.randn(4, requires_grad=True))
        >>> result.compute_grads(backward_fns={"TorchOp": estimator.backward})

    Note:
        This estimator is only compatible with `TorchOp` operations.
    """

    def __init__(self, module: torch.nn.Module):
        if torch is None:
            raise RuntimeError(
                f"PyTorch library not found. Unable to use {type(self).__name__} class."
                " To install PyTorch dependencies, please run `pip install ldp[nn]`."
            )
        self.params = dict(module.named_parameters())

    def backward(
        self,
        ctx: OpCtx,
        input_args: list[ResultOrValue],
        input_kwargs: dict[str, ResultOrValue],
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        tensor_args, tensor_kwargs = ctx.get(call_id, TorchOp.CTX_TENSOR_INPUT_KEY)
        n_pos_args = len(tensor_args)
        n_pos_kwargs = len(tensor_kwargs)
        output = ctx.get(call_id, "output").value

        if not isinstance(grad_output, torch.Tensor):
            grad_output = torch.tensor(
                grad_output,
                dtype=output.dtype,
            )
        grad_output = grad_output.to(output.device)

        while grad_output.ndim < output.ndim:
            # Assume we can broadcast, so expand dims
            # e.g. if output.shape = (2, 1, 1) and grad_output is a scalar
            # then we want to expand to (1, 1, 1) and then broadcast
            grad_output = grad_output.unsqueeze(-1)

        if output.shape != grad_output.shape:
            raise RuntimeError(
                f"Output shape {output.shape} does not match grad_output shape"
                f" {grad_output.shape}"
            )

        gradients = torch.autograd.grad(
            output,
            [*tensor_args, *tensor_kwargs.values(), *self.params.values()],
            grad_outputs=grad_output,
        )

        grad_args = [grad.detach().cpu().float() for grad in gradients[:n_pos_args]]
        grad_kwargs = {
            k: grad.detach().cpu().float()
            for k, grad in zip(
                tensor_kwargs.keys(), gradients[n_pos_args:n_pos_kwargs], strict=True
            )
        }
        grad_params = {
            name: grad.detach().cpu().float()
            for name, grad in zip(
                self.params.keys(), gradients[n_pos_kwargs:], strict=True
            )
        }

        ctx.update(call_id=call_id, key="grad_params", value=grad_params)

        return grad_args, grad_kwargs
