import inspect
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

try:
    import torch
    from torch import nn
except ImportError:
    raise ImportError(
        "ldp.graph.torch_ops requires PyTorch as a dependency. "
        "Please run `pip install ldp[nn]`."
    ) from None

from .async_torch import async_protect_torch_call
from .op_utils import CallID, get_call_id, get_training_mode
from .ops import GradInType, Op, OpCtx, ResultOrValue


class TorchOp(Op[torch.Tensor]):
    """An operation that wraps a PyTorch module."""

    CTX_TENSOR_INPUT_KEY: ClassVar[str] = "tensor_input"

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        # override forward args with the signature of the function
        fwd_sig = inspect.signature(self.module.forward)
        self._fwd_args = list(fwd_sig.parameters.values())

    def __str__(self) -> str:
        return f"{type(self).__name__} {type(self.module).__name__} ({id(self)})"

    async def forward(self, *args, **kwargs: Any) -> torch.Tensor:
        tensor_args = [
            (
                arg
                if isinstance(arg, torch.Tensor)
                else torch.tensor(arg, requires_grad=True)
            )
            for arg in args
        ]
        tensor_kwargs = {
            k: v if isinstance(v, torch.Tensor) else torch.tensor(v, requires_grad=True)
            for k, v in kwargs.items()
        }

        is_training = get_training_mode()

        if is_training:
            store_tensor_inputs(
                ctx=self.ctx,
                key=self.CTX_TENSOR_INPUT_KEY,
                tensor_args=tensor_args,
                tensor_kwargs=tensor_kwargs,
                fwd_args=self._fwd_args,
            )

        return await async_protect_torch_call(self.module, no_grad=not is_training)(
            *tensor_args, **tensor_kwargs
        )

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args: list[ResultOrValue],
        input_kwargs: dict[str, ResultOrValue],
        grad_output: float | torch.Tensor,
        call_id: CallID,
    ) -> GradInType:
        tensor_args, tensor_kwargs = ctx.get(call_id, cls.CTX_TENSOR_INPUT_KEY)
        n_pos_args = len(tensor_args)
        output = ctx.get(call_id, "output").value

        if not isinstance(grad_output, torch.Tensor):
            grad_output = torch.tensor(grad_output, dtype=output.dtype)
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
            [*tensor_args, *tensor_kwargs.values()],
            grad_outputs=grad_output,
            allow_unused=True,
            retain_graph=True,
        )

        grad_args = [grad.detach().cpu().float() for grad in gradients[:n_pos_args]]
        grad_kwargs = {
            k: grad.detach().cpu().float()
            for k, grad in zip(
                tensor_kwargs.keys(), gradients[n_pos_args:], strict=True
            )
        }

        return grad_args, grad_kwargs


def store_tensor_inputs(
    ctx: OpCtx,
    key: str,
    tensor_args: Sequence[torch.Tensor],
    tensor_kwargs: Mapping[str, torch.Tensor],
    fwd_args: Sequence[inspect.Parameter],
    detach: bool = False,
) -> None:
    call_id = get_call_id()
    # Save tensor inputs for backward pass. Do not clobber "input", since
    # that is needed for compute graph. Map positional args to kwargs
    # Copying so we don't modify tensor_kwargs in-place
    ctx_kwargs = tensor_kwargs.copy()  # type: ignore[attr-defined]

    # See Op.__call__ for some notes on what this is doing.
    for i_arg, (arg, param) in enumerate(
        # strict=False b/c not all params in _fwd_args will be in args (i.e. defaults and **kwargs)
        zip(tensor_args, fwd_args, strict=False)
    ):
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            ctx_args = list(tensor_args[i_arg:])
            break

        # Normal positional arg
        ctx_kwargs[param.name] = arg
    else:
        ctx_args = []  # if we got here, there were no *args

    if detach:
        # Detach the tensors from the compute graph and move to CPU
        ctx_args = [arg.detach().cpu() for arg in ctx_args]
        ctx_kwargs = {k: v.detach().cpu() for k, v in ctx_kwargs.items()}

    ctx.update(call_id, key, (ctx_args, ctx_kwargs))
