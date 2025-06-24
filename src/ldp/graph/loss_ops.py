"""This module contains loss Op implementations."""

from typing import TYPE_CHECKING

from .op_utils import CallID
from .ops import GradInType, Op, OpCtx

if TYPE_CHECKING:
    import numpy.typing as npt
    import torch
    import tree


class MSELossOp(Op):
    async def forward(
        self,
        prediction: "npt.NDArray | torch.Tensor",
        target: "npt.NDArray | torch.Tensor",
    ) -> "float | torch.Tensor":
        return ((prediction - target) ** 2).mean()

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args,
        input_kwargs,
        grad_output: "tree.Structure",
        call_id: CallID,
    ) -> GradInType:
        prediction = input_kwargs["prediction"]
        target = input_kwargs["target"]
        grad = 2 * (prediction - target)
        return [], {"prediction": grad, "target": None}
