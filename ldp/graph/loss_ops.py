"""This module contains loss Op implementations."""

import numpy as np
import tree

from ldp.graph.op_utils import CallID
from ldp.graph.ops import GradInType, Op, OpCtx


class MSELossOp(Op):
    async def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return np.mean((prediction - target) ** 2)

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args,
        input_kwargs,
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        prediction = input_kwargs["prediction"]
        target = input_kwargs["target"]
        grad = 2 * (prediction - target)
        return [], {"prediction": grad, "target": None}
