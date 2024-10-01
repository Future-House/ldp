import pytest
import numpy as np

from ldp.graph.loss_ops import MSELossOp
from ldp.graph.op_utils import compute_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("input_size", [4, 10])
async def test_embedding_op(input_size) -> None:
    op = MSELossOp()
    async with compute_graph():
        op_result = await op(
            np.random.rand(input_size),
            np.random.rand(input_size),
        )
    assert isinstance(op_result.value, float)
    op_result.compute_grads()
    grads = op.get_input_grads(op_result.call_id)
    assert grads[0] == []
    assert grads[1].keys() == {"prediction", "target"}
    assert grads[1]["target"] is None
    assert grads[1]["prediction"].shape == (input_size,)
