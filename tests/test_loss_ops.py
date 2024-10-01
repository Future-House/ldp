import numpy as np
import pytest

from ldp.graph.loss_ops import MSELossOp
from ldp.graph.op_utils import compute_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("input_size", [4, 10])
async def test_embedding_op(input_size) -> None:
    op = MSELossOp()
    rng = np.random.default_rng(12345)
    async with compute_graph():
        op_result = await op(
            prediction=rng.random(input_size),
            target=rng.random(input_size),
        )
    assert isinstance(op_result.value, float)
    op_result.compute_grads()
    grads = op.get_input_grads(op_result.call_id)
    assert grads[0] == []
    assert grads[1].keys() == {"prediction", "target"}
    assert grads[1].get("target") is None
    pred = grads[1].get("prediction")
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (input_size,)
