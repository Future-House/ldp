import numpy as np
import pytest
import torch

from ldp.graph import MSELossOp, compute_graph


@pytest.mark.asyncio
@pytest.mark.parametrize("input_size", [4, 10])
@pytest.mark.parametrize("dtype", ["numpy", "torch"])
async def test_embedding_op(input_size, dtype) -> None:
    op = MSELossOp()

    # Generate data based on dtype
    if dtype == "numpy":
        rng = np.random.default_rng(12345)
        prediction = rng.random(input_size)
        target = rng.random(input_size)
    else:
        prediction = torch.rand(input_size)
        target = torch.rand(input_size)
    async with compute_graph():
        op_result = await op(
            prediction=prediction,
            target=target,
        )

    # Validate the output and grads
    op_result.compute_grads()
    grads = op.get_input_grads(op_result.call_id)
    assert grads[0] == []
    assert grads[1].keys() == {"prediction", "target"}
    assert grads[1].get("target") is None
    pred = grads[1].get("prediction")
    if dtype == "numpy":
        assert isinstance(pred, np.ndarray)
        assert isinstance(op_result.value, float)
    else:
        assert isinstance(pred, torch.Tensor)
        assert isinstance(op_result.value, torch.Tensor)
    assert pred.shape == (input_size,)
