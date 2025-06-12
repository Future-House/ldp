import asyncio
import random
import time
from contextlib import nullcontext
from uuid import UUID

import numpy as np
import pytest
import torch
from torch import nn

from ldp.graph import ConfigOp, FxnOp, PromptOp, compute_graph, set_training_mode
from ldp.graph.async_torch import (
    AsyncTorchModule,
    _get_autocast_context,  # noqa: PLC2701
    async_protect_torch_call,
)
from ldp.graph.gradient_estimators import straight_through_estimator as ste
from ldp.graph.torch_ops import TorchOp


@pytest.fixture(name="run_id")
def fixture_run_id() -> UUID:
    return UUID("12345678-1234-5678-1234-567812345678")


class AddModule(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


class MulModule(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * y


class SinModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


TEST_CASES = [
    pytest.param(
        AddModule(),
        (
            torch.tensor(2.0, requires_grad=True),
            torch.tensor(3.0, requires_grad=True),
        ),
        5.0,
        {"x": 1.0, "y": 1.0},
        id="addition",
    ),
    pytest.param(
        MulModule(),
        (
            torch.tensor(2.0, requires_grad=True),
            torch.tensor(3.0, requires_grad=True),
        ),
        6.0,
        {"x": 3.0, "y": 2.0},
        id="multiplication",
    ),
    pytest.param(
        SinModule(),
        (torch.tensor(np.pi / 4, requires_grad=True),),
        np.sin(np.pi / 4),
        {"x": np.cos(np.pi / 4)},
        id="non_linear",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "inputs", "expected_output", "expected_grads"), TEST_CASES
)
@pytest.mark.parametrize("training_mode", [True, False])
async def test_torch_op(
    module, inputs, expected_output, expected_grads, training_mode: bool
):
    set_training_mode(training_mode)
    op = TorchOp(module)
    async with compute_graph():
        result = await op(*inputs)
    assert result.value.requires_grad == training_mode

    if isinstance(expected_output, torch.Tensor):
        assert torch.allclose(result.value, expected_output)
    else:
        assert result.value.detach() == pytest.approx(expected_output)

    raises_context = (
        nullcontext()
        if training_mode
        else pytest.raises(KeyError, match="key='tensor_input' not found")
    )

    with raises_context:
        grad_output = torch.ones_like(result.value)
        arg_grads, kwarg_grads = TorchOp.backward(  # noqa: RUF059
            op.ctx,
            [],
            {},  # NOTE: compute_grads() would fill this, but TorchOp.backwards() should ignore it
            grad_output=grad_output,
            call_id=result.call_id,
        )

        assert set(kwarg_grads.keys()) == set(expected_grads.keys())

        for param, expected in expected_grads.items():
            computed = kwarg_grads[param]
            if isinstance(expected, torch.Tensor):
                assert torch.allclose(torch.tensor(computed), expected)
            else:
                assert computed == pytest.approx(expected)


@pytest.mark.asyncio
async def test_with_kwargs():
    class ScaledAddModule(nn.Module):
        def forward(self, x, y, scale=1.0):
            return (x + y) * scale

    op = TorchOp(ScaledAddModule())
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    scale = torch.tensor(2.0, requires_grad=True)

    async with compute_graph():
        result = await op(x, y, scale=scale)
    assert isinstance(result.value.detach(), torch.Tensor)
    assert result.value.detach() == pytest.approx(10.0)

    grad_output = torch.tensor(1.0)
    arg_grads, kwarg_grads = TorchOp.backward(
        op.ctx,
        [],
        {"x": x, "y": y, "scale": scale},
        grad_output,
        result.call_id,
    )

    assert not arg_grads
    assert len(kwarg_grads) == 3
    assert kwarg_grads["x"] == pytest.approx(2.0)  # d(result)/dx = scale = 2.0
    assert kwarg_grads["y"] == pytest.approx(2.0)  # d(result)/dy = scale = 2.0
    assert kwarg_grads["scale"] == pytest.approx(
        5.0
    )  # d(result)/d(scale) = x + y = 5.0


@pytest.mark.asyncio
async def test_torch_op_composition() -> None:
    # Define our ops
    config_op = ConfigOp(config={"scale": 2.0})
    # Don't use FURB118 here because we assert on kwargs below
    fxn_op = FxnOp(lambda x: x["scale"])  # noqa: FURB118

    class ScaleSum(nn.Module):
        def forward(self, x, y):
            return y * torch.sum(x)

    torch_op = TorchOp(ScaleSum())
    prompt_op = PromptOp("The result is: {result}")

    # Forward pass
    async with compute_graph():
        config = await config_op()
        fxn_result = await fxn_op(config)
        x = [1.0, 2.0, 3.0]
        torch_result = await torch_op(x, fxn_result)
        prompt_result = await prompt_op(result=torch_result)

    # Check forward pass results
    assert isinstance(torch_result.value, torch.Tensor)
    assert torch_result.value.detach() == pytest.approx(
        12.0
    )  # y * sum(x) = 2.0 * 6.0 = 12.0
    assert fxn_result.value == pytest.approx(2.0)
    assert prompt_result.value == "The result is: 12.0"

    # Backward pass
    loss_grad = -1.0  # Arbitrary d(loss)/d(result)
    prompt_result.compute_grads(
        loss_grad,
        backward_fns={
            PromptOp: ste,
            FxnOp: ste,
        },
    )
    fxn_grad = fxn_op.get_input_grads(fxn_result.call_id)
    torch_grad_args, torch_grad_kwargs = torch_op.get_input_grads(torch_result.call_id)

    # Check backward pass results

    # d(loss)/d(scale) = d(loss)/d(result) * d(result)/d(scale) = -1.0 * 6.0 = -6.0
    assert fxn_grad == ([], {"x": {"scale": -6.0}})

    assert not torch_grad_args
    # d(result)/dx = y = 2.0
    assert torch.allclose(torch_grad_kwargs["x"], torch.tensor([-2.0, -2.0, -2.0]))  # type: ignore[arg-type]
    # d(result)/dy = sum(x) = 6.0
    assert torch.allclose(torch_grad_kwargs["y"], torch.tensor(-6.0))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_torch_concurrency():
    assert torch.is_grad_enabled()

    def check_no_grad(grad_expected: bool):
        # This takes the place of a pytorch module - we want to
        # see if torch.is_grad_enabled() is set correctly
        assert torch.is_grad_enabled() == grad_expected

    async def change_grads_a_lot():
        for _ in range(10):
            no_grad = random.choice([True, False])
            await asyncio.sleep(0)
            await async_protect_torch_call(check_no_grad, no_grad=no_grad)(  # type: ignore[arg-type]
                not no_grad
            )
            assert torch.is_grad_enabled()  # Make sure grad state is reset properly

            # The below is an example of doing unsafe concurrent operations in torch.
            # if no_grad:
            #     with torch.no_grad():
            #         asyncio.sleep(0.0)
            #         assert not torch.is_grad_enabled()
            # else:
            #     asyncio.sleep(0.0)
            #     check_no_grad(not no_grad)

    await asyncio.gather(*[change_grads_a_lot() for _ in range(10)])


@pytest.mark.asyncio
async def test_torch_autocast():
    model = nn.Linear(4, 4)

    def call_model():
        result = model(torch.rand(4))
        assert result.dtype == torch.bfloat16

    await async_protect_torch_call(
        call_model,  # type: ignore[arg-type]
        autocast_dtype=torch.bfloat16,
        autocast_device_type=torch.device("cpu").type,
    )()


class TestAsyncTorchModule:
    @pytest.mark.asyncio
    async def test_batching(self):
        model = torch.nn.Linear(2, 2)
        batch_size = 4
        max_wait = 1.0  # long timeout so we can measure it
        async_module = AsyncTorchModule(model, batch_size, max_wait)

        # First test that we can do a single call
        start = time.time()
        result = await async_module(input=torch.rand(2))
        assert max_wait < time.time() - start < max_wait * 2
        assert result.shape == (2,)

        # Now check that we can do a batched call
        start = time.time()
        results = await asyncio.gather(*[
            async_module(input=torch.rand(2)) for _ in range(batch_size)
        ])
        # should not have waited for timeout since we hit batch_size
        assert time.time() - start < max_wait
        assert len(results) == batch_size
        assert all(r.shape == (2,) for r in results)

        # Finally check an ueven number of calls
        results = await asyncio.gather(*[
            async_module(input=torch.rand(2)) for _ in range(2 * batch_size + 1)
        ])
        assert len(results) == 2 * batch_size + 1
        assert all(r.shape == (2,) for r in results)

    @pytest.mark.parametrize("batch_size", [4, 10])
    @pytest.mark.parametrize("max_wait", [0.01, 0.1, 1.0])
    @pytest.mark.asyncio
    async def test_ordering(self, batch_size: int, max_wait: float):
        # Make sure that we are actually getting the right results back, and that results
        # aren't jumbled

        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.data.fill_(1.0)
        async_module = AsyncTorchModule(model, batch_size, max_wait)

        # Run 2 loops to make sure subsequent calls don't interfere
        for _ in range(2):
            inputs = [torch.rand(2) for _ in range(32)]
            outputs = await asyncio.gather(*[async_module(input=x) for x in inputs])

            for inp, out in zip(inputs, outputs, strict=True):
                assert torch.allclose(out, inp.sum())

            # Make sure we didn't leave any dangling work in the buffer
            assert not async_module._work_buffer

    # For tiny models, we expect a slowdown, but less than 3x.
    # As the model gets bigger, we get a slight speedup.
    # The real benefit is when the model fills up a GPU, but we don't want that in tests.
    @pytest.mark.skip(
        reason="The speedups are very hardware-dependent, so do not automatically run.",
    )
    @pytest.mark.parametrize(("dim", "expected_speedup"), [(2, 3), (4096, 0.8)])
    @pytest.mark.asyncio
    async def test_performance(self, dim: int, expected_speedup: float):
        batch_size = 10
        model = torch.nn.Linear(dim, dim)
        async_module = AsyncTorchModule(model, batch_size, 0.001)
        sync_batch = torch.rand(batch_size, dim)
        start = time.time()
        _ = model(sync_batch)
        sync_time = time.time() - start

        async_batch = list(sync_batch)
        start = time.time()
        _ = await asyncio.gather(*[async_module(input=x) for x in async_batch])
        async_time = time.time() - start
        print(
            f"Dimension: {dim}; torch.nn.Module: {sync_time:.6f}s; AsyncTorchModule:"
            f" {async_time:.6f}s"
        )

        assert async_time < sync_time * expected_speedup


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "mps"])
def test_autocast(device_type: str):
    with _get_autocast_context(torch.bfloat16, device_type) as ctx:
        if device_type in {"mps", "cpu"}:
            # MPS and CPU devices don't support autocast
            assert not torch.is_autocast_enabled()
        else:
            # CUDA will only work if we've got a GPU
            assert torch.is_autocast_enabled() == torch.cuda.is_available()
