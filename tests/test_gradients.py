import asyncio
import copy
import math
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, cast

import numpy as np
import pytest
import torch
import tree

from ldp.graph import CallID, ConfigOp, FxnOp, Op, OpCtx, OpResult, compute_graph
from ldp.graph.gradient_estimators import (
    TorchParamBackwardEstimator,
    assign_constant_grads,
    assign_default_grads,
)
from ldp.graph.ops import GradInType, ResultOrValue
from ldp.graph.torch_ops import TorchOp


class PoissonSamplerOp(Op):
    @staticmethod
    def _probability(lam: float, k: int) -> float:
        if k < 0:
            # negative k can happen when taking a gradient
            return 0.0
        return np.exp(-lam) * lam**k / math.factorial(k)

    async def forward(self, lam: float) -> int:
        return np.random.poisson(max(0.01, lam))  # noqa: NPY002

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args,
        input_kwargs,
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        # This op has no internal parameters, so we just compute delta_{j,i} in section 4.
        lam = max(0.01, input_kwargs["lam"])
        k = ctx.get(call_id, "output").value

        p_k = cls._probability(lam, k)
        p_km1 = cls._probability(lam, k - 1)

        # dp(k)/dlam
        grad_lam_p = p_km1 - p_k

        # d[lnp(k)]/dlam
        grad_lam_lnp = grad_lam_p / p_k

        # define dk/dlam in expectation: dE[k]/dlam = dlam/dlam = 1
        grad_lam_k = 1.0

        # delta_{j,i}
        delta_lam = grad_lam_lnp + (grad_lam_k * cast("float", grad_output))

        return [], {"lam": delta_lam}


class FloatParamOp(Op):
    def __init__(self, init_param: float):
        self.param = init_param

    async def forward(self) -> float:
        return self.param

    @classmethod
    def backward(
        cls, ctx: OpCtx, input_args, input_kwargs, grad_output: Any, call_id: CallID
    ) -> GradInType:
        return [], {}


class SGDOptimizer:
    def __init__(self, op: FloatParamOp, lr: float = 0.01, lr_decay: float = 1.0):
        self.op = op
        self.lr = lr
        self.lr_decay = lr_decay

        self.accumulated_updates: list[float] = []

    def aggregate(self, samples: Iterable[tuple[OpResult, float]]):
        for result, reward in samples:
            assert result.call_id is not None
            call_ids = self.op.get_call_ids({result.call_id.run_id})

            # These are delta_{k,j} in equation 20
            grads = [
                cast("float", g)
                for g in (
                    self.op.ctx.get(call_id, "grad_output") for call_id in call_ids
                )
                if g is not None
            ]
            if not grads:
                # this op call was pruned from the backwards call graph
                continue

            # We assume FloatParamOp is deterministic, meaning grad_j lnp_j = 0
            # Furthermore, grad_j x_j = 1 (dlam/dlam), so Eq 20 reduces to R*sum_k(delta_{k,j})
            self.accumulated_updates.append(reward * sum(grads))

    def update(self):
        self.op.param += self.lr * cast("float", np.mean(self.accumulated_updates))
        self.accumulated_updates.clear()
        self.lr *= self.lr_decay


@pytest.mark.parametrize("init_lam", [0.1, 20.0, 10.0])
@pytest.mark.flaky(reruns=3, only_on=[AssertionError])
@pytest.mark.asyncio
async def test_poisson_sgd(init_lam: float):
    # This test optimizes the rate parameter of a Poisson distribution such that
    # the expected value of the distribution is 10.
    target = 10

    def reward_fn(k: int) -> float:
        return -np.abs(k - target)

    lam = FloatParamOp(init_lam)
    poisson = PoissonSamplerOp()
    opt = SGDOptimizer(lam, lr=1.0, lr_decay=0.9)

    @compute_graph()
    async def fwd() -> OpResult[int]:
        return await poisson(await lam())

    bsz = 4
    n_epochs = 20
    for _ in range(n_epochs):
        samples = await asyncio.gather(*[fwd() for _ in range(bsz)])

        training_batch: list[tuple[OpResult, float]] = []
        for k in samples:
            k.compute_grads()
            reward = reward_fn(k.value)
            training_batch.append((k, reward))

        opt.aggregate(training_batch)
        opt.update()

        print(lam.param, opt.lr, np.mean([k.value for k in samples]))

    # noisy learning, so just make sure we went in the right direction
    assert np.isclose(lam.param, target, atol=4.0)


def assign_constant_grads_alter_inputs(
    _ctx: OpCtx,
    input_args,
    input_kwargs,
    _grad_output: int,
    _call_id: CallID,
    grad_val: float = 0.0,
    input_func: Callable = lambda x, y: (x, y),
) -> GradInType:
    input_args, input_kwargs = copy.deepcopy(input_args), copy.deepcopy(input_kwargs)
    input_args, input_kwargs = input_func(input_args, input_kwargs)
    return assign_constant_grads(input_args, input_kwargs, grad_val)


@pytest.mark.asyncio
async def test_nested_dict_kwargs_grad_aggregation_success():
    """Tests gradient aggregation across two ops that return tree.Structure objects."""
    cfg = {"a": {"b": [1, 2, 3], "c": 4}}
    config_op = ConfigOp(cfg)

    op1 = FxnOp[int](lambda _input_dict: 2)
    op1.set_name("op1")
    op2 = FxnOp[int](lambda _input_dict: 3)
    op2.set_name("op2")
    agg_op = FxnOp[int](lambda *args: sum(args))  # noqa: FURB111

    backward1 = partial(assign_constant_grads_alter_inputs, grad_val=1.0)
    backward2 = partial(assign_constant_grads_alter_inputs, grad_val=2.0)

    @compute_graph()
    async def fwd() -> OpResult[int]:
        config = await config_op()
        a = await op1(config)
        b = await op2(config)
        return await agg_op(a, b)

    output = await fwd()
    output.compute_grads(backward_fns={"op1": backward1, "op2": backward2})
    assert output.value == 5
    config_op_grad = output.inputs[0][0].inputs[1]["_input_dict"].grad
    assert config_op_grad == {"a": {"b": [3.0, 3.0, 3.0], "c": 3.0}}


@pytest.mark.asyncio
async def test_nested_dict_kwargs_missing_inner_grad_aggregation_fail():
    """Tests that two ops that return different tree.Structure objects crash in grad aggregation."""
    cfg = {"a": {"b": [1, 2, 3], "c": 4}}
    config_op = ConfigOp(cfg)

    op1 = FxnOp[int](lambda _input_dict: 3)
    op2_missing_grad = FxnOp[int](lambda _input_dict: 4)
    op2_missing_grad.set_name("op2_missing_grad")
    agg_op = FxnOp[int](lambda *args: sum(args))  # noqa: FURB111

    def backward_with_missing_leaf(input_args, input_kwargs):
        del input_kwargs["_input_dict"]["a"]["c"]  # missing gradient for this key
        return input_args, input_kwargs

    backward_with_missing_leaf = partial(
        assign_constant_grads_alter_inputs, input_func=backward_with_missing_leaf
    )

    @compute_graph()
    async def fwd() -> OpResult[int]:
        config = await config_op()
        a = await op1(config)
        b = await op2_missing_grad(config)
        return await agg_op(a, b)

    output = await fwd()
    with pytest.raises(ValueError, match="Mismatched gradient structures"):
        output.compute_grads(
            backward_fns={"op2_missing_grad": backward_with_missing_leaf}
        )


@pytest.mark.asyncio
async def test_nested_dict_kwargs_missing_or_extra_inner_grad_ok():
    """Tests that missing or extra gradients for a dict input are not failing."""
    input_dict = {"a": {"b": [1, 2, 3], "c": 4}}

    op = FxnOp[int](lambda _input_dict: 4)
    op.set_name("op")
    agg_op = FxnOp[int](lambda *args: sum(args))  # noqa: FURB111

    def backward_with_missing_leaf(input_args, input_kwargs):
        del input_kwargs["_input_dict"]["a"]["c"]  # missing gradient for this key
        return input_args, input_kwargs

    backward_with_missing_leaf = partial(
        assign_constant_grads_alter_inputs, input_func=backward_with_missing_leaf
    )

    @compute_graph()
    async def fwd() -> OpResult[int]:
        a = await op(input_dict)
        return await agg_op(a)

    output = await fwd()
    output.compute_grads(backward_fns={"op": backward_with_missing_leaf})
    assert output.value == 4

    def backward_with_extra_leaf(input_args, input_kwargs):
        input_kwargs["_input_dict"]["a"]["d"] = 3  # add extra key
        return input_args, input_kwargs

    backward_with_extra_leaf = partial(
        assign_constant_grads_alter_inputs, input_func=backward_with_extra_leaf
    )

    output = await fwd()
    output.compute_grads(backward_fns={"op": backward_with_extra_leaf})
    assert output.value == 4


@pytest.mark.asyncio
async def test_args_missing_or_extra_grad():
    op = FxnOp[int](lambda x: x)
    agg_op = FxnOp[int](lambda *args: sum(args))  # noqa: FURB111
    agg_op.set_name("agg_op")

    def backward_with_missing_arg(input_args, input_kwargs):
        input_args = input_args[:-1]  # remove last arg
        return input_args, input_kwargs

    backward_with_missing_arg = partial(
        assign_constant_grads_alter_inputs, input_func=backward_with_missing_arg
    )

    @compute_graph()
    async def fwd() -> OpResult[int]:
        a = await op(1)
        b = await op(2)
        return await agg_op(a, b)

    output = await fwd()
    with pytest.raises(ValueError, match="argument 2 is shorter than argument 1"):
        output.compute_grads(backward_fns={"agg_op": backward_with_missing_arg})

    def backward_with_extra_arg(input_args, input_kwargs):
        input_args += [3]  # add extra arg
        return input_args, input_kwargs

    backward_with_extra_arg = partial(
        assign_constant_grads_alter_inputs, input_func=backward_with_extra_arg
    )

    output = await fwd()
    with pytest.raises(ValueError, match="argument 2 is longer than argument 1"):
        output.compute_grads(backward_fns={"agg_op": backward_with_extra_arg})


@pytest.mark.asyncio
async def test_kwargs_missing_or_extra_grad() -> None:
    input1 = 1
    input2 = 2
    # Don't use FURB118 here because we assert on kwargs below
    op = FxnOp[int](lambda _input1, _input2: _input1 + _input2)  # noqa: FURB118
    op.set_name("op")

    def backward_with_missing_kwarg(input_args, input_kwargs):
        del input_kwargs["_input1"]
        return input_args, input_kwargs

    backward_with_missing_kwarg = partial(
        assign_constant_grads_alter_inputs, input_func=backward_with_missing_kwarg
    )

    @compute_graph()
    async def fwd() -> OpResult[int]:
        return await op(input1, input2)

    output = await fwd()
    with pytest.raises(ValueError, match="Mismatch between grads"):
        output.compute_grads(backward_fns={"op": backward_with_missing_kwarg})

    def backward_with_extra_kwarg(input_args, input_kwargs):
        input_kwargs["_input3"] = 3
        return input_args, input_kwargs

    backward_with_extra_kwarg = partial(
        assign_constant_grads_alter_inputs, input_func=backward_with_extra_kwarg
    )

    output = await fwd()
    with pytest.raises(ValueError, match="Mismatch between grads"):
        output.compute_grads(backward_fns={"op": backward_with_extra_kwarg})


@pytest.mark.asyncio
async def test_assign_default_grads():  # noqa: RUF029
    input_args: list[ResultOrValue] = [1, 2]
    input_kwargs: dict[str, ResultOrValue] = {
        "a": {"b": 3, "c": 4},
        "d": {"e": {"f": 5}},
    }
    input_grad_args = [0.1]
    input_grad_kwargs = {"a": {"b": 0.2}}

    expected_grad_args = [0.1, 7.0]
    expected_grad_kwargs = {
        "a": {
            "b": 0.2,
            "c": 7.0,
        },
        "d": {"e": {"f": 7.0}},
    }

    input_grads = (input_grad_args, input_grad_kwargs)
    output_grad_args, output_grad_kwargs = assign_default_grads(
        input_grads,
        input_args,
        input_kwargs,
        default_grad_val=7.0,
    )

    assert output_grad_args == expected_grad_args
    assert output_grad_kwargs == expected_grad_kwargs


# test running 2 ops serially without calling @compute_graph
@pytest.mark.asyncio
async def test_serial_ops_diff_run_id():
    op1 = FxnOp[int](lambda x: x + 1)
    op2 = FxnOp[int](lambda x: x * 2)

    result1 = await op1(1)

    with pytest.raises(RuntimeError, match="args and kwargs must have the same run_id"):
        await op2(result1)


@pytest.mark.parametrize("hidden_nodes", [1, 4])
@pytest.mark.asyncio
async def test_torch_param_backward_estimator(hidden_nodes: int):
    torch_module = torch.nn.Sequential(
        torch.nn.Linear(4, hidden_nodes),
        torch.nn.Linear(hidden_nodes, 1),
    )
    torch_op = TorchOp(torch_module)
    estimator = TorchParamBackwardEstimator(torch_module)

    # Forward pass
    result = await torch_op(torch.randn(4, requires_grad=True))

    # Backward pass
    result.compute_grads(backward_fns={torch_op.name: estimator.backward})

    # Check that the gradients are computed and have the correct shape
    call_ids = torch_op.get_call_ids({result.call_id.run_id})
    grad_params = torch_op.ctx.get(next(iter(call_ids)), "grad_params")

    for named_param, grad_param in torch_module.named_parameters():
        assert named_param in grad_params
        assert grad_param.shape == grad_params[named_param].shape
