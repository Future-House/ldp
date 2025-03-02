import asyncio
import operator
import random
from typing import TypeVar, cast
from uuid import UUID

import litellm
import numpy as np
import pytest
import tenacity
import tree
from aviary.core import DummyEnv, Message, Tool, ToolRequestMessage
from lmi import CommonLLMNames, LLMResult
from lmi import LiteLLMModel as LLMModel

from ldp.graph import (
    CallID,
    ConfigOp,
    EmbeddingOp,
    FxnOp,
    LLMCallOp,
    Op,
    OpCtx,
    OpResult,
    PromptOp,
    compute_graph,
    eval_mode,
    get_call_id,
    get_run_id,
    set_training_mode,
)
from ldp.graph.gradient_estimators import straight_through_estimator as ste
from ldp.graph.ops import GradInType, ResultOrValue, TOutput_co
from ldp.llms.prompts import append_to_sys


class StatefulFxnOp(FxnOp[TOutput_co]):
    async def forward(self, *args, **kwargs) -> TOutput_co:
        result = await super().forward(*args, **kwargs)
        self.ctx.update(get_call_id(), "observed", value=True)
        return result


@pytest.mark.asyncio
async def test_call_ids() -> None:
    async def fxn(x) -> int:
        await asyncio.sleep(x)
        return int(x > 0)

    op = StatefulFxnOp[int](fxn)
    # In this test, we want to make sure that op calls can't interfere with each
    # each other's compute graphs. So we launch two tasks, where the second
    # one should finish before the first, allowing for the possibility of a
    # clobber.
    xs = [1, 0.5]

    async def call_op(x) -> tuple[OpResult, UUID]:
        async with compute_graph():
            result = await op(x)
            return result, get_run_id()

    results = await asyncio.gather(*[call_op(x) for x in xs])

    for result, run_id in results:
        assert run_id == result.call_id.run_id, "Inconsistent run IDs"
        assert op.ctx.get(key="observed", call_id=result.call_id) is True


@pytest.mark.asyncio
async def test_FxnOp() -> None:
    def fxn(x) -> int:
        return int(x > 0)

    op = FxnOp(fxn)
    async with compute_graph():
        op_result = await op(1)
    assert op_result.value == 1
    assert op_result.logprob is None

    with pytest.raises(ValueError, match="No gradients"):
        op.get_input_grads(op_result.call_id)
    op_result.compute_grads(1.0, backward_fns={FxnOp: ste})
    assert op_result.grad == 1.0, "We didn't persist the starting gradient"
    assert op.get_input_grads(op_result.call_id) == ([], {"x": 1})


T = TypeVar("T")


@pytest.mark.parametrize(
    "op_return",
    [
        (1, int),
        (1.0, float),
        ("1", str),
        (Message(content="1"), Message),
        (ToolRequestMessage(content="1"), ToolRequestMessage),
    ],
)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.asyncio
async def test_opresult_typing(op_return: tuple[T, type[T]], training: bool) -> None:
    op_return_value, op_return_type = op_return
    op = FxnOp[op_return_type](lambda: op_return_value)  # type: ignore[valid-type]

    set_training_mode(training)
    async with compute_graph():
        op_result = await op()

    # Confirm that the OpResult's type matches the value's type
    assert isinstance(op_result, OpResult)
    assert isinstance(op_result.value, op_return_type)


class TestLLMCallOp:
    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_cost_tracking(self) -> None:
        model_name = "gpt-3.5-turbo"

        class LLMCallingEnv(DummyEnv):
            """Showing how environments can use LiteLLM to track their own costs."""

            def __init__(self):
                self.total_cost = 0.0

            async def reset(self) -> tuple[list[Message], list[Tool]]:
                async def generate_story() -> str:
                    """Generate a story."""
                    response = await litellm.acompletion(
                        model=model_name,
                        messages=[
                            {"content": "Please write a 5 word story", "role": "user"}
                        ],
                    )
                    self.total_cost += litellm.completion_cost(response)
                    return response.choices[0].message.content

                self.state = type(self).State(
                    messages=[Message(content="Generate a story")]
                )
                self.tools = [Tool.from_function(generate_story)]
                return self.state.messages, self.tools

        env = LLMCallingEnv()
        obs, tools = await env.reset()
        config = {"name": model_name, "temperature": 0.1}
        llm_op = LLMCallOp()

        # Perform one step
        async with compute_graph():
            op_result = await llm_op(config, msgs=obs, tools=tools)
        await env.exec_tool_calls(cast("ToolRequestMessage", op_result.value))

        # LLMCallOp track cost using run context
        result = llm_op.ctx.get(op_result.call_id, "result")
        assert result.cost > 0

        # Environment tracks its internal costs
        assert env.total_cost > 0

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_empty_tools(self) -> None:
        llm_call_op = LLMCallOp()
        message_result = await llm_call_op(
            LLMModel.model_fields["config"].default_factory(),  # type: ignore[call-arg, misc]
            msgs=[Message(content="Hello")],
            tools=[],
        )
        assert isinstance(message_result.value, ToolRequestMessage)
        assert not message_result.value.tool_calls

    @pytest.mark.vcr
    @pytest.mark.asyncio
    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    async def test_compute_logprob(self, temperature) -> None:
        config = {
            "name": CommonLLMNames.OPENAI_TEST.value,
            "temperature": temperature,
            "logprobs": True,
        }
        llm_op = LLMCallOp()
        output = await llm_op(config, msgs=[Message(content="Hello")])
        logp = llm_op.ctx.get(output.call_id, "logprob")
        raw_logp = llm_op.ctx.get(output.call_id, "raw_logprob")
        if temperature == 0.0:
            assert logp == 1.0
        elif temperature == 1.0:
            assert logp == raw_logp
        else:
            assert logp is None

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_validation(self) -> None:
        config = {"name": CommonLLMNames.OPENAI_TEST.value, "num_retries": 1}
        validator = StatefulValidator()
        llm_op = LLMCallOp(response_validator=validator)
        await llm_op(config, msgs=[Message(content="Hello")])
        assert validator.counter == 2  # first attempt should have failed

        config = {"name": CommonLLMNames.OPENAI_TEST.value, "num_retries": 0}
        llm_op = LLMCallOp(response_validator=StatefulValidator())
        with pytest.raises(tenacity.RetryError, match="ResponseValidationError"):
            await llm_op(config, msgs=[Message(content="Hello")])


class StatefulValidator:
    def __init__(self):
        self.counter = 0

    def __call__(self, response: LLMResult) -> None:
        self.counter += 1
        if self.counter == 1:
            raise ValueError("Invalid response")


@pytest.mark.asyncio
async def test_simple_prompt_graph() -> None:
    config = ConfigOp(config={"name": "hello"})
    prompt = PromptOp("Hello, {name}! You are {age} years old.")

    async with compute_graph():
        c = await config()
        s = await prompt(c, age=20)
    assert s.value == "Hello, hello! You are 20 years old."

    my_loss_grad = -2.0
    s.compute_grads(my_loss_grad, backward_fns={PromptOp: ste})
    grad = prompt.get_input_grads(s.call_id)
    assert grad[1]["age"] == -2.0


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_llm_call_graph() -> None:
    sys_prompt_op = PromptOp(
        "Respond by first planning your actions, then write code, "
        "inspect its effect, and reason about correctness"
    )
    user_prompt_op = PromptOp("What is the result of this math equation: {equation}?")

    package_msg_op = FxnOp(append_to_sys)
    config = {
        "name": "gpt-3.5-turbo-0125",
        "temperature": 0.1,
        "logprobs": True,
        "top_logprobs": 1,
    }
    config_op = ConfigOp(config=config)

    # Now forward pass
    my_equation = "2 + 2"
    async with compute_graph():
        sys_prompt = await sys_prompt_op()
        user_prompt = await user_prompt_op(equation=my_equation)
        package_msg = await package_msg_op(user_prompt, sys_prompt)
        c = await config_op()
        result = await (llm_op := LLMCallOp())(c, package_msg)
    assert result.value is not None
    assert len(result.value.content) > 10  # type: ignore[arg-type]

    output_grad = -2.0  # some grad accrued from result
    result.compute_grads([output_grad])

    # check some grads are present
    _, g = llm_op.get_input_grads(result.call_id)
    assert g["config"] == dict.fromkeys(config, 0.0)
    assert g["msgs"] == 0.0

    _, g = config_op.get_input_grads(c.call_id)
    assert not g  # config op has no inputs

    _, g = user_prompt_op.get_input_grads(user_prompt.call_id)
    assert g["equation"] == 0

    # get examples
    assert llm_op.get_examples()

    # now inference pass
    with eval_mode():
        async with compute_graph():
            sys_prompt = await sys_prompt_op()
            user_prompt = await user_prompt_op(equation=my_equation)
            package_msg = await package_msg_op(user_prompt, sys_prompt)
            c = await config_op()
            result = await (llm_op := LLMCallOp())(c, package_msg)
    assert result.value.content is not None
    assert len(result.value.content) > 10


@pytest.mark.asyncio
async def test_nested_op():
    """Test that we can have a forward function that calls another Op."""
    inner_op_a = FxnOp(lambda x: x + 1)
    inner_op_b = FxnOp(lambda x: x + 1)

    async def nested_op(x: ResultOrValue[int]) -> OpResult[int]:
        async with compute_graph():
            x = await inner_op_a(x)
            return await inner_op_b(x)

    result = await nested_op(1)
    assert result.value == 3

    input_args, input_kwargs = result.inputs
    assert not input_args
    assert input_kwargs

    result.compute_grads(1)
    assert result.grad == 1


class PickFirstOp(Op[int]):
    async def forward(self, *args: int) -> int:
        return args[0]

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args,
        input_kwargs,
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        return [grad_output] + [None] * (len(input_args) - 1), {}


@pytest.mark.asyncio
async def test_multiple_op_calls():
    op_a = FxnOp[int](lambda x: x)
    op_b = PickFirstOp()

    # compute graph: A -> B <- A
    async with compute_graph():
        n_samples = 2
        samples = await asyncio.gather(*[op_a(i) for i in range(n_samples)])
        selected = await op_b(*samples)
        assert len(samples) == n_samples

        run_id = get_run_id()
        op_a_call_ids = op_a.get_call_ids({run_id})
        assert len(op_a_call_ids) == n_samples
        assert selected.value == 0

    selected.compute_grads(10.0, backward_fns={FxnOp: ste})
    for call_id in op_a_call_ids:
        call_idx = op_a.ctx.get(call_id, "output").value
        if call_idx == 0:
            # first call - grads should've backproped
            assert op_a.get_input_grads(call_id)[1]["x"] == 10.0
        else:
            # second call - compute graph should have been pruned by
            # PickFirstOp
            with pytest.raises(
                ValueError, match=r"No gradients have been computed for .*"
            ):
                op_a.get_input_grads(call_id)


@pytest.mark.asyncio
async def test_branching_compute_graph():
    # The goal of this test is to make sure that gradients are properly aggregated
    # when the output of an op is consumed by multiple downstream ops. We expect
    # the gradients from each downstream op to be summed together.

    op_a = FxnOp[int](lambda x: x)
    op_b: FxnOp[int] = FxnOp(lambda x: x + random.randint(1, 10))
    op_c = FxnOp[int](lambda *x: sum(x))  # noqa: FURB111

    # compute graph: a -> b1, b2 -> c
    async with compute_graph():
        a = await op_a(3)
        b1 = await op_b(a)
        b2 = await op_b(a)
        c = await op_c(b1, b2)

    loss_grad = -5.0
    c.compute_grads(loss_grad, backward_fns={FxnOp: ste})

    # b1, b2, c receive only one copy of gradient
    for result in (c, b2, b2):
        assert result.grad == loss_grad

    # since a is used by two op calls, it should receive 2x the gradient:
    assert a.grad == loss_grad * 2

    # and 2x gradient should be passed back to the input
    assert op_a.get_input_grads(a.call_id)[1]["x"] == loss_grad * 2


@pytest.mark.asyncio
async def test_clear_contexts():
    """Test that we can clear the contexts of Ops."""
    op1 = FxnOp(lambda x: x)
    op1.set_name("op1")
    op2 = FxnOp(lambda x: x)
    op2.set_name("op2")
    async with compute_graph():
        # Test global clear
        await op2(await op1(1))
        assert len(op1.ctx.data) == len(op2.ctx.data) == 1
        OpCtx.clear_contexts()
        assert not op1.ctx.data
        assert not op2.ctx.data

        # Test global clear by op name
        await op2(await op1(1))
        OpCtx.clear_contexts(op_names=["op1"])
        assert not op1.ctx.data
        assert len(op2.ctx.data) == 1

        # Test instance clear
        op2.clear_ctx()
        assert not op2.ctx.data


@pytest.mark.asyncio
@pytest.mark.vcr
@pytest.mark.parametrize("dense_embedding_size", [0, 256, 512])
@pytest.mark.parametrize("sparse_embedding_size", [0, 32, 64])
@pytest.mark.parametrize("name", ["text-embedding-3-small", "text-embedding-3-large"])
async def test_embedding_op(name, dense_embedding_size, sparse_embedding_size) -> None:
    if dense_embedding_size == sparse_embedding_size == 0:
        return
    op = EmbeddingOp(
        dense_embedding=name,
        dense_embedding_dim=dense_embedding_size,
        sparse_embedding_dim=sparse_embedding_size,
    )
    async with compute_graph():
        op_result = await op("test text")
    assert isinstance(op_result.value, np.ndarray)
    assert op_result.value.shape[0] == dense_embedding_size + sparse_embedding_size
    op_result.compute_grads()
    assert op.get_input_grads(op_result.call_id) == ([], {"string_input": None})


@pytest.mark.asyncio
async def test_op_lookup():
    op_a = FxnOp[int](lambda: 1)
    op_a.set_name("op_a")

    # don't set name on these two to make sure default names
    # still map uniquely
    op_b = FxnOp[int](lambda x: 2 * x)
    op_c = FxnOp[int](operator.neg)

    async with compute_graph():
        a = await op_a()
        b = await op_b(a)
        c = await op_c(b)

    assert a.op == op_a
    assert b.op == op_b
    assert c.op == op_c
