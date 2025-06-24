from collections.abc import Iterable, Sequence
from typing import cast
from uuid import UUID

import litellm
import pytest
import tenacity
import tree
from aviary.core import Message
from lmi import CommonLLMNames
from lmi import LiteLLMModel as LLMModel
from pydantic import BaseModel, Field, JsonValue

from ldp.agent import Agent, MemoryAgent, ReActAgent
from ldp.alg.optimizer import (
    MemoryFactory,
    MemoryOpt,
    Optimizer,
    default_optimizer_factory,
)
from ldp.alg.optimizer.ape import APEOpt, APEScoreFn, Example
from ldp.data_structures import Trajectory, Transition
from ldp.graph import (
    CallID,
    FxnOp,
    LLMCallOp,
    Memory,
    MemoryOp,
    Op,
    OpCtx,
    OpResult,
    PromptOp,
    compute_graph,
    eval_mode,
)
from ldp.graph.gradient_estimators import (
    llm_straight_through_estimator as llm_ste,
)
from ldp.graph.gradient_estimators import (
    straight_through_estimator as ste,
)
from ldp.graph.ops import GradInType
from ldp.llms.prompts import append_to_sys
from tests.conftest import VCR_DEFAULT_MATCH_ON


@pytest.mark.parametrize(
    ("agent_cls", "optimizer_cls", "optimizer_kwargs"),
    [
        (MemoryAgent, MemoryOpt, {}),
        (ReActAgent, APEOpt, {"score_fn": APEScoreFn.GRADIENT}),
    ],
)
def test_optimizer_factory(
    agent_cls: type[Agent], optimizer_cls: type[Optimizer], optimizer_kwargs: dict
):
    agent = agent_cls()
    opt = default_optimizer_factory(agent, optimizer_cls, **optimizer_kwargs)
    assert isinstance(opt, optimizer_cls)


class SquaredErrorLoss(Op[int]):
    async def forward(self, y: str, yhat: str) -> int:
        try:
            return (int(y) - int(yhat)) ** 2
        except ValueError:  # For example, yhat may be "I guess the number is 7."
            return 100

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args,
        input_kwargs,
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        try:
            y = int(input_kwargs["y"])
            yhat = int(input_kwargs["yhat"])
        except ValueError:
            loss = ctx.get(call_id, "output").value
            return [], {"y": loss, "yhat": loss}  # Straight-through approximation
        # d/dy of (y - y^)^2 = 2 (y - y^), and d/dy^ of (y - y^)^2 = -2 (y - y^)
        # return  dL/dy,  dL/dy^
        # Note that grad_output is ignored because this is assumed to be a terminal scalar,
        # much like calling loss.backward() in pytorch.
        return [], {
            "y": 2 * (y - yhat),
            "yhat": -2 * (y - yhat),
        }


@pytest.mark.asyncio
async def test_ape_optimizer() -> None:
    sys_prompt_op = PromptOp("Guess a number based on the input word.")
    package_msg_op = FxnOp(append_to_sys)
    config = {"max_retries": 3}  # we seem to be hitting rate limits frequently
    llm = LLMModel(config=config)
    llm_call_op = LLMCallOp()
    strip_op = FxnOp(lambda x: x.content)
    loss_op = SquaredErrorLoss()

    @compute_graph()
    async def forward(xi_: str, yi_: str) -> OpResult[int]:
        """Perform a forward pass through the model to the resultant SE loss."""
        s = await sys_prompt_op()
        m = await package_msg_op(xi_, s)
        c = await llm_call_op(config, m)
        yh = await strip_op(c)
        return await loss_op(yi_, yh)

    # Sequentially run a forward pass for each (x, y)
    x = ["Hello", "Day", "Bar"]
    y = [str(len(xi)) for xi in x]  # Number to guess should be word's length
    opt = APEOpt(
        llm=llm,
        llm_call_op=llm_call_op,
        prompt_op=sys_prompt_op,
        good_examples=[
            Example(input=x, output=y, score=0) for x, y in zip(x, y, strict=True)
        ],
        score_fn=APEScoreFn.GRADIENT,
    )
    assert opt.trace == [sys_prompt_op.prompt]

    trajectory = Trajectory()
    for i, (xi, yi) in enumerate(zip(x, y, strict=True)):
        loss = await forward(xi, yi)
        if i == 0:
            assert loss.value > 0, (
                "First example's loss should be non-zero - otherwise, no learning"
                " signal."
            )
        # Sets grad_output and grad_input in context, to be used by optimizer
        loss.compute_grads(backward_fns={LLMCallOp: llm_ste, FxnOp: ste})

        # APE in gradient mode is only going to pay attention to the action, so set
        # placeholders for the other attributes
        trajectory.steps.append(
            Transition(
                timestep=0,
                agent_state=None,
                next_agent_state=None,
                observation=[],
                next_observation=Transition.NO_OBSERVATION,
                action=loss,
                reward=0,
                done=False,
            )
        )

    # Run full optimizer step
    for i in range(3):  # Retries
        opt.aggregate([trajectory])
        assert opt.good_examples == [
            Example(input=x, output=y, score=0) for x, y in zip(x, y, strict=True)
        ]

        await opt.update()
        assert not opt.examples, "Expected reset of examples after update."
        assert len(opt.trace) == i + 2, "Expected new prompt to be recorded."

        with eval_mode():
            if (await forward(xi, yi)).value == 0:  # pylint: disable=undefined-loop-variable
                return

    raise AssertionError("Failed to complete optimization after retries.")


class NumberGuesserModule:
    """Made up module used to enable simple training scripts."""

    def __init__(self):
        self.mem_op = MemoryOp()
        self.package_msg_op = FxnOp(self._package)
        self.llm_call_op = LLMCallOp()
        self.strip_op = FxnOp(lambda x: x.content)

    @staticmethod
    def _package(mems: Iterable[Memory], query: str) -> list[Message]:
        itemized_mems = "\n\n".join(str(m) for m in mems)
        return [
            Message(content="Guess a number based on the input word."),
            Message(content=f"Previous attempts:\n{itemized_mems}\n-----\n\n{query}"),
        ]

    async def __call__(self, query: str) -> tuple[OpResult[str], list[Message]]:
        mems = await self.mem_op(query)
        msgs = await self.package_msg_op(mems, query)
        c = await self.llm_call_op(
            config={
                "name": "gpt-4-turbo",  # this is flaky, so use a smarter model
                "temperature": 0,
                "max_retries": 3,
            },
            msgs=msgs,
        )
        return await self.strip_op(c), msgs.value


async def nondifferentiable_reward_model(target: str, result: OpResult[str]) -> float:
    se = (await SquaredErrorLoss()(target, result)).value
    if se == 0:
        return 1.0  # Positive reward if it got it right
    return -se  # Squared error is positive, so convert to negative


class TestMemoryOpt:
    @staticmethod
    def _mem_opt_failed(exc: BaseException) -> bool:
        # Sometimes the memory opt fails to converge because the training examples
        # are not informative. Try again
        return isinstance(exc, AssertionError) and "should be less than" in str(exc)

    @pytest.mark.flaky(reruns=3, only_on=[litellm.exceptions.APIConnectionError])
    @pytest.mark.asyncio
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception(_mem_opt_failed),
    )
    async def test_standard_memory_optimizer(self) -> None:
        model = NumberGuesserModule()
        # seed with one memory to show example
        await model.mem_op.memory_model.add_memory(
            Memory(
                query="Great",
                output=str(len("Great")),
                value=1.0,
                metadata={"timestep": 0, "done": False, "truncated": False},
            )
        )
        prior_num_memories = 1
        opt = MemoryOpt(memory_op=model.mem_op, output_op=model.llm_call_op)

        x = ["Hello", "Day", "Bar"]
        y = [str(len(xi)) for xi in x]
        trajectory = Trajectory()
        for xi, yi in zip(x, y, strict=True):
            async with compute_graph():
                yh, _ = await model(xi)
                # MemoryOp works with rewards, not gradients. So instead of backpropagating
                # through the loss, for training we compute a non-differentiable reward.
                reward = await nondifferentiable_reward_model(yi, yh)
            yh.compute_grads()

            # MemoryOpt is only going to look at action and reward,
            # so set placeholders for the other values
            trajectory.steps.append(
                Transition(
                    timestep=0,
                    agent_state=None,
                    next_agent_state=None,
                    observation=Transition.NO_OBSERVATION,
                    next_observation=Transition.NO_OBSERVATION,
                    action=yh,
                    reward=reward,
                    done=False,
                )
            )

        opt.aggregate([trajectory])
        await opt.update()

        assert len(model.mem_op.memory_model.memories) == len(x) + prior_num_memories, (
            "Incorrect number of stored memories after optimization step."
        )
        assert all(
            not cast("dict", m.metadata)["done"]
            for m in model.mem_op.memory_model.memories.values()
        )
        assert not opt.example_buffer, (
            "MemoryOpt buffer should be empty after applying update"
        )

        x_eval, y_eval = xi, yi  # pylint: disable=undefined-loop-variable
        async with compute_graph():
            with eval_mode():
                yh, msgs = await model(x_eval)
            assert len(msgs) > 1, "Message should have multiple memories."
            # check that Input appears a few times (from memories)
            assert msgs[-1].content, "unexpected message content"
            assert msgs[-1].content.count("Input") > 2, (
                "Input should appear multiple times in the response."
            )
            se_loss = (await SquaredErrorLoss()(y_eval, yh)).value

        assert se_loss < 100, (
            f"Loss ({se_loss:.3f}) should be less than 100 after training."
        )

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.asyncio
    async def test_lessons_memory_optimizer(self) -> None:
        """
        Test we can use LLM completions to generate lessons instead of memories.

        This test is loosely based on Reflexion (https://arxiv.org/abs/2303.11366).
        """
        memory_distiller = LLMModel(config={"name": CommonLLMNames.OPENAI_TEST.value})

        class LessonEntry(BaseModel):
            """Entry for a lesson created from some example data."""

            query: str = Field(
                description=(
                    "Plain text string for retrieving this lesson from a database of"
                    " lessons."
                )
            )
            lesson: str = Field(description="Lesson generated from past attempts.")

            @staticmethod
            def make_prompt(examples: Sequence[tuple]) -> str:
                """Create an LLM prompt to generate a lesson from examples."""
                itemized_examples = "\n-".join(str(x) for x in examples)
                return (
                    "We are trying to guess a number based on the input word. We just"
                    f" tried {len(examples)} times, and collected rewards where a"
                    " higher reward is better. Here are the results in three-tuples of"
                    f" input, output, reward:\n- {itemized_examples}\n\nPlease create"
                    " a lesson based on this data referencing the relative success or"
                    " failure associated with the reward, use concise wording and"
                    " don't repeat yourself."
                )

            @classmethod
            def to_memory(cls, lesson_json: str, run_id: UUID | None = None) -> Memory:
                lesson = cls.model_validate_json(lesson_json)
                return Memory(
                    query=lesson.query, output=lesson.lesson, value="", run_id=run_id
                )

            @classmethod
            async def memory_factory(
                cls,
                memory_op: MemoryOp,
                output_op: Op[Message],
                memory_template: str,
                example_buffer: Iterable[tuple[CallID, CallID, float, JsonValue]],
            ) -> list[Memory]:
                example_buffer = list(example_buffer)
                if not example_buffer:
                    raise RuntimeError("Expected non-empty example buffer.")
                query_airesponse_dreturns: list[tuple[str, str, float]] = [
                    (
                        memory_op.ctx.get(mem_call_id, "query"),
                        output_op.ctx.get(output_call_id, "output").value.content,
                        d_return,
                    )
                    for mem_call_id, output_call_id, d_return, metadata in example_buffer
                ]
                response = await memory_distiller.call_single(
                    messages=[
                        Message(
                            content=LessonEntry.make_prompt(query_airesponse_dreturns)
                        )
                    ],
                    tool_choice=memory_distiller.NO_TOOL_CHOICE,
                    output_type=LessonEntry,
                )
                if (
                    not response.messages
                    or len(response.messages) != 1
                    or not response.messages[0].content
                ):
                    raise ValueError(
                        "Expected a single message in the response containing a"
                        f" serialized {cls.__name__}."
                    )
                return [
                    cls.to_memory(
                        response.messages[0].content, run_id=example_buffer[0][0].run_id
                    )
                ]

        assert isinstance(LessonEntry.memory_factory, MemoryFactory)

        model = NumberGuesserModule()
        opt = MemoryOpt(
            memory_op=model.mem_op,
            output_op=model.llm_call_op,
            memory_factory=LessonEntry.memory_factory,
        )

        x = ["Hello", "Day", "Bar"]
        y = [str(len(xi)) for xi in x]
        trajectory = Trajectory()
        for xi, yi in zip(x, y, strict=True):
            async with compute_graph():
                yh, _ = await model(xi)
                # MemoryOp works with rewards, not gradients. So instead of backpropagating
                # through the loss, for training we compute a non-differentiable reward.
                reward = await nondifferentiable_reward_model(yi, yh)
            yh.compute_grads()

            # MemoryOpt is only going to look at action and reward,
            # so set placeholders for the other values
            trajectory.steps.append(
                Transition(
                    timestep=0,
                    agent_state=None,
                    next_agent_state=None,
                    observation=Transition.NO_OBSERVATION,
                    next_observation=Transition.NO_OBSERVATION,
                    action=yh,
                    reward=reward,
                    done=False,
                )
            )

        opt.aggregate([trajectory])
        await opt.update()

        assert not opt.example_buffer, (
            "MemoryOpt buffer should be empty after applying update"
        )
