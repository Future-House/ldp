import itertools
from collections.abc import Iterable
from typing import cast
from uuid import UUID

import litellm
import pytest
import tenacity
import tree
from aviary.core import Message
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
from ldp.llms import LLMModel, append_to_sys
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
    llm = LLMModel()
    llm.config["max_retries"] = 3  # we seem to be hitting rate limits frequently
    llm_call_op = LLMCallOp()
    strip_op = FxnOp(lambda x: x.content)
    loss_op = SquaredErrorLoss()

    @compute_graph()
    async def forward(xi_: str, yi_: str) -> OpResult[int]:
        """Perform a forward pass through the model to the resultant SE loss."""
        s = await sys_prompt_op()
        m = await package_msg_op(xi_, s)
        c = await llm_call_op(llm.config, m)
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

    def __init__(self, matches: int = MemoryOp.DEFAULT_NUM_MATCHES):
        self.matches = matches
        self.mem_op = MemoryOp()
        self.package_msg_op = FxnOp(self._package)
        self.llm_call_op = LLMCallOp()
        self.strip_op = FxnOp(lambda x: x.content)

    @staticmethod
    def _package(mems: Iterable[Memory], query: str) -> list[Message]:
        itemized_mems = "\n\n".join(str(m) for m in mems)
        return [
            Message(
                content=(
                    "Guess a number based on the input word, and make sure the response"
                    " only contains the guessed number."
                )
            ),
            Message(
                content=(
                    f"Previous memories:\n{itemized_mems}\n-----\n\nInput word:"
                    f" {query!r}"
                )
            ),
        ]

    async def __call__(self, query: str) -> tuple[OpResult[str], list[Message]]:
        mems = await self.mem_op(query, matches=self.matches)
        msgs = await self.package_msg_op(mems, query)
        c = await self.llm_call_op(
            config={
                "model": "gpt-4-turbo",  # this is flaky, so use a smarter model
                "temperature": 0,
                "max_retries": 3,
            },
            msgs=msgs,
        )
        return await self.strip_op(c), msgs.value


CORRECT_REWARD = 1.0


async def nondifferentiable_reward_model(target: str, result: OpResult[str]) -> float:
    se = (await SquaredErrorLoss()(target, result)).value
    if se == 0:
        return CORRECT_REWARD  # Positive reward if it got it right
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
            not cast(dict, m.metadata)["done"]
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
    async def test_lessons_memory_optimizer(self) -> None:  # noqa: C901
        """
        Test we can use LLM completions to generate lessons instead of memories.

        This test is loosely based on Reflexion (https://arxiv.org/abs/2303.11366).
        """
        memory_distiller = LLMModel(config={"model": "gpt-4o-2024-11-20"})

        class LessonEntry(BaseModel):
            """Entry for a lesson created from some example data."""

            query: str = Field(
                description=(
                    "Plain text string for retrieving this lesson from a database of"
                    " lessons. Please fill this field with specific inputs and outputs,"
                    " so we can fetch relevant lessons in the future."
                )
            )
            lesson: str = Field(
                description=(
                    "Lesson generated from past attempts. Please fill this field with"
                    " specific inputs and outputs, as well as speculate on possible"
                    " ways to attain desirable results."
                )
            )

            @classmethod
            def to_memory(
                cls,
                lesson_json: str,
                run_id: UUID | None = None,
                value: float | str = "",
            ) -> Memory:
                lesson: LessonEntry = cls.model_validate_json(lesson_json)
                return Memory(
                    query=lesson.query,
                    output=lesson.lesson,
                    value=value,
                    run_id=run_id,
                    template="{output}\n\nReturn: {value}",
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
                query_airesponse_dreturns: list[tuple[str, str, float]] = [
                    (
                        memory_op.ctx.get(mem_call_id, "query"),
                        output_op.ctx.get(output_call_id, "output").value.content,
                        d_return,
                    )
                    for mem_call_id, output_call_id, d_return, _ in example_buffer
                ]
                itemized_examples = "\n-".join(
                    str(x) for x in query_airesponse_dreturns
                )
                message_history = [
                    Message(
                        content=(
                            "We are trying to guess a number based on the input word."
                            f" We just tried {len(query_airesponse_dreturns)} times,"
                            " and collected rewards where a higher reward is better."
                            " Here are the results in three-tuples of input, output,"
                            f" reward:\n- {itemized_examples}\n\nPlease create a lesson"
                            " based on this data referencing the relative success or"
                            " failure associated with the reward, use concise wording"
                            " and don't repeat yourself."
                        )
                    )
                ]
                response = await memory_distiller.call(
                    messages=message_history,
                    tool_choice=memory_distiller.UNSPECIFIED_TOOL_CHOICE,
                    output_type=LessonEntry.model_json_schema(),
                )
                lesson_message, lesson_json = response.extract_single_message_content()
                memory = cls.to_memory(
                    lesson_json=lesson_json,
                    run_id=example_buffer[0][0].run_id,
                    value=sum(x[2] for x in query_airesponse_dreturns),
                )
                prefix = "Given that lesson"
                if memory_op.memory_model.memories:
                    # Lead the prompt with the best memories, but don't sort the
                    # memories because the LLM may interpret ordering as a progression
                    # of heuristics to align with
                    best_memories: list[Memory] = []
                    for m in memory_op.memory_model.memories.values():
                        if isinstance(m.value, float):
                            if len(best_memories) < 8:
                                best_memories.append(m)
                                continue
                            argmin_memory = min(
                                range(len(best_memories)),
                                key=lambda x: best_memories[x].value,
                            )
                            if best_memories[argmin_memory].value < m.value:  # type: ignore[operator]
                                best_memories[argmin_memory] = m
                    div = "\n\n---\n\n"
                    prefix += (
                        ", for reference here are some past lessons with proposed"
                        " heuristics, separated by dashed"
                        f" lines:{div}{div.join(str(m) for m in best_memories)}{div}Now"
                    )
                message_history.extend([
                    lesson_message,
                    Message(
                        content=(
                            f"{prefix}, let's consider proposing a new heuristic to"
                            " predict the output based on the input. If the results"
                            f" had rewards equal to {CORRECT_REWARD}, propose the same"
                            " heuristic as it was correct. Otherwise, the heuristic"
                            " should be a complete and precise algorithm or"
                            " performance metric, not a vague statement. Do no restate"
                            " or reference any lesson in the heuristic. If the past"
                            " rewards were quite negative, please come up with a"
                            " completely different heuristic. Higher rewards are"
                            " better, so consider comparing lessons and heuristics"
                            " based on reward. Do not begin with phrases like"
                            " 'Heuristic: ', 'Proposed heuristic: ', or 'New"
                            " heuristic: '.\n\nGood examples are 'Sum the ordinal"
                            " value of each character in the word.' or 'Count the"
                            " number of unique letters in the word'.\n\nBad examples"
                            " are 'Heuristic: Potentially favoring previously"
                            " established successful pairings.' or 'Select outputs"
                            " closer to 12.'."
                        )
                    ),
                ])
                response = await memory_distiller.call(
                    messages=message_history,
                    tool_choice=memory_distiller.UNSPECIFIED_TOOL_CHOICE,
                )
                memory.output = (
                    f"Lesson: {memory.output}\n\nProposed heuristic:"
                    f" {response.extract_single_message_content()[1]}"
                )
                return [memory]

        assert isinstance(LessonEntry.memory_factory, MemoryFactory)

        model = NumberGuesserModule(matches=6)
        opt = MemoryOpt(
            memory_op=model.mem_op,
            output_op=model.llm_call_op,
            memory_factory=LessonEntry.memory_factory,
        )

        async def train(x: Iterable[str], y: Iterable[str]) -> None:
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

        @eval_mode()
        async def evaluate(x: Iterable[str], y: Iterable[str]) -> bool:
            se_losses = []
            for xi, yi in zip(x, y, strict=True):
                async with compute_graph():
                    yh, _ = await model(xi)
                    se_losses.append((await SquaredErrorLoss()(yi, yh)).value)
            return all(se_loss == 0 for se_loss in se_losses)

        all_x: list[Iterable[str]] = [
            ["Bird", "Bat"],
            ["Hello", "World"],
            ["Head", "Shoulders", "Knees", "Toes"],
            ["Spam", "Ham", "Eggs"],
            ["Cat", "Dog", "Fish"],
            ["Mango", "Banana", "Apples"],
            ["Turnips", "Lettuce"],
            ["Eyes", "Ears", "Mouth", "Nose"],
            ["Coffee", "Tea", "Crumpets"],
            ["Perception", "Is", "Reality"],
            ["Life", "And", "Death"],
            ["Car", "Boat", "Plane"],
        ]
        all_y = [[str(len(xi)) for xi in x] for x in all_x]
        success = False
        for (train_x, train_y), (eval_x, eval_y) in itertools.pairwise(
            zip(all_x, all_y, strict=True)
        ):
            await train(train_x, train_y)
            if await evaluate(eval_x, eval_y):
                success = True

        assert success, "Failed to complete optimization."
        assert not opt.example_buffer, (
            "MemoryOpt buffer should be empty after applying update"
        )
