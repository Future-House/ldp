from __future__ import annotations

import asyncio
import logging
from collections import UserDict
from enum import StrEnum, auto
from typing import Any, Self, cast

from aviary.core import Message
from lmi import LiteLLMModel as LLMModel
from lmi import LLMResult
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    RootModel,
    ValidationError,
    model_validator,
)

from ldp.agent import ReActAgent
from ldp.alg.optimizer.opt import Optimizer
from ldp.data_structures import Trajectory
from ldp.graph import LLMCallOp, OpResult, PromptOp

logger = logging.getLogger(__name__)


class APEScoreFn(StrEnum):
    # Use the reward as the APE score (as proposed in the paper).
    # Goal is to maximize this score.
    REWARD = auto()
    # Use the gradient of the output of the PromptOp as the APE score.
    # Goal is to push this to zero.
    GRADIENT = auto()


class _FormatDict(UserDict):
    """Custom dictionary that stores missing items."""

    def __init__(self) -> None:
        super().__init__()
        self.key_set: set[str] = set()

    def __missing__(self, key: str) -> str:
        self.key_set.add(key)
        return key


def get_formatted_variables(s: str) -> set[str]:
    """Returns the set of variables implied by the format string."""
    format_dict = _FormatDict()
    s.format_map(format_dict)
    return format_dict.key_set


class OutputPrompt(BaseModel):
    prompt: str = Field(description="Prompt for language model")


class Example(BaseModel):
    input: JsonValue
    output: JsonValue
    score: float


ExampleList = RootModel[list[Example]]


class APEOpt(BaseModel, Optimizer):
    """
    Basic optimizer that acts as an Automatic Prompt Engineer (APE).

    Paper: https://openreview.net/pdf?id=92gvk82DE-

    Details:
    - This implements the "forward mode generation" strategy.
    - The score function used is the gradient (float) at the output of the
      PromptOp being optimized. A zero gradient means the prompt was "good",
      and a non-zero gradient means we can learn from the prompt.
    - Possible improvements include:
        - Extending the score function to the LLM result's logprobs
        - Iterating with Monte Carlo Search
        - Use of memory for further example augmentation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Configuration
    max_examples: int | None = Field(
        default=50,  # Comes from APE paper section 5.2
        description=(
            "Max number of examples to include in the below query_prompt, or None for"
            " no limit. The paper mentions that more examples produce better prompt"
            " proposals."
        ),
    )
    system_prompt: str = (
        "We are optimizing prompts for a language model."
        " The model sees a prompt, an input, and then generates an output."
    )
    query_prompt: str = (
        "Here are correct example outputs that the language model"
        " and prompt should produce:\n{good_examples}"
        '\n\nThe current prompt is: "{prompt}"'
        "\n\nWhich resulted in the following incorrect input, output, and {score}:"
        "\n{examples}\n\nRevise the current prompt to improve the outputs."
        " Your proposed prompt should be concise, correct, and specify the desired"
        " output format."
    )
    llm: LLMModel = Field(
        default_factory=LLMModel,
        description=(
            "LLM used to update the prompt inside the PromptOp. The paper mentions that"
            " larger models produce better prompt proposals."
        ),
    )

    prompt_op: PromptOp = Field(description="PromptOp to be optimized.")
    llm_call_op: LLMCallOp = Field(description="LLMCallOp to be optimized.")

    score_fn: APEScoreFn = APEScoreFn.REWARD
    good_reward_threshold: float | None = Field(
        default=None,
        description=(
            "If using reward as the score_fn, then a good example is defined by "
            "reward>=good_reward_threshold."
        ),
    )
    reward_discount: float = 1.0

    # State
    examples: list[Example] = Field(default_factory=list)
    good_examples: list[Example] = Field(default_factory=list)
    steps: int = 0
    trace: list[str] = Field(
        default_factory=list, description="History of prompts used."
    )

    @model_validator(mode="after")
    def validate_score_fn(self):
        if self.score_fn == APEScoreFn.REWARD:
            if self.good_reward_threshold is None:
                raise ValueError(
                    "good_reward_threshold must be set if using reward as the score"
                    " function"
                )
            self._score_str = "rewards"
        elif self.score_fn == APEScoreFn.GRADIENT:
            # The gradient into the prompt op is the (signed) backpropagated error, and "gradient" would
            # be confusing to the model in the prompt.
            self._score_str = "errors"
        else:
            raise ValueError(f"Invalid score function {self.score_fn}")

        return self

    def model_post_init(self, __context: Any) -> None:  # noqa: PYI063
        if self.prompt_op.prompt not in self.trace:
            self.trace.append(self.prompt_op.prompt)

        # Make sure updates are not run concurrently
        self._update_lock = asyncio.Lock()

    @classmethod
    def from_agent(cls, agent: ReActAgent, **kwargs) -> Self:
        return cls(
            llm_call_op=agent._react_module.llm_call_op,
            prompt_op=agent._react_module.prompt_op,
            **kwargs,
        )

    def aggregate_trajectory(self, trajectory: Trajectory) -> None:
        if trajectory.failed:
            return

        if self.score_fn == APEScoreFn.REWARD:
            d_returns = trajectory.compute_discounted_returns(self.reward_discount)

        for i_step, step in enumerate(trajectory.steps):
            action = cast("OpResult", step.action)

            if self.score_fn == APEScoreFn.GRADIENT:
                prompt_op_result, *extra = action.get_upstream_results(self.prompt_op)
                assert not extra, "APE only supports one prompt call per run"

            for op_result in action.get_upstream_results(self.llm_call_op):
                result = cast(
                    "LLMResult | None",
                    self.llm_call_op.ctx.get(op_result.call_id, "result"),
                )
                if result is None or not result.messages or not result.prompt:
                    continue
                # (x: first prompt's user message's content, y: AI response's content)
                x = next(
                    # m is a Message with a result of the LLM. Which completes only strings.
                    # and we checked the result exists above.
                    cast("str", m.content)
                    for m in result.prompt
                    if (isinstance(m, Message) and m.role == "user")
                )
                y = cast("str", result.messages[0].content)

                if self.score_fn == APEScoreFn.GRADIENT:
                    score = self.prompt_op.ctx.get(
                        prompt_op_result.call_id, "grad_output", default=None
                    )
                    if score is None:
                        # backprop did not reach this op call - move on
                        continue
                    is_good = score == 0
                else:
                    score = d_returns[i_step]  # pylint: disable=possibly-used-before-assignment
                    is_good = score >= cast("float", self.good_reward_threshold)

                example = Example(input=x, output=y, score=score)
                (self.good_examples if is_good else self.examples).append(example)

    async def update(self) -> None:
        async with self._update_lock:
            if not self.examples:
                raise ValueError("No examples to update the prompt with.")

            new_p = await self._get_updated_prompt(
                self.examples, self.good_examples, self.prompt_op.prompt
            )
            # Check any template vars remain, and if some were added or
            # lost, discard this new prompt
            if new_p != self.prompt_op.prompt and get_formatted_variables(
                new_p
            ) != get_formatted_variables(self.prompt_op.prompt):
                logger.warning(
                    "Update broke prompt templating."
                    f"\n\nNew prompt:\n{new_p}"
                    f"\n\nPrior prompt:\n{self.prompt_op.prompt}"
                )
            else:
                if new_p == self.prompt_op.prompt:
                    logger.warning("Update did not change the prompt.")
                self.examples.clear()
                self.prompt_op.prompt = new_p
                self.trace.append(new_p)
            self.steps += 1

    def _prepare_examples(self, examples: list[Example]) -> str:
        if not examples:
            return ""
        if self.max_examples and len(examples) > self.max_examples:
            if self.score_fn == APEScoreFn.GRADIENT:
                # Return examples closest to decision boundary,
                # aka ones with the lowest L1-normalized error
                # NOTE: this pairs with our setting of Example.score = PromptOp's output
                # gradient inside the update method, so examples with error values closer to
                # 0 are defined to be higher quality
                examples = sorted(examples, key=lambda e: abs(e.score))[
                    : self.max_examples
                ]
            else:
                # In reward mode, we want to show the examples with the highest reward, per the paper
                # TODO: consider whether uniform sampling is better
                examples = sorted(examples, key=lambda e: -e.score)[: self.max_examples]
        return ExampleList.model_validate(examples).model_dump_json()

    async def _get_updated_prompt(
        self, examples: list[Example], good_examples: list[Example], prompt: str
    ) -> str:
        messages = [
            Message(
                role="system",
                content=self.system_prompt,
            ),
            Message(
                role="user",
                content=self.query_prompt.format(
                    examples=self._prepare_examples(examples),
                    good_examples=self._prepare_examples(good_examples),
                    prompt=prompt,
                    score=self._score_str,
                ),
            ),
            Message(
                role="assistant",
                content=OutputPrompt(prompt=prompt).model_dump_json(indent=2),
            ),
            Message(
                content=(
                    "You responded without changing the prompt. Don't forget to revise"
                    " the prompt."
                )
            ),
        ]
        result = await self.llm.call_single(messages, output_type=OutputPrompt)
        message_content = cast(
            "str", cast("list[Message]", result.messages)[-1].content
        )
        try:
            return OutputPrompt.model_validate_json(message_content).prompt
        except ValidationError:
            return prompt
