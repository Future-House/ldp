from typing import ClassVar
from unittest.mock import Mock

import litellm
import numpy as np
import pytest
from aviary.core import DummyEnv, Message, Tool, ToolRequestMessage
from pydantic import BaseModel, Field

from ldp.llms import (
    JSONSchemaValidationError,
    LLMModel,
    LLMResult,
    MultipleCompletionLLMModel,
    validate_json_completion,
)

from . import CILLMModelNames


def test_json_schema_validation() -> None:
    # Invalid JSON
    mock_completion1 = Mock()
    mock_completion1.choices = [Mock()]
    mock_completion1.choices[0].message.content = "not a json"
    # Invalid schema
    mock_completion2 = Mock()
    mock_completion2.choices = [Mock()]
    mock_completion2.choices[0].message.content = '{"name": "John", "age": "nan"}'
    # Valid schema
    mock_completion3 = Mock()
    mock_completion3.choices = [Mock()]
    mock_completion3.choices[0].message.content = '{"name": "John", "age": 30}'

    class DummyModel(BaseModel):
        name: str
        age: int

    with pytest.raises(JSONSchemaValidationError):
        validate_json_completion(mock_completion1, DummyModel)
    with pytest.raises(JSONSchemaValidationError):
        validate_json_completion(mock_completion2, DummyModel)
    validate_json_completion(mock_completion3, DummyModel)


@pytest.mark.parametrize(
    "model_name", ["gpt-3.5-turbo", CILLMModelNames.ANTHROPIC.value]
)
@pytest.mark.asyncio
async def test_achat(model_name: str) -> None:
    model = LLMModel(name=model_name)
    response = await model.achat(
        messages=[
            Message(content="What are three things I should do today?"),
        ]
    )

    assert len(response.choices) == 1

    # Check we can iterate through the response
    async for chunk in await model.achat_iter(
        messages=[
            Message(content="What are three things I should do today?"),
        ]
    ):
        assert len(chunk.choices) == 1


@pytest.mark.parametrize(
    "model_name", [CILLMModelNames.OPENAI.value, CILLMModelNames.ANTHROPIC.value]
)
@pytest.mark.flaky(reruns=3, only_on=[litellm.exceptions.APIConnectionError])
@pytest.mark.asyncio
async def test_tools(dummy_env: DummyEnv, model_name: str) -> None:
    model = LLMModel(name=model_name)
    messages = [
        Message(content="What are three things I should do today?"),
    ]

    def get_todo_list(n: int) -> str:
        """Get todo list for today.

        Args:
            n: number of items to return
        """
        return "\n".join(["Go for a walk", "Read a book", "Call a friend"][:n])

    tool = Tool.from_function(get_todo_list)
    dummy_env.tools = [tool]
    result = await model.call(messages, tools=dummy_env.tools)
    assert result.completion_count > 0

    # try specifying tool choice
    result = await model.call(messages, tools=dummy_env.tools, tool_choice=tool)
    assert result.completion_count > 0, "Tool choice failed to execute tool"
    assert result.messages
    (tool_request_message,) = result.messages
    assert isinstance(tool_request_message, ToolRequestMessage)

    new_messages = await dummy_env.exec_tool_calls(tool_request_message)
    (new_message,) = new_messages
    assert new_message.content == "Go for a walk\nRead a book\nCall a friend"
    assert new_message.tool_call_id == tool_request_message.tool_calls[0].id

    def get_todo_list_no_args() -> str:
        """Get todo list for today."""
        return "\n".join(["Go for a walk", "Read a book", "Call a friend"])

    tool = Tool.from_function(get_todo_list_no_args)
    dummy_env.tools = [tool]
    result = await model.call(messages, tools=dummy_env.tools)
    assert result.completion_count > 0
    assert result.messages
    (tool_request_message,) = result.messages
    assert isinstance(tool_request_message, ToolRequestMessage)

    new_messages = await dummy_env.exec_tool_calls(tool_request_message)
    (new_message,) = new_messages
    assert new_message.content == "Go for a walk\nRead a book\nCall a friend"
    assert new_message.tool_call_id == tool_request_message.tool_calls[0].id

    # ok now try with multiple functions
    messages = [
        Message(
            content=(
                "What items will I have time to accomplish on my todo list today based"
                " on my calendar?"
            )
        ),
    ]

    def get_calendar() -> str:
        """Get text version of calendar for today."""
        return "9:00am Wake-up\n10:00pm Go to bed\n"

    tool2 = Tool.from_function(get_calendar)
    dummy_env.tools = [tool, tool2]
    result = await model.call(messages, tools=dummy_env.tools)
    assert result.messages
    (tool_request_message,) = result.messages
    assert isinstance(tool_request_message, ToolRequestMessage)
    new_messages = await dummy_env.exec_tool_calls(tool_request_message)
    if model_name.startswith("claude"):
        # Anthropic not always so smart
        assert 1 <= len(new_messages) <= 2
    else:
        assert len(new_messages) == 2

    # ok now try continuation - I AM NOT SURE IF THIS IS VALID?
    # TODO: - supported on openai, but not litellm
    # messages = messages + result.messages + new_messages
    # result = await model.call(messages)


class DummyOutputSchema(BaseModel):
    name: str
    age: int


class TestMultipleCompletionLLMModel:
    NUM_COMPLETIONS: ClassVar[int] = 2
    DEFAULT_CONFIG: ClassVar[dict] = {"n": NUM_COMPLETIONS}
    MODEL_CLS: ClassVar[type[MultipleCompletionLLMModel]] = MultipleCompletionLLMModel

    async def call_model(
        self, model: MultipleCompletionLLMModel, *args, **kwargs
    ) -> list[LLMResult]:
        return await model.call(*args, **kwargs)

    @pytest.mark.vcr
    @pytest.mark.parametrize("model_name", ["gpt-3.5-turbo"])
    @pytest.mark.asyncio
    async def test_model(self, model_name: str) -> None:
        # Make model_name an arg so that TestLLMModel can parametrize it
        # only testing OpenAI, as other APIs don't support n>1
        model = self.MODEL_CLS(name=model_name, config=self.DEFAULT_CONFIG)
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(content="Hello, how are you?"),
        ]
        results = await self.call_model(model, messages)
        assert len(results) == self.NUM_COMPLETIONS

        for result in results:
            assert result.prompt_count > 0
            assert result.completion_count > 0
            prompt_cost, completion_cost = result.prompt_and_completion_costs
            assert prompt_cost > 0
            assert completion_cost > 0
            assert result.logprob is None or result.logprob <= 0

    @pytest.mark.parametrize(
        "model_name", [CILLMModelNames.ANTHROPIC.value, "gpt-3.5-turbo"]
    )
    @pytest.mark.asyncio
    async def test_streaming(self, model_name: str) -> None:
        model = self.MODEL_CLS(name=model_name, config=self.DEFAULT_CONFIG)
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(content="Hello, how are you?"),
        ]

        def callback(_) -> None:
            return

        with pytest.raises(
            NotImplementedError,
            match="Multiple completions with callbacks is not supported",
        ):
            await self.call_model(model, messages, [callback])

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_parameterizing_tool_from_arg_union(self) -> None:
        def play(move: int | None) -> None:
            """Play one turn by choosing a move.

            Args:
                move: Choose an integer to lose, choose None to win.
            """

        results = await self.call_model(
            self.MODEL_CLS(name="gpt-3.5-turbo", config=self.DEFAULT_CONFIG),
            messages=[Message(content="Please win.")],
            tools=[Tool.from_function(play)],
        )
        assert len(results) == self.NUM_COMPLETIONS
        for result in results:
            assert result.messages
            assert len(result.messages) == 1
            assert isinstance(result.messages[0], ToolRequestMessage)
            assert result.messages[0].tool_calls
            assert result.messages[0].tool_calls[0].function.arguments["move"] is None

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_output_schema(self) -> None:
        model = self.MODEL_CLS(name="gpt-3.5-turbo", config=self.DEFAULT_CONFIG)
        messages = [
            Message(
                content=(
                    "My name is Claude and I am 1 year old. What is my name and age?"
                )
            ),
        ]
        results = await self.call_model(model, messages, output_type=DummyOutputSchema)
        assert len(results) == self.NUM_COMPLETIONS
        for result in results:
            assert result.messages
            assert len(result.messages) == 1
            assert result.messages[0].content
            DummyOutputSchema.model_validate_json(result.messages[0].content)

    @pytest.mark.parametrize("model_name", [CILLMModelNames.OPENAI.value])
    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_text_image_message(self, model_name: str) -> None:
        model = self.MODEL_CLS(name=model_name, config=self.DEFAULT_CONFIG)

        # An RGB image of a red square
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:] = [255, 0, 0]  # (255 red, 0 green, 0 blue) is maximum red in RGB

        results = await self.call_model(
            model,
            messages=[
                Message.create_message(
                    text="What color is this square? Respond only with the color name.",
                    image=image,
                )
            ],
        )
        assert len(results) == self.NUM_COMPLETIONS
        for result in results:
            assert result.messages is not None, (
                "Expected messages in result, but got None"
            )
            assert result.messages[-1].content is not None, (
                "Expected content in message, but got None"
            )
            assert "red" in result.messages[-1].content.lower()


class TestLLMModel(TestMultipleCompletionLLMModel):
    NUM_COMPLETIONS: ClassVar[int] = 1
    DEFAULT_CONFIG: ClassVar[dict] = {}
    MODEL_CLS: ClassVar[type[MultipleCompletionLLMModel]] = LLMModel

    async def call_model(self, model: LLMModel, *args, **kwargs) -> list[LLMResult]:  # type: ignore[override]
        return [await model.call(*args, **kwargs)]

    @pytest.mark.parametrize(
        "model_name", [CILLMModelNames.ANTHROPIC.value, "gpt-3.5-turbo"]
    )
    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_model(self, model_name: str) -> None:
        await super().test_model(model_name)

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        "model_name", [CILLMModelNames.ANTHROPIC.value, "gpt-3.5-turbo"]
    )
    @pytest.mark.asyncio
    async def test_streaming(self, model_name: str) -> None:
        model = LLMModel(name=model_name)
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(content="Hello, how are you?"),
        ]
        content = []

        def callback(s):
            content.append(s)

        result = await model.call(messages, [callback])
        assert result.completion_count > 0
        assert content

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_parameterizing_tool_from_arg_union(self) -> None:
        await super().test_parameterizing_tool_from_arg_union()

    @pytest.mark.asyncio
    async def test_output_type_rejected_validation(self) -> None:
        class InstructionList(BaseModel):
            instructions: list[str] = Field(description="list of instructions")

        model = self.MODEL_CLS(name=CILLMModelNames.ANTHROPIC.value)
        with pytest.raises(litellm.BadRequestError, match="anthropic"):
            await model.call(
                [Message(content="What are three things I should do today?")],
                output_type=InstructionList,
            )

    @pytest.mark.parametrize(
        "model_name",
        [CILLMModelNames.ANTHROPIC.value, "gpt-4-turbo", CILLMModelNames.OPENAI.value],
    )
    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_text_image_message(self, model_name: str) -> None:
        await super().test_text_image_message(model_name)
