import pathlib
import pickle
from enum import StrEnum
from typing import Any, ClassVar
from unittest.mock import Mock, patch

import litellm
import numpy as np
import pytest
from aviary.core import Message, Tool, ToolRequestMessage
from pydantic import BaseModel

from lcaw.exceptions import JSONSchemaValidationError
from lcaw.llms import (
    Chunk,
    LiteLLMModel,
    MultipleCompletionLLMModel,
    validate_json_completion,
)
from lcaw.types import LLMResult
from tests.conftest import VCR_DEFAULT_MATCH_ON


class TestLiteLLMModel:

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "model_name": "gpt-4o-mini",
                    "model_list": [
                        {
                            "model_name": "gpt-4o-mini",
                            "litellm_params": {
                                "model": "gpt-4o-mini",
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ],
                },
                id="chat-model",
            ),
            pytest.param(
                {
                    "model_name": "gpt-3.5-turbo-instruct",
                    "model_list": [
                        {
                            "model_name": "gpt-3.5-turbo-instruct",
                            "litellm_params": {
                                "model": "gpt-3.5-turbo-instruct",
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ],
                },
                id="completion-model",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_call(self, config: dict[str, Any]) -> None:
        llm = LiteLLMModel(name=config["model_name"], config=config)
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(role="user", content="What is the meaning of the universe?"),
        ]
        results = await llm.call(messages)
        assert isinstance(results.prompt, list)
        assert isinstance(results.prompt[1], Message)
        assert all(isinstance(msg, Message) for msg in results.prompt)
        assert len(results.prompt) == 2
        assert results.prompt[1].content
        assert results.text

    # @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.asyncio
    async def test_call_w_figure(self) -> None:
        llm = LiteLLMModel(name="gpt-4o")
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:] = [255, 0, 0]
        messages = [
            Message(
                role="system", content="You are a detective who investigate colors"
            ),
            Message.create_message(
                role="user",
                text="What color is this square? Show me your chain of reasoning.",
                image=image,
            ),
        ]  # TODO: It's not decoding the image. It's trying to guess the color from the encoded image string.
        results = await llm.call(messages)
        assert isinstance(results.prompt, list)
        assert all(isinstance(msg, Message) for msg in results.prompt)
        assert isinstance(results.prompt[1], Message)
        assert len(results.prompt) == 2
        assert results.prompt[1].content
        assert "red" in results.text.lower()
        assert results.seconds_to_last_token > 0
        assert results.prompt_count > 0
        assert results.completion_count > 0
        assert results.cost > 0

        # Also test with a callback
        async def ac(x) -> None:
            pass

        results = await llm.call(messages, [ac])
        assert isinstance(results.prompt, list)
        assert all(isinstance(msg, Message) for msg in results.prompt)
        assert isinstance(results.prompt[1], Message)
        assert len(results.prompt) == 2
        assert results.prompt[1].content
        assert "red" in results.text.lower()
        assert results.seconds_to_first_token > 0
        assert results.prompt_count > 0
        assert results.completion_count > 0
        assert results.cost > 0

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "model_list": [
                        {
                            "model_name": "gpt-4o-mini",
                            "litellm_params": {
                                "model": "gpt-4o-mini",
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ]
                },
                id="with-router",
            ),
            pytest.param(
                {
                    "pass_through_router": True,
                    "router_kwargs": {"temperature": 0, "max_tokens": 56},
                },
                id="without-router",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_run_prompt(self, config: dict[str, Any]) -> None:
        llm = LiteLLMModel(name="gpt-4o-mini", config=config)

        outputs = []

        def accum(x) -> None:
            outputs.append(x)

        completion = await llm.run_prompt(
            prompt="The {animal} says",
            data={"animal": "duck"},
            system_prompt=None,
            callbacks=[accum],
        )
        assert completion.model == "gpt-4o-mini"
        assert completion.seconds_to_first_token > 0
        assert completion.prompt_count > 0
        assert completion.completion_count > 0
        assert str(completion) == "".join(outputs)
        assert completion.cost > 0

        completion = await llm.run_prompt(
            prompt="The {animal} says",
            data={"animal": "duck"},
            system_prompt=None,
        )
        assert completion.seconds_to_first_token == 0
        assert completion.seconds_to_last_token > 0
        assert completion.cost > 0

        # check with mixed callbacks
        async def ac(x) -> None:
            pass

        completion = await llm.run_prompt(
            prompt="The {animal} says",
            data={"animal": "duck"},
            system_prompt=None,
            callbacks=[accum, ac],
        )
        assert completion.cost > 0

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        ("config", "bypassed_router"),
        [
            pytest.param(
                {
                    "model_list": [
                        {
                            "model_name": "gpt-4o-mini",
                            "litellm_params": {"model": "gpt-4o-mini", "max_tokens": 3},
                        }
                    ]
                },
                False,
                id="with-router",
            ),
            pytest.param(
                {"pass_through_router": True, "router_kwargs": {"max_tokens": 3}},
                True,
                id="without-router",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_max_token_truncation(
        self, config: dict[str, Any], bypassed_router: bool
    ) -> None:
        llm = LiteLLMModel(name="gpt-4o-mini", config=config)
        with patch(
            "litellm.Router.atext_completion",
            side_effect=litellm.Router.atext_completion,
            autospec=True,
        ) as mock_atext_completion:
            chunk = await llm.acomplete("Please tell me a story")  # type: ignore[call-arg]
        if bypassed_router:
            mock_atext_completion.assert_not_awaited()
        else:
            mock_atext_completion.assert_awaited_once()
        assert isinstance(chunk, Chunk)
        assert chunk.completion_tokens == 3
        assert chunk.text
        assert len(chunk.text) < 20

    def test_pickling(self, tmp_path: pathlib.Path) -> None:
        pickle_path = tmp_path / "llm_model.pickle"
        llm = LiteLLMModel(
            name="gpt-4o-mini",
            config={
                "model_list": [
                    {
                        "model_name": "gpt-4o-mini",
                        "litellm_params": {
                            "model": "gpt-4o-mini",
                            "temperature": 0,
                            "max_tokens": 56,
                        },
                    }
                ]
            },
        )
        with pickle_path.open("wb") as f:
            pickle.dump(llm, f)
        with pickle_path.open("rb") as f:
            rehydrated_llm = pickle.load(f)
        assert llm.name == rehydrated_llm.name
        assert llm.config == rehydrated_llm.config
        assert llm.router.deployment_names == rehydrated_llm.router.deployment_names


class CILLMModelNames(StrEnum):
    """Models to use for generic CI testing."""

    ANTHROPIC = "claude-3-haiku-20240307"  # Cheap and not Anthropic's cutting edge
    OPENAI = "gpt-4o-mini-2024-07-18"  # Cheap and not OpenAI's cutting edge


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

    @pytest.mark.parametrize(
        "model_name", ["gpt-3.5-turbo", CILLMModelNames.ANTHROPIC.value]
    )
    @pytest.mark.asyncio
    async def test_achat(self, model_name: str) -> None:
        model = MultipleCompletionLLMModel(name=model_name)
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

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
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
            assert result.cost > 0
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
    @pytest.mark.parametrize(
        ("model_name", "output_type"),
        [
            pytest.param("gpt-3.5-turbo", DummyOutputSchema, id="json-mode"),
            pytest.param(
                "gpt-4o", DummyOutputSchema.model_json_schema(), id="structured-outputs"
            ),
        ],
    )
    async def test_output_schema(
        self, model_name: str, output_type: type[BaseModel] | dict[str, Any]
    ) -> None:
        model = self.MODEL_CLS(name=model_name, config=self.DEFAULT_CONFIG)
        messages = [
            Message(
                content=(
                    "My name is Claude and I am 1 year old. What is my name and age?"
                )
            ),
        ]
        results = await self.call_model(model, messages, output_type=output_type)
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
            assert (
                result.messages is not None
            ), "Expected messages in result, but got None"
            assert (
                result.messages[-1].content is not None
            ), "Expected content in message, but got None"
            assert "red" in result.messages[-1].content.lower()

    @pytest.mark.parametrize(
        "model_name", [CILLMModelNames.ANTHROPIC.value, "gpt-3.5-turbo"]
    )
    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_single_completion(self, model_name: str) -> None:
        model = self.MODEL_CLS(name=model_name, config={"n": 1})
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(content="Hello, how are you?"),
        ]
        result = await model.call_single(messages)
        assert isinstance(result, LLMResult)

        assert isinstance(result, LLMResult)
        assert result.messages
        assert len(result.messages) == 1
        assert result.messages[0].content

        model = self.MODEL_CLS(name=model_name, config={"n": 2})
        result = await model.call_single(messages)
        assert isinstance(result, LLMResult)
        assert result.messages
        assert len(result.messages) == 1
        assert result.messages[0].content

    @pytest.mark.asyncio
    @pytest.mark.vcr
    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param(CILLMModelNames.ANTHROPIC.value, id="anthropic"),
            pytest.param(CILLMModelNames.OPENAI.value, id="openai"),
        ],
    )
    async def test_multiple_completion(self, model_name: str, request) -> None:
        model = self.MODEL_CLS(name=model_name, config={"n": self.NUM_COMPLETIONS})
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(content="Hello, how are you?"),
        ]
        if request.node.callspec.id == "anthropic":
            # Anthropic does not support multiple completions
            with pytest.raises(litellm.BadRequestError, match="anthropic"):
                await model.call(messages)
        else:
            results = await model.call(messages)  # noqa: FURB120
            assert len(results) == self.NUM_COMPLETIONS

            model = self.MODEL_CLS(name=model_name, config={"n": 5})
            results = await model.call(messages, n=self.NUM_COMPLETIONS)
            assert len(results) == self.NUM_COMPLETIONS


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
