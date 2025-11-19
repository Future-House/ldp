import pathlib
import pickle
from collections.abc import AsyncIterator
from typing import Any, ClassVar
from unittest.mock import AsyncMock, Mock, patch

import litellm
import numpy as np
import pytest
from aviary.core import Message, Tool, ToolRequestMessage, ToolResponseMessage
from pydantic import BaseModel, Field, TypeAdapter, computed_field

from lmi.exceptions import JSONSchemaValidationError
from lmi.llms import (
    CommonLLMNames,
    LiteLLMModel,
    validate_json_completion,
)
from lmi.types import LLMResult
from lmi.utils import VCR_DEFAULT_MATCH_ON


class TestLiteLLMModel:
    def test_instantiation_methods(self) -> None:
        # Test default instantiation
        model1 = LiteLLMModel()
        assert model1.name == CommonLLMNames.GPT_4O.value
        assert model1.provider == "openai"
        assert isinstance(model1.config, dict)
        assert "model_list" in model1.config

        # Test name-only instantiation
        name = CommonLLMNames.ANTHROPIC_TEST.value
        model2 = LiteLLMModel(name=name)
        assert model2.name == name
        assert model2.provider == "anthropic"
        assert model2.config["model_list"][0]["model_name"] == name

        # Test config-only instantiation
        config = {
            "name": CommonLLMNames.OPENAI_TEST.value,
            "temperature": 0,
            "max_tokens": 56,
        }
        model3 = LiteLLMModel(config=config)
        assert model3.name == CommonLLMNames.OPENAI_TEST.value
        assert model3.provider == "openai"
        assert (
            model3.config["model_list"][0]["model_name"]
            == CommonLLMNames.OPENAI_TEST.value
        )
        assert model3.config["model_list"][0]["litellm_params"]["temperature"] == 0
        assert model3.config["model_list"][0]["litellm_params"]["max_tokens"] == 56

        # Test name and config instantiation
        name = CommonLLMNames.OPENAI_TEST.value
        config = {"temperature": 0.5, "max_tokens": 100}
        model4 = LiteLLMModel(name=name, config=config)
        assert model4.name == name
        assert model4.provider == "openai"
        assert model4.config["model_list"][0]["model_name"] == name
        assert model4.config["model_list"][0]["litellm_params"]["temperature"] == 0.5
        assert model4.config["model_list"][0]["litellm_params"]["max_tokens"] == 100

        # Test logprobs and top_logprobs configuration passing through (OpenAI-specific)
        name = CommonLLMNames.OPENAI_TEST.value
        config_with_logprobs = {
            "logprobs": True,
            "top_logprobs": 20,
            "temperature": 0.7,
        }
        model5 = LiteLLMModel(name=name, config=config_with_logprobs)
        assert model5.config["model_list"][0]["litellm_params"]["logprobs"] is True
        assert model5.config["model_list"][0]["litellm_params"]["top_logprobs"] == 20
        assert model5.config["model_list"][0]["litellm_params"]["temperature"] == 0.7

        model6 = LiteLLMModel(name="definitely/not-a-provider")
        assert (
            model6.name
            == model6.config["model_list"][0]["model_name"]
            == "definitely/not-a-provider"
        )
        with pytest.raises(litellm.BadRequestError, match="definitely"):
            _ = model6.provider

        model7 = LiteLLMModel(
            name="nvidia/nemotron-parse",
            config={"api_base": "https://integrate.api.nvidia.com/v1"},
        )
        assert (
            model7.name
            == model7.config["model_list"][0]["model_name"]
            == "nvidia/nemotron-parse"
        )
        assert model7.provider == "nvidia_nim"

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "name": CommonLLMNames.OPENAI_TEST.value,
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.OPENAI_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.OPENAI_TEST.value,
                                "temperature": 0,
                                "max_tokens": 56,
                                "logprobs": True,
                                "top_logprobs": 5,
                            },
                        }
                    ],
                },
                id="OpenAI-model-1",
            ),
            pytest.param(
                {"name": CommonLLMNames.GPT_5_MINI.value}, id="OpenAI-model-2"
            ),
            pytest.param(
                {
                    "name": CommonLLMNames.ANTHROPIC_TEST.value,
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.ANTHROPIC_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.ANTHROPIC_TEST.value,
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ],
                },
                id="Anthropic-model",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_call(self, config: dict[str, Any]) -> None:
        llm = LiteLLMModel(config=config)
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(role="user", content="What is the meaning of the universe?"),
        ]
        result_name = f"test-{llm.name}"
        results = await llm.call(messages, name=result_name)
        assert isinstance(results, list)

        result = results[0]
        assert isinstance(result, LLMResult)
        assert isinstance(result.prompt, list)
        assert isinstance(result.prompt[1], Message)
        assert all(isinstance(msg, Message) for msg in result.prompt)
        assert len(result.prompt) == 2  # role + user messages
        assert result.prompt[1].content
        assert result.text
        if llm.config["model_list"][0]["litellm_params"].get("logprobs"):
            assert isinstance(result.logprob, float)
            assert result.logprob <= 0
            # Test top_logprobs only for OpenAI models (top_logprobs is OpenAI-specific)
            if llm.config["model_list"][0]["litellm_params"].get("top_logprobs"):
                assert isinstance(result.top_logprobs, list)
                assert len(result.top_logprobs) > 0
                # Each position should have a list of (token, logprob) tuples
                for position_logprobs in result.top_logprobs:
                    assert isinstance(position_logprobs, list)
                    for token, logprob in position_logprobs:
                        assert isinstance(token, str)
                        assert isinstance(logprob, float)
            else:
                # For non-OpenAI models or when top_logprobs not configured
                assert result.top_logprobs is None
        else:
            assert result.logprob is None
            assert result.top_logprobs is None
        assert result.name == result_name
        result = await llm.call_single(messages)
        assert isinstance(result, LLMResult)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "name": CommonLLMNames.OPENAI_TEST.value,
                    "fallbacks": [
                        {
                            CommonLLMNames.OPENAI_TEST.value: [
                                CommonLLMNames.ANTHROPIC_TEST.value
                            ]
                        }
                    ],
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.OPENAI_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.OPENAI_TEST.value,
                                "api_key": "invalid_key_to_force_fallback",  # pragma: allowlist secret
                            },
                        },
                        {
                            "model_name": CommonLLMNames.ANTHROPIC_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.ANTHROPIC_TEST.value,
                                "max_tokens": 56,
                            },
                        },
                    ],
                },
                id="multiple-models",
            ),
        ],
    )
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    async def test_call_w_multiple_models(self, config: dict[str, Any]) -> None:
        llm = LiteLLMModel(config=config)
        messages = [
            Message(role="system", content="Respond with single words."),
            Message(role="user", content="What is the meaning of the universe?"),
        ]
        results = await llm.call(messages)
        assert isinstance(results, list)
        assert isinstance(results[0], LLMResult)
        assert results[0].model == CommonLLMNames.ANTHROPIC_TEST.value, (
            "Expected Anthropic to be used as fallback"
        )

        llm.name = CommonLLMNames.ANTHROPIC_TEST.value
        outputs: list[str] = []
        results = await llm.call(messages, callbacks=[outputs.append])
        assert isinstance(results, list)
        assert isinstance(results[0], LLMResult)
        assert results[0].model == CommonLLMNames.ANTHROPIC_TEST.value, (
            "Expected Anthropic to be used after changing llm.name"
        )

    # @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON])
    @pytest.mark.asyncio
    async def test_call_w_figure(self) -> None:
        llm = LiteLLMModel(name=CommonLLMNames.GPT_4O.value)
        sys_message = Message(
            role="system", content="You are a detective who investigates colors"
        )

        # First let's get baseline prompt tokens and cost
        (result,) = await llm.call([
            sys_message,
            Message(
                content=(
                    "What color is this square? Show me your chain of reasoning."
                    " Alternately, if there is no square, just answer 'no square'."
                )
            ),
        ])
        assert isinstance(result, LLMResult)
        assert result.prompt_count is not None
        assert result.prompt_count > 0
        assert result.cost > 0
        assert (result.text or "").strip().rstrip(".").lower() == "no square"
        no_image_prompt_count = result.prompt_count
        no_image_completion_count = result.completion_count
        no_image_cost = result.cost

        # Now let's prompt an image and confirm its used and incorporated into cost
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:] = [255, 0, 0]
        messages = [
            sys_message,
            Message.create_message(
                text=(
                    "What color is this square? Show me your chain of reasoning."
                    " Alternately, if there is no square, just answer 'no square'."
                ),
                images=image,
            ),
        ]
        (result,) = await llm.call(messages)
        assert isinstance(result, LLMResult)
        assert isinstance(result.prompt, list)
        assert all(isinstance(msg, Message) for msg in result.prompt)
        assert isinstance(result.prompt[1], Message)
        assert len(result.prompt) == 2
        assert result.prompt[1].content
        assert isinstance(result.text, str)
        assert "red" in result.text.lower()
        assert result.seconds_to_last_token > 0
        assert (  # noqa: PT018
            result.prompt_count is not None
            and result.prompt_count > 1.25 * no_image_prompt_count
        ), "Image usage should require more prompt tokens"
        assert result.completion_count is not None
        assert result.completion_count > 0
        assert result.cost > 1.25 * no_image_cost, (
            f"Image usage should require higher cost. For reference,"
            f" {result.prompt_count=}, {result.completion_count=},"
            f" {result.cost=}, {no_image_prompt_count=},"
            f" {no_image_completion_count=}, and {no_image_cost=}."
        )

        # Also test with a callback
        async def ac(x) -> None:
            pass

        (result,) = await llm.call(messages, [ac])
        assert isinstance(result, LLMResult)
        assert isinstance(result.prompt, list)
        assert all(isinstance(msg, Message) for msg in result.prompt)
        assert isinstance(result.prompt[1], Message)
        assert len(result.prompt) == 2
        assert result.prompt[1].content
        assert isinstance(result.text, str)
        assert "red" in result.text.lower()
        assert result.seconds_to_last_token > 0
        assert result.prompt_count is not None
        assert result.prompt_count > 0
        assert result.completion_count is not None
        assert result.completion_count > 0
        assert result.cost > 0

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "name": CommonLLMNames.OPENAI_TEST.value,
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.OPENAI_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.OPENAI_TEST.value,
                                "temperature": 0,
                                "max_tokens": 56,
                                "logprobs": True,
                            },
                        }
                    ],
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
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.asyncio
    async def test_call_single(self, config: dict[str, Any], subtests) -> None:
        llm = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value, config=config)

        outputs = []

        def accum(x) -> None:
            outputs.append(x)

        prompt = "The {animal} says"
        data = {"animal": "duck"}
        system_prompt = "You are a helpful assistant."
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt.format(**data)),
        ]

        completion = await llm.call_single(
            messages=messages,
            callbacks=[accum],
        )
        assert completion.model == CommonLLMNames.OPENAI_TEST.value
        assert completion.seconds_to_last_token > 0
        assert completion.prompt_count is not None
        assert completion.prompt_count > 0
        assert completion.completion_count is not None
        assert completion.completion_count > 0
        assert str(completion) == "".join(outputs)
        assert completion.cost > 0

        completion = await llm.call_single(
            messages=messages,
        )
        assert completion.seconds_to_last_token > 0
        assert completion.cost > 0

        # check with mixed callbacks
        async def ac(x) -> None:
            pass

        completion = await llm.call_single(
            messages=messages,
            callbacks=[accum, ac],
        )
        assert completion.cost > 0

        with subtests.test(msg="passing-kwargs"):
            completion = await llm.call_single(
                messages=[Message(role="user", content="Tell me a very long story")],
                max_tokens=1000,
            )
            assert completion.cost > 0
            assert (  # noqa: PT018
                completion.completion_count is not None
                and completion.completion_count > 100
            ), "Expected a long completion"

        with subtests.test(msg="autowraps message"):

            def mock_call(messages, *_, **__):
                assert isinstance(messages, list)
                return [None]

            with patch.object(LiteLLMModel, "call", side_effect=mock_call):
                await llm.call_single("Test message")

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        ("config", "bypassed_router"),
        [
            pytest.param(
                {
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.OPENAI_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.OPENAI_TEST.value,
                                "max_tokens": 3,
                            },
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
        llm = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value, config=config)
        with patch(
            "litellm.Router.acompletion",
            side_effect=litellm.Router.acompletion,
            autospec=True,
        ) as mock_completion:
            completions = await llm.acompletion([
                Message(content="Please tell me a story")
            ])
        if bypassed_router:
            mock_completion.assert_not_awaited()
        else:
            mock_completion.assert_awaited_once()
        assert isinstance(completions, list)
        completion = completions[0]
        assert completion.completion_count == 3
        assert completion.text
        assert len(completion.text) < 20

    def test_pickling(self, tmp_path: pathlib.Path) -> None:
        pickle_path = tmp_path / "llm_model.pickle"
        llm = LiteLLMModel(
            name=CommonLLMNames.OPENAI_TEST.value,
            config={
                "model_list": [
                    {
                        "model_name": CommonLLMNames.OPENAI_TEST.value,
                        "litellm_params": {
                            "model": CommonLLMNames.OPENAI_TEST.value,
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
        assert (
            llm.get_router().deployment_names
            == rehydrated_llm.get_router().deployment_names
        )

    @pytest.mark.asyncio
    async def test_acompletion_iter_logprobs_edge_cases(self) -> None:
        """Test that acompletion_iter handles various logprobs edge cases gracefully."""
        model = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value)
        messages = [Message(content="Say hello")]

        def _build_mock_completion(
            model: str = "test-model",
            logprobs: Any = None,
            delta_content: str = "",
            delta_reasoning_content: str = "hmmm",
            delta_role: str = "assistant",
            usage: Any = None,
        ) -> Mock:
            return Mock(
                model=model,
                choices=[
                    Mock(
                        logprobs=logprobs,
                        delta=Mock(
                            content=delta_content,
                            reasoning_content=delta_reasoning_content,
                            role=delta_role,
                        ),
                    )
                ],
                usage=usage,
            )

        # Mock the router to return different logprobs scenarios
        with patch.object(model, "_router") as mock_router:
            # Mock completion with None logprobs
            mock_completion_none = _build_mock_completion(delta_content="Hello")

            # Mock completion with logprobs but no content
            mock_completion_no_content = _build_mock_completion(
                logprobs=Mock(content=None),
                delta_content=" world",
            )

            # Mock completion with empty content list
            mock_completion_empty = _build_mock_completion(
                logprobs=Mock(content=[]), delta_content="!"
            )

            # Mock completion with valid logprobs
            mock_completion_valid = _build_mock_completion(
                logprobs=Mock(content=[Mock(logprob=-0.5)])
            )

            # Mock completion with usage info
            mock_completion_usage = _build_mock_completion(
                usage=Mock(prompt_tokens=10, completion_tokens=5)
            )

            # Create async generator that yields mock completions
            async def mock_stream():  # noqa: RUF029
                async def mock_stream_iter():  # noqa: RUF029
                    yield mock_completion_none
                    yield mock_completion_no_content
                    yield mock_completion_empty
                    yield mock_completion_valid
                    yield mock_completion_usage

                return mock_stream_iter()

            mock_router.acompletion.return_value = mock_stream()

            # Test that the method doesn't raise exceptions
            async_iterable = await model.acompletion_iter(messages)
            results = [result async for result in async_iterable]

            # Verify we got one final result
            assert len(results) == 1
            result = results[0]
            assert isinstance(result, LLMResult)
            assert result.text == "Hello world!"
            assert result.model == "test-model"
            assert result.logprob == -0.5
            assert result.prompt_count == 10
            assert result.completion_count == 5


class DummyOutputSchema(BaseModel):
    name: str
    age: int = Field(description="Age in years.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def name_and_age(self) -> str:  # So we can test computed_field is not included
        return f"{self.name}, {self.age}"


class TestMultipleCompletion:
    NUM_COMPLETIONS: ClassVar[int] = 2
    DEFAULT_CONFIG: ClassVar[dict] = {"n": NUM_COMPLETIONS}
    MODEL_CLS: ClassVar[type[LiteLLMModel]] = LiteLLMModel

    async def call_model(self, model: LiteLLMModel, *args, **kwargs) -> list[LLMResult]:
        return await model.call(*args, **kwargs)

    @pytest.mark.parametrize(
        "model_name",
        [CommonLLMNames.GPT_35_TURBO.value, CommonLLMNames.ANTHROPIC_TEST.value],
    )
    @pytest.mark.asyncio
    async def test_acompletion(self, model_name: str) -> None:
        model = self.MODEL_CLS(name=model_name)
        messages = [
            Message(content="What are three things I should do today?"),
        ]
        response = await model.acompletion(messages)

        assert isinstance(response, list)
        assert len(response) == 1
        assert isinstance(response[0], LLMResult)

    @pytest.mark.parametrize(
        "model_name",
        [CommonLLMNames.OPENAI_TEST.value, CommonLLMNames.ANTHROPIC_TEST.value],
    )
    @pytest.mark.asyncio
    async def test_acompletion_iter(self, model_name: str) -> None:
        model = self.MODEL_CLS(name=model_name)
        messages = [Message(content="What are three things I should do today?")]
        responses = await model.acompletion_iter(messages)
        assert isinstance(responses, AsyncIterator)

        async for response in responses:
            assert isinstance(response, LLMResult)
            assert isinstance(response.prompt, list)

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize("model_name", [CommonLLMNames.GPT_35_TURBO.value])
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
            assert result.prompt_count is not None
            assert result.prompt_count > 0
            assert result.completion_count is not None
            assert result.completion_count > 0
            assert result.cost > 0
        if model.config["model_list"][0]["litellm_params"].get("logprobs"):
            assert isinstance(result.logprob, float)
            assert result.logprob <= 0
        else:
            assert result.logprob is None

    @pytest.mark.parametrize(
        "model_name",
        [CommonLLMNames.ANTHROPIC_TEST.value, CommonLLMNames.GPT_35_TURBO.value],
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
            self.MODEL_CLS(
                name=CommonLLMNames.GPT_35_TURBO.value, config=self.DEFAULT_CONFIG
            ),
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
            pytest.param(
                CommonLLMNames.GPT_35_TURBO.value,
                DummyOutputSchema,
                id="json-mode-base-model",
            ),
            pytest.param(
                CommonLLMNames.GPT_4O.value,
                TypeAdapter(DummyOutputSchema),
                id="json-mode-type-adapter",
            ),
            pytest.param(
                CommonLLMNames.GPT_4O.value,
                DummyOutputSchema.model_json_schema(),
                id="structured-outputs",
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

    @pytest.mark.parametrize("model_name", [CommonLLMNames.OPENAI_TEST.value])
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
                    images=image,
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

    @pytest.mark.parametrize(
        "model_name",
        [CommonLLMNames.ANTHROPIC_TEST.value, CommonLLMNames.GPT_35_TURBO.value],
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
        assert result.messages
        assert len(result.messages) == 1
        assert result.messages[0].content
        assert not hasattr(result.messages[0], "tool_calls"), "Expected normal message"

        model = self.MODEL_CLS(name=model_name, config={"n": 2})
        result = await model.call_single(messages)
        assert isinstance(result, LLMResult)
        assert result.messages
        assert len(result.messages) == 1
        assert result.messages[0].content
        assert not hasattr(result.messages[0], "tool_calls"), "Expected normal message"

    @pytest.mark.asyncio
    @pytest.mark.vcr
    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param(CommonLLMNames.ANTHROPIC_TEST.value, id="anthropic"),
            pytest.param(CommonLLMNames.OPENAI_TEST.value, id="openai"),
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


class TestTooling:
    @pytest.mark.asyncio
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    async def test_tool_selection(self) -> None:
        model = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value, config={"n": 1})

        def double(x: int) -> int:
            """Double the input.

            Args:
                x: The input to double
            Returns:
                The double of the input.
            """
            return 2 * x

        tools = [Tool.from_function(double)]
        messages = [
            Message(
                role="system",
                content=(
                    "You are a helpful assistant who can use tools to do math. Use a"
                    " tool if needed. If you don't need a tool, just respond with the"
                    " answer."
                ),
            ),
            Message(role="user", content="What is double of 8?"),
        ]

        results = await model.call(
            messages, tools=tools, tool_choice=LiteLLMModel.MODEL_CHOOSES_TOOL
        )
        assert isinstance(results, list)
        assert isinstance(results[0].messages, list)

        tool_message = results[0].messages[0]

        assert isinstance(tool_message, ToolRequestMessage), (
            "It should have selected a tool"
        )
        assert not tool_message.content
        assert tool_message.tool_calls[0].function.arguments["x"] == 8, (
            "LLM failed in select the correct tool or arguments"
        )

        # Simulate the observation
        observation = ToolResponseMessage(
            role="tool",
            name="double",
            content="Observation: 16",
            tool_call_id=tool_message.tool_calls[0].id,
        )
        messages.extend([tool_message, observation])

        results = await model.call(
            messages, tools=tools, tool_choice=LiteLLMModel.MODEL_CHOOSES_TOOL
        )
        assert isinstance(results, list)
        assert isinstance(results[0].messages, list)
        assert results[0].messages[0].content
        assert "16" in results[0].messages[0].content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tools", "model_name"),
        [
            pytest.param([], CommonLLMNames.OPENAI_TEST.value, id="OpenAI-empty-tools"),
            pytest.param(None, CommonLLMNames.OPENAI_TEST.value, id="OpenAI-no-tools"),
            pytest.param(
                [], CommonLLMNames.ANTHROPIC_TEST.value, id="Anthropic-empty-tools"
            ),
            pytest.param(
                None, CommonLLMNames.ANTHROPIC_TEST.value, id="Anthropic-no-tools"
            ),
        ],
    )
    @pytest.mark.vcr
    async def test_empty_tools(self, tools: list | None, model_name: str) -> None:
        model = LiteLLMModel(name=model_name, config={"n": 1, "max_tokens": 56})

        result = await model.call_single(
            messages=[Message(content="What does 42 mean?")],
            tools=tools,
            tool_choice=LiteLLMModel.MODEL_CHOOSES_TOOL,
        )

        assert isinstance(result.messages, list)
        if tools is None:
            assert isinstance(result.messages[0], Message)
        else:
            assert isinstance(result.messages[0], ToolRequestMessage)
            assert not result.messages[0].tool_calls

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_multi_response_validation(self) -> None:
        model = LiteLLMModel(name="text-completion-openai/babbage-002")
        with pytest.raises(ValueError, match="2 results"):
            # Confirming https://github.com/BerriAI/litellm/issues/12298
            # does not silently pass through LMI
            await model.call_single(
                messages=[
                    Message(role="system", content="Answer in a concise tone."),
                    Message(content="What is your name?"),
                ]
            )


class TestReasoning:
    @pytest.mark.parametrize(
        "llm_name",
        [
            pytest.param(
                "deepseek/deepseek-reasoner",
                id="deepseek-reasoner",
            ),
            pytest.param(
                "openrouter/deepseek/deepseek-r1",
                id="openrouter-deepseek",
            ),
        ],
    )
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.asyncio
    async def test_deepseek_model(self, llm_name: str) -> None:
        llm = LiteLLMModel(name=llm_name)
        messages = [
            Message(
                role="system",
                content="Think deeply about the following question and answer it.",
            ),
            Message(content="What is the meaning of life?"),
        ]
        results = await llm.call(messages)
        for result in results:
            assert result.reasoning_content

        outputs: list[str] = []
        results = await llm.call(messages, callbacks=[outputs.append])

        for i, result in enumerate(results):
            assert result.reasoning_content
            assert outputs[i] == result.text

    @pytest.mark.vcr
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model", [CommonLLMNames.GPT_5, CommonLLMNames.CLAUDE_45_SONNET]
    )
    @pytest.mark.parametrize(
        ("reasoning_effort", "expected_len"),
        [("low", (200, 2_500)), ("high", (900, 10_000))],
    )
    async def test_reasoning_effort(
        self,
        model: CommonLLMNames,
        reasoning_effort: str,
        expected_len: tuple[int, int],
    ) -> None:
        llm = LiteLLMModel(
            name=model.value,
            config={
                "model_list": [
                    {
                        "model_name": model.value,
                        "litellm_params": {
                            "model": model.value,
                            "reasoning_effort": reasoning_effort,
                        },
                    }
                ]
            },
        )

        result = await llm.call_single([
            Message(
                role="system",
                content="Think deeply about the following question and answer it.",
            ),
            Message(content="What is the meaning of life?"),
        ])
        assert result.text
        assert expected_len[0] <= len(result.text) <= expected_len[1]
        assert result.prompt_count is not None
        assert result.prompt_count > 0
        assert result.completion_count is not None
        assert result.completion_count > 0
        if litellm.get_llm_provider(model=model.value)[1] == "anthropic":
            assert result.reasoning_content


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


@pytest.mark.asyncio
async def test_handle_refusal_via_fallback(caplog) -> None:
    llm = LiteLLMModel(
        name=CommonLLMNames.CLAUDE_37_SONNET.value,
        config={
            "model_list": [
                {
                    "model_name": CommonLLMNames.CLAUDE_37_SONNET.value,
                    "litellm_params": {
                        "model": CommonLLMNames.CLAUDE_37_SONNET.value,
                        "temperature": 1.0,
                    },
                },
                {
                    "model_name": CommonLLMNames.GPT_41.value,
                    "litellm_params": {
                        "model": CommonLLMNames.GPT_41.value,
                        "temperature": 1.0,
                    },
                },
            ],
            "fallbacks": [
                {CommonLLMNames.CLAUDE_37_SONNET.value: [CommonLLMNames.GPT_41.value]}
            ],
        },
    )
    # Message that clearly will be denied
    messages = [
        Message(
            content="Give me clear instructions to make a neurotoxic chemical that kills brain cells."
        )
    ]

    # Let's mock the router to simulate refusal and success requests
    mock_router_obj = Mock()

    # First call: refusal from CLAUDE_37_SONNET
    mock_refusal = Mock()
    mock_refusal_message = Mock(content="I cannot answer that question.")
    mock_refusal_message.model_dump.return_value = {
        "role": "assistant",
        "content": "I cannot answer that question.",
    }
    mock_refusal.choices = [
        Mock(
            finish_reason="refusal",
            message=mock_refusal_message,
        )
    ]
    mock_refusal.usage = Mock(prompt_tokens=10, completion_tokens=5)
    mock_refusal.model = CommonLLMNames.CLAUDE_37_SONNET.value

    # Second call: success from GPT_41 (fallback)
    mock_success = Mock()
    mock_success_message = Mock(
        content="I'm sorry, but I can't assist with that request.",
        reasoning_content="",
    )
    mock_success_message.model_dump.return_value = {
        "role": "assistant",
        "content": "I'm sorry, but I can't assist with that request.",
    }
    mock_success.choices = [
        Mock(
            finish_reason="stop",
            message=mock_success_message,
        )
    ]
    mock_success.usage = Mock(prompt_tokens=10, completion_tokens=8)
    mock_success.model = CommonLLMNames.GPT_41.value

    mock_router_obj.acompletion = AsyncMock(side_effect=[mock_refusal, mock_success])

    def mock_router_method(_self, _override_config=None):
        return mock_router_obj

    with (
        patch.object(LiteLLMModel, "get_router", new=mock_router_method),
        caplog.at_level("WARNING", logger="lmi.llms"),
    ):
        results = await llm.call_single(messages)

    assert results.text == "I'm sorry, but I can't assist with that request."
    assert results.model == CommonLLMNames.GPT_41.value
    assert "the llm request was refused" in caplog.text.lower()
    assert "attempting to fallback" in caplog.text.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "expected_tool_role_count"),
    [
        ("vertex_ai/gemini-3-pro-preview", 0),
        (CommonLLMNames.GPT_4O.value, 1),
    ],
)
async def test_gemini3_tool_patch(
    model_name: str, expected_tool_role_count: int
) -> None:
    # Setup
    model = LiteLLMModel(name=model_name)

    messages = [
        Message(role="user", content="Hello"),
        ToolResponseMessage(
            role="tool", content="Result", tool_call_id="123", name="test_tool"
        ),
    ]

    # Execute
    with patch.object(
        LiteLLMModel, "acompletion", new_callable=AsyncMock
    ) as mock_acompletion:
        mock_acompletion.return_value = []
        await model.call(messages)

        # Verify
        call_args = mock_acompletion.call_args
        assert call_args is not None
        # call_args[0] are positional args: (messages,)
        called_messages = call_args[0][0]

        assert len(called_messages) == 2
        assert called_messages[0].role == "user"

        tool_messages = [m for m in called_messages if m.role == "tool"]
        assert len(tool_messages) == expected_tool_role_count
