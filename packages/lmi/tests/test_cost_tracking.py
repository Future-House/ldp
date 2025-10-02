from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from aviary.core import Message

from lmi import cost_tracking_ctx
from lmi.cost_tracker import GLOBAL_COST_TRACKER, TrackedStreamWrapper
from lmi.embeddings import LiteLLMEmbeddingModel
from lmi.llms import CommonLLMNames, LiteLLMModel
from lmi.utils import VCR_DEFAULT_MATCH_ON


@contextmanager
def assert_costs_increased():
    """All tests in this file should increase accumulated costs."""
    initial_cost = GLOBAL_COST_TRACKER.lifetime_cost_usd
    yield
    assert GLOBAL_COST_TRACKER.lifetime_cost_usd > initial_cost


class TestLiteLLMEmbeddingCosts:
    @pytest.mark.asyncio
    async def test_embed_documents(self):
        stub_texts = ["test1", "test2"]
        with assert_costs_increased(), cost_tracking_ctx():
            model = LiteLLMEmbeddingModel(name="text-embedding-3-small", ndim=8)
            await model.embed_documents(stub_texts)


class TestLiteLLMModel:
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "model_name": CommonLLMNames.OPENAI_TEST.value,
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.OPENAI_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.OPENAI_TEST.value,
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ],
                },
                id="OpenAI-model",
            ),
            pytest.param(
                {
                    "model_name": CommonLLMNames.ANTHROPIC_TEST.value,
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
    async def test_cost_call(self, config: dict[str, Any]) -> None:
        with assert_costs_increased(), cost_tracking_ctx():
            llm = LiteLLMModel(name=config["model_name"], config=config)
            messages = [
                Message(role="system", content="Respond with single words."),
                Message(role="user", content="What is the meaning of the universe?"),
            ]
            await llm.call(messages)

    @pytest.mark.asyncio
    async def test_cost_call_w_figure(self) -> None:
        async def ac(x) -> None:
            pass

        with cost_tracking_ctx():
            with assert_costs_increased():
                llm = LiteLLMModel(name=CommonLLMNames.GPT_4O.value)
                image = np.zeros((32, 32, 3), dtype=np.uint8)
                image[:] = [255, 0, 0]
                messages = [
                    Message(
                        role="system",
                        content="You are a detective who investigate colors",
                    ),
                    Message.create_message(
                        role="user",
                        text=(
                            "What color is this square? Show me your chain of"
                            " reasoning."
                        ),
                        images=image,
                    ),
                ]  # TODO: It's not decoding the image. It's trying to guess the color from the encoded image string.
                await llm.call(messages)

            with assert_costs_increased():
                await llm.call(messages, [ac])

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
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
    async def test_cost_call_single(self, config: dict[str, Any]) -> None:
        with cost_tracking_ctx(), assert_costs_increased():
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

            await llm.call_single(
                messages=messages,
                callbacks=[accum],
            )


class TestCostTrackerCallback:
    @pytest.mark.asyncio
    async def test_callback_succeeds(self):
        mock_response = MagicMock(cost=0.01)
        callback_calls = []

        async def async_callback(response):  # noqa: RUF029
            callback_calls.append(response)

        GLOBAL_COST_TRACKER.add_callback(async_callback)

        with (
            cost_tracking_ctx(),
            patch("litellm.cost_calculator.completion_cost", return_value=0.01),
        ):
            await GLOBAL_COST_TRACKER.record(mock_response)

            assert len(callback_calls) == 1
            assert callback_calls[0] == mock_response
            assert GLOBAL_COST_TRACKER.lifetime_cost_usd > 0

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_break_tracker(self, caplog):
        mock_response = MagicMock(cost=0.01)
        failing_callback = MagicMock(side_effect=Exception("Callback failed"))
        GLOBAL_COST_TRACKER.add_callback(failing_callback)

        with (
            cost_tracking_ctx(),
            patch("litellm.cost_calculator.completion_cost", return_value=0.01),
        ):
            await GLOBAL_COST_TRACKER.record(mock_response)

            failing_callback.assert_called_once_with(mock_response)

            assert "Callback failed during cost tracking" in caplog.text
            assert "Callback failed" in caplog.text
            assert GLOBAL_COST_TRACKER.lifetime_cost_usd > 0

    @pytest.mark.asyncio
    async def test_multiple_callbacks_with_one_failing(self, caplog):
        mock_response = MagicMock(cost=0.01)
        failing_callback = MagicMock(side_effect=Exception("Callback failed"))
        succeeding_callback = MagicMock()

        GLOBAL_COST_TRACKER.add_callback(failing_callback)
        GLOBAL_COST_TRACKER.add_callback(succeeding_callback)

        with (
            cost_tracking_ctx(),
            patch("litellm.cost_calculator.completion_cost", return_value=0.01),
        ):
            await GLOBAL_COST_TRACKER.record(mock_response)

            failing_callback.assert_called_once_with(mock_response)
            succeeding_callback.assert_called_once_with(mock_response)

            assert "Callback failed during cost tracking" in caplog.text
            assert GLOBAL_COST_TRACKER.lifetime_cost_usd > 0

    @pytest.mark.asyncio
    async def test_async_context_with_stream_wrapper(self):
        mock_response = MagicMock(cost=0.01)
        mock_stream = MagicMock(__anext__=AsyncMock(return_value=mock_response))
        wrapper = TrackedStreamWrapper(mock_stream)

        callback_calls = []

        async def async_callback(response):  # noqa: RUF029
            callback_calls.append(response)

        GLOBAL_COST_TRACKER.add_callback(async_callback)

        with (
            cost_tracking_ctx(),
            patch("litellm.cost_calculator.completion_cost", return_value=0.01),
        ):
            result = await anext(wrapper)

            assert result == mock_response
            assert len(callback_calls) == 1
            assert callback_calls[0] == mock_response
            assert GLOBAL_COST_TRACKER.lifetime_cost_usd > 0

    @pytest.mark.asyncio
    async def test_stream_wrapper_only_records_final_chunk(self):
        """Test that cost callbacks are only fired on the final chunk with usage info."""
        # Create mock stream that yields 3 chunks: 2 intermediate, 1 final
        intermediate_chunk_1 = MagicMock(usage=None)
        intermediate_chunk_2 = MagicMock(usage=None)
        final_chunk = MagicMock(usage=MagicMock(prompt_tokens=10, completion_tokens=20))

        mock_stream = MagicMock(
            __anext__=AsyncMock(
                side_effect=[
                    intermediate_chunk_1,
                    intermediate_chunk_2,
                    final_chunk,
                    StopAsyncIteration,
                ]
            )
        )

        wrapper = TrackedStreamWrapper(mock_stream)

        callback_calls = []

        async def async_callback(response):  # noqa: RUF029
            callback_calls.append(response)

        GLOBAL_COST_TRACKER.add_callback(async_callback)

        with (
            cost_tracking_ctx(),
            patch("litellm.cost_calculator.completion_cost", return_value=0.01),
        ):
            # Consume all chunks
            chunks = [chunk async for chunk in wrapper]

            # Should have received 3 chunks
            assert len(chunks) == 3
            assert chunks[0] == intermediate_chunk_1
            assert chunks[1] == intermediate_chunk_2
            assert chunks[2] == final_chunk

            # But callback should only have been called once (for final chunk)
            assert len(callback_calls) == 1
            assert callback_calls[0] == final_chunk
            assert GLOBAL_COST_TRACKER.lifetime_cost_usd > 0

    @pytest.mark.vcr(match_on=VCR_DEFAULT_MATCH_ON)
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_name", "stream"),
        [
            (CommonLLMNames.OPENAI_TEST, False),
            (CommonLLMNames.OPENAI_TEST, True),
            (CommonLLMNames.ANTHROPIC_TEST, False),
            (CommonLLMNames.ANTHROPIC_TEST, True),
        ],
    )
    async def test_cost_tracking_with_streaming_modes(self, model_name, stream):
        """Test cost tracking works for both streaming and non-streaming completions."""
        model = LiteLLMModel(name=model_name)
        callback_calls = []

        async def track_callback(response):  # noqa: RUF029
            callback_calls.append(response)

        GLOBAL_COST_TRACKER.add_callback(track_callback)

        with cost_tracking_ctx():
            initial_cost = GLOBAL_COST_TRACKER.lifetime_cost_usd

            if stream:
                # Test streaming via callbacks
                chunks: list[str] = []
                await model.call_single(
                    messages=[Message(content="Say hello")],
                    callbacks=[chunks.append],
                )
                assert chunks  # Should have received streaming chunks
            else:
                # Test non-streaming
                result = await model.call_single(
                    messages=[Message(content="Say hello")],
                )
                assert result.text

            # Cost should have increased
            assert GLOBAL_COST_TRACKER.lifetime_cost_usd > initial_cost

            # Callback should have been called exactly once (on final result with cost)
            assert len(callback_calls) == 1
            response = callback_calls[0]
            assert response.usage is not None
            assert response.usage.prompt_tokens > 0
            assert response.usage.completion_tokens > 0

    @pytest.mark.vcr(match_on=VCR_DEFAULT_MATCH_ON)
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model_name",
        [
            "text-embedding-3-small",
            # Note: Most embedding APIs don't support streaming
        ],
    )
    async def test_cost_tracking_embeddings(self, model_name):
        """Test cost tracking works for embedding models."""
        model = LiteLLMEmbeddingModel(name=model_name)
        callback_calls = []

        async def track_callback(response):  # noqa: RUF029
            callback_calls.append(response)

        GLOBAL_COST_TRACKER.add_callback(track_callback)

        with cost_tracking_ctx():
            initial_cost = GLOBAL_COST_TRACKER.lifetime_cost_usd

            embeddings = await model.embed_documents(["Hello world", "Test"])
            assert len(embeddings) == 2

            # Cost should have increased
            assert GLOBAL_COST_TRACKER.lifetime_cost_usd > initial_cost

            # Callback should have been called exactly once
            assert len(callback_calls) == 1
