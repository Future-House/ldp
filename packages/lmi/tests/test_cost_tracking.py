from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from aviary.core import Message

from lmi import cost_tracking_ctx
from lmi.cost_tracker import GLOBAL_COST_TRACKER, TrackedStreamWrapper
from lmi.embeddings import LiteLLMEmbeddingModel
from lmi.llms import CommonLLMNames, LiteLLMModel, parse_cached_usage
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


class TestCachedTokenCosts:
    """Tests for cached token cost tracking and field extraction."""

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_anthropic_cache_creation_cost_accuracy(self):
        """
        Audit that cache creation tokens are priced correctly.

        Verifies that when cache_creation_input_tokens > 0, the cost calculation
        applies the cache creation rate.
        """
        with cost_tracking_ctx():
            model = LiteLLMModel(
                name="claude-sonnet-4-5-20250929", config={"temperature": 0}
            )

            # Use system message with cache_control for cache creation
            # Anthropic requires at least 1024 tokens for caching
            long_text = "You are analyzing a large document. " * 250  # ~1000+ tokens

            # Pass system with cache_control via kwargs (LiteLLM passes through to Anthropic)
            messages = [
                Message(role="user", content="Summarize your task in one sentence."),
            ]

            result = await model.call_single(
                messages,
                system=[
                    {
                        "type": "text",
                        "text": long_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            )

            # Simple assertions like litellm examples
            assert (  # noqa: PT018
                result.cache_creation_tokens is not None
                and result.cache_creation_tokens > 0
            ), (
                f"Expected cache_creation_tokens > 0, got {result.cache_creation_tokens}. "
                f"Prompt had {result.prompt_count} tokens (need 1024+ for caching)."
            )
            assert result.cost > 0

    @pytest.mark.asyncio
    @pytest.mark.vcr
    @pytest.mark.parametrize(
        ("model_name", "use_cache_control"),
        [
            ("claude-sonnet-4-5-20250929", True),  # Anthropic: explicit cache_control
            ("gpt-4o-2024-11-20", False),  # OpenAI: automatic caching
        ],
    )
    async def test_cache_read_cost_accuracy(self, model_name, use_cache_control):
        """
        Audit that cache read tokens are priced at lower rate.

        Verifies that cache_read tokens > 0 and cost calculation
        applies the reduced cache read rate instead of regular input rate.

        Note: Anthropic requires explicit cache_control.
        """
        with cost_tracking_ctx():
            model = LiteLLMModel(name=model_name, config={"temperature": 0})

            long_text = "You are analyzing a large document. " * 250

            def make_user_content(text: str) -> str | list[dict[str, Any]]:
                if use_cache_control:
                    return [
                        {
                            "type": "text",
                            "text": text,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                return text

            messages = [
                Message(role="system", content=long_text),
                Message(
                    role="user",
                    content=make_user_content("Summarize your task in one sentence."),
                ),
            ]
            result1 = await model.call_single(messages)

            messages_2 = [
                Message(role="system", content=long_text),
                Message(
                    role="user", content=make_user_content("What is your objective?")
                ),
            ]
            result2 = await model.call_single(messages_2)

            assert (
                (
                    result1.cache_creation_tokens is not None
                    and result1.cache_creation_tokens > 0
                )
                or (
                    result1.cache_read_tokens is not None
                    and result1.cache_read_tokens > 0
                )
                or (
                    result2.cache_read_tokens is not None
                    and result2.cache_read_tokens > 0
                )
            ), (
                f"Expected caching in at least one call for {model_name}, "
                f"got call1 (creation={result1.cache_creation_tokens}, read={result1.cache_read_tokens}), "
                f"call2 (read={result2.cache_read_tokens})"
            )

            assert (  # noqa: PT018
                result2.cache_read_tokens is not None and result2.cache_read_tokens > 0
            ), f"Expected cache_read_tokens > 0 on second call for {model_name}"
            assert result2.cost > 0

            if (
                result1.cache_creation_tokens is not None
                and result1.cache_creation_tokens > 0
            ):
                # Anthropic: cache read (0.1x) < cache creation (1.25x)
                assert result2.cost < result1.cost, (
                    f"Cache read cost ({result2.cost:.6f}) should be < cache creation cost ({result1.cost:.6f}) for {model_name}. "
                    "If this fails, caching discount is not being applied!"
                )
            elif result1.cache_read_tokens is None or result1.cache_read_tokens == 0:
                # OpenAI: cache read (discounted) <= no cache
                assert result2.cost <= result1.cost, (
                    f"Call 2 with cache ({result2.cost:.6f}) should be <= call 1 without cache ({result1.cost:.6f}) for {model_name}"
                )

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_anthropic_incremental_cache_with_conversation(self):
        """
        Test incremental caching as conversation grows.

        Demonstrates how cache grows with each message by placing cache_control
        on the last message in the conversation. Shows that:
        1. First call: Creates cache for system + first user message
        2. Second call: Reads system+user cache, adds assistant response to cache
        3. Third call: Reads all previous messages from cache
        """
        with cost_tracking_ctx():
            model = LiteLLMModel(
                name="claude-sonnet-4-5-20250929", config={"temperature": 0}
            )

            # System prompt (different from other tests to ensure fresh cache creation)
            long_text = (
                "You are a helpful assistant for testing incremental prompt caching. "
                * 200
            )  # ~1000+ tokens

            # Call 1: System + first user message
            # Put cache_control ONLY on the last message to cache everything up to it
            messages_1 = [
                Message(
                    role="user",
                    content=[
                        {
                            "type": "text",
                            "text": "What is your first task?",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                ),
            ]

            system_with_cache = [
                {
                    "type": "text",
                    "text": long_text,
                }
            ]

            result1 = await model.call_single(messages_1, system=system_with_cache)

            # Call 2: Add assistant response + new user message
            # Only need cache_control on LAST message (Anthropic auto-finds longest prefix)
            messages_2 = [
                Message(role="user", content="What is your first task?"),
                Message(
                    role="assistant", content=result1.text or "Analyzing documents."
                ),
                Message(
                    role="user",
                    content=[
                        {
                            "type": "text",
                            "text": "What is your second task?",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                ),
            ]

            result2 = await model.call_single(messages_2, system=system_with_cache)

            # Call 3: Add another assistant response + new user message
            # Only cache_control on LAST message
            messages_3 = [
                Message(role="user", content="What is your first task?"),
                Message(
                    role="assistant", content=result1.text or "Analyzing documents."
                ),
                Message(role="user", content="What is your second task?"),
                Message(
                    role="assistant", content=result2.text or "Summarizing findings."
                ),
                Message(
                    role="user",
                    content=[
                        {
                            "type": "text",
                            "text": "What is your third task?",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                ),
            ]

            result3 = await model.call_single(messages_3, system=system_with_cache)

            # Assertions
            # Call 1: Should create initial cache (system + first user)
            assert (
                result1.cache_creation_tokens is not None
                and result1.cache_creation_tokens > 0
            ) or (
                result1.cache_read_tokens is not None and result1.cache_read_tokens > 0
            ), "Expected caching on first call"

            # Call 2: Should read some cache and create new cache for added messages
            assert (
                result2.cache_read_tokens is not None and result2.cache_read_tokens > 0
            ) or (
                result2.cache_creation_tokens is not None
                and result2.cache_creation_tokens > 0
            ), "Expected cache activity on second call"

            # Call 3: Should read from cache (conversation already cached)
            assert (
                result3.cache_read_tokens is not None and result3.cache_read_tokens > 0
            ) or (
                result3.cache_creation_tokens is not None
                and result3.cache_creation_tokens > 0
            ), "Expected cache activity on third call"

            # Call 4: Skip messages 2 & 3 - test partial cache hit
            # Only cache_control on LAST message
            messages_4 = [
                Message(role="user", content="What is your first task?"),
                Message(
                    role="user",
                    content=[
                        {
                            "type": "text",
                            "text": "What is your FOURTH task?",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                ),
            ]

            result4 = await model.call_single(messages_4, system=system_with_cache)

            # Assertions - Verify caching works + test incremental growth
            # Call 1: Should have meaningful cache activity
            assert (
                result1.cache_creation_tokens is not None
                and result1.cache_creation_tokens > 0
            ) or (
                result1.cache_read_tokens is not None and result1.cache_read_tokens > 0
            ), "Expected cache activity on call 1"
            assert result1.cost > 0
            total_cache_1 = (result1.cache_creation_tokens or 0) + (
                result1.cache_read_tokens or 0
            )

            # Call 2: Cache should grow (adds assistant + user message)
            assert (
                result2.cache_creation_tokens is not None
                and result2.cache_creation_tokens > 0
            ) or (
                result2.cache_read_tokens is not None and result2.cache_read_tokens > 0
            ), "Expected cache activity on call 2"
            assert result2.cost > 0
            total_cache_2 = (result2.cache_creation_tokens or 0) + (
                result2.cache_read_tokens or 0
            )
            assert total_cache_2 >= total_cache_1, (
                f"Expected cache to grow: call 2 ({total_cache_2}) >= call 1 ({total_cache_1})"
            )

            # CRITICAL: If call 2 reads from cache, verify discount is applied
            if (
                result2.cache_read_tokens is not None
                and result2.cache_read_tokens > 0
                and result1.cache_creation_tokens is not None
                and result1.cache_creation_tokens > 0
            ):
                # Cache read (0.1x) is much cheaper than cache creation (1.25x)
                # Even with small new writes, total cost should be significantly lower
                assert result2.cost < result1.cost, (
                    f"Call 2 with cache reads ({result2.cost:.6f}) should be < "
                    f"call 1 with cache creation ({result1.cost:.6f}). "
                    "Cache discount not being applied!"
                )

            # Call 3: Cache should continue growing (adds more conversation)
            assert (
                result3.cache_creation_tokens is not None
                and result3.cache_creation_tokens > 0
            ) or (
                result3.cache_read_tokens is not None and result3.cache_read_tokens > 0
            ), "Expected cache activity on call 3"
            assert result3.cost is not None
            assert result3.cost > 0
            total_cache_3 = (result3.cache_creation_tokens or 0) + (
                result3.cache_read_tokens or 0
            )
            assert total_cache_3 >= total_cache_2, (
                f"Expected cache to keep growing: call 3 ({total_cache_3}) >= call 2 ({total_cache_2})"
            )

            # Call 4: Should have cache (partial hit - sequence changed)
            assert (
                result4.cache_creation_tokens is not None
                and result4.cache_creation_tokens > 0
            ) or (
                result4.cache_read_tokens is not None and result4.cache_read_tokens > 0
            ), "Expected cache activity on call 4"
            assert result4.cost is not None
            assert result4.cost > 0
            total_cache_4 = (result4.cache_creation_tokens or 0) + (
                result4.cache_read_tokens or 0
            )
            assert total_cache_4 >= total_cache_1, (
                f"Expected at least partial cache: call 4 ({total_cache_4}) >= call 1 ({total_cache_1})"
            )

    @pytest.mark.asyncio
    @pytest.mark.vcr
    @pytest.mark.parametrize(
        ("model_name", "use_cache_control"),
        [
            ("claude-sonnet-4-5-20250929", True),  # Anthropic: needs cache_control
            ("gpt-4o-2024-11-20", False),  # OpenAI: automatic caching
        ],
    )
    async def test_cache_with_streaming(self, model_name, use_cache_control):
        """
        Test that cached tokens are extracted correctly with streaming responses.

        Verifies that acompletion_iter() (streaming mode) correctly extracts
        cache_read_tokens and cache_creation_tokens from streamed responses.

        Note: Anthropic requires explicit cache_control.
        """
        with cost_tracking_ctx():
            model = LiteLLMModel(name=model_name, config={"temperature": 0})

            long_text = "You are a streaming response assistant. " * 250

            user_content: str | list[dict[str, Any]] = "Count to 5."
            if use_cache_control:
                user_content = [
                    {
                        "type": "text",
                        "text": "Count to 5.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

            messages = [
                Message(role="system", content=long_text),
                Message(role="user", content=user_content),
            ]

            chunks1: list[str] = []
            result1 = await model.call_single(messages, callbacks=[chunks1.append])

            chunks2: list[str] = []
            result2 = await model.call_single(messages, callbacks=[chunks2.append])

            assert (
                (
                    result1.cache_creation_tokens is not None
                    and result1.cache_creation_tokens > 0
                )
                or (
                    result1.cache_read_tokens is not None
                    and result1.cache_read_tokens > 0
                )
                or (
                    result2.cache_read_tokens is not None
                    and result2.cache_read_tokens > 0
                )
            ), f"Expected cache activity in at least one call for {model_name}"

            assert result1.cost > 0
            assert result2.cost > 0

            if (
                result2.cache_read_tokens is not None
                and result2.cache_read_tokens > 0
                and result1.cache_creation_tokens is not None
                and result1.cache_creation_tokens > 0
            ):
                assert result2.cost < result1.cost, (
                    f"Cache read cost ({result2.cost:.6f}) should be < "
                    f"cache creation cost ({result1.cost:.6f}) for {model_name}"
                )

            assert chunks1, f"Expected streaming chunks on first call for {model_name}"
            assert chunks2, f"Expected streaming chunks on second call for {model_name}"


class TestCachedTokenEdgeCases:
    """Edge case tests for cached token extraction."""

    def test_parse_cached_usage_with_none(self):
        """Test parse_cached_usage handles None usage gracefully."""
        cache_read, cache_creation = parse_cached_usage(None)
        assert cache_read is None
        assert cache_creation is None

    def test_parse_cached_usage_with_missing_fields(self):
        """Test parse_cached_usage handles usage object with no cached fields."""
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        # No prompt_tokens_details, no cache fields
        del usage.prompt_tokens_details
        del usage.cache_read_input_tokens
        del usage.cache_creation_input_tokens

        cache_read, cache_creation = parse_cached_usage(usage)
        assert cache_read is None
        assert cache_creation is None

    def test_parse_cached_usage_with_dict_format(self):
        """Test parse_cached_usage handles dict-style prompt_tokens_details."""
        usage = MagicMock()
        usage.prompt_tokens_details = {"cached_tokens": 1024}

        cache_read, cache_creation = parse_cached_usage(usage)
        assert cache_read == 1024
        assert cache_creation is None

    def test_parse_cached_usage_with_object_format(self):
        """Test parse_cached_usage handles object-style prompt_tokens_details."""
        usage = MagicMock()
        prompt_details = MagicMock()
        prompt_details.cached_tokens = 2048
        usage.prompt_tokens_details = prompt_details

        cache_read, cache_creation = parse_cached_usage(usage)
        assert cache_read == 2048
        assert cache_creation is None

    def test_parse_cached_usage_with_anthropic_cache_creation(self):
        """Test parse_cached_usage extracts Anthropic's cache_creation_input_tokens."""
        usage = MagicMock()
        # Anthropic responses have both normalized cache_read and top-level cache_creation
        usage.prompt_tokens_details = {"cached_tokens": 1500}
        usage.cache_creation_input_tokens = 200

        cache_read, cache_creation = parse_cached_usage(usage)
        assert cache_read == 1500
        assert cache_creation == 200

    def test_parse_cached_usage_with_non_int_values(self):
        """Test parse_cached_usage handles non-integer values (returns None)."""
        usage = MagicMock()
        usage.prompt_tokens_details = {"cached_tokens": "not_an_int"}

        cache_read, cache_creation = parse_cached_usage(usage)
        assert cache_read is None  # Should handle gracefully by ignoring non-int values
        assert cache_creation is None

    @pytest.mark.asyncio
    @pytest.mark.vcr
    async def test_completion_cost_failure_with_cached_tokens(self):
        """
        Test that cached token fields are still populated even if completion_cost fails.

        Verifies that:
        1. Cached token fields (cache_read_tokens, cache_creation_tokens) are extracted
        2. cost is 0 when completion_cost fails
        """
        with cost_tracking_ctx():
            from unittest.mock import patch

            model = LiteLLMModel(
                name="claude-sonnet-4-5-20250929", config={"temperature": 0}
            )

            long_text = "You are testing cost calculation failure handling. " * 250

            messages = [Message(role="user", content="Say hi.")]

            system_with_cache = [
                {
                    "type": "text",
                    "text": long_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

            # Patch completion_cost to raise an exception
            with patch(
                "lmi.llms.completion_cost", side_effect=Exception("Cost calc failed")
            ):
                result = await model.call_single(messages, system=system_with_cache)

                # Cached token fields should still be populated
                assert (
                    result.cache_creation_tokens is not None
                    and result.cache_creation_tokens > 0
                ) or (
                    result.cache_read_tokens is not None
                    and result.cache_read_tokens > 0
                ), "Cached tokens should be extracted even if cost calculation fails"

                # cost should be 0 (calculation failed)
                assert result.cost == 0, (
                    "Expected cost to be 0 when completion_cost fails"
                )
