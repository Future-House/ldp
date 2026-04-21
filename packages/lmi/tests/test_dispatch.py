"""Tests for the retry/fallback loop, streaming first-chunk commit, and dispatch.

End-to-end tests mock `litellm.acompletion` to drive `LiteLLMModel.call()` through
retry-same-model → advance-to-next-model → all-exhausted scenarios without any
network access. Unit-level tests exercise `_run_with_fallbacks` and
`_commit_stream` directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from aviary.core import Message
from litellm.types.utils import Choices, ModelResponse, Usage
from litellm.types.utils import Message as LiteLLMMessage

from lmi.config import LLMConfig, ModelSpec
from lmi.exceptions import AllModelsExhaustedError, ModelRefusalError
from lmi.llms import LiteLLMModel, _commit_stream
from lmi.types import LLMResult


def _chunk(text: str) -> LLMResult:
    return LLMResult(model=PRIMARY, text=text)


PRIMARY = "gpt-5-mini"
FALLBACK = "claude-sonnet-4-6"


def _two_model_chain(primary: str = PRIMARY, fallback: str = FALLBACK) -> LLMConfig:
    return LLMConfig(
        models=[
            ModelSpec(name=primary, max_retries=2, timeout=5.0),
            ModelSpec(name=fallback, max_retries=1, timeout=5.0),
        ]
    )


def _model(cfg: LLMConfig | None = None) -> LiteLLMModel:
    return LiteLLMModel(name=PRIMARY, llm_config=cfg or _two_model_chain())


def _rate_limit_exc(model: str = PRIMARY) -> litellm.RateLimitError:
    return litellm.RateLimitError(
        message="rate limited", model=model, llm_provider="openai"
    )


def _context_overflow_exc(model: str = PRIMARY) -> litellm.ContextWindowExceededError:
    return litellm.ContextWindowExceededError(
        message="too many tokens", model=model, llm_provider="openai"
    )


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero out the exponential-jitter sleep so tests don't wait."""
    monkeypatch.setattr("lmi.llms.backoff_seconds", lambda _attempt: 0.0)


class TestRunWithFallbacks:
    @pytest.mark.asyncio
    async def test_returns_first_success(self) -> None:
        model = _model()
        attempt = AsyncMock(return_value="ok")
        result = await model._run_with_fallbacks(attempt)
        assert result == "ok"
        assert attempt.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_same_model_on_transient(self) -> None:
        model = _model()
        attempt = AsyncMock(side_effect=[_rate_limit_exc(), _rate_limit_exc(), "ok"])
        result = await model._run_with_fallbacks(attempt)
        assert result == "ok"
        # All three calls used the primary spec (same model)
        assert attempt.call_count == 3
        specs = [call.args[0] for call in attempt.await_args_list]
        assert all(s.name == PRIMARY for s in specs)

    @pytest.mark.asyncio
    async def test_retries_exhausted_advances_to_fallback(self) -> None:
        model = _model()
        # Primary: max_retries=2 -> 3 attempts, all rate-limited. Then fallback succeeds.
        attempt = AsyncMock(
            side_effect=[
                _rate_limit_exc(),
                _rate_limit_exc(),
                _rate_limit_exc(),
                "ok",
            ]
        )
        result = await model._run_with_fallbacks(attempt)
        assert result == "ok"
        specs = [call.args[0] for call in attempt.await_args_list]
        assert [s.name for s in specs] == [PRIMARY, PRIMARY, PRIMARY, FALLBACK]

    @pytest.mark.asyncio
    async def test_fallbackable_exception_advances_immediately(self) -> None:
        model = _model()
        # Context overflow on primary -> no retries, go straight to fallback.
        attempt = AsyncMock(side_effect=[_context_overflow_exc(), "ok"])
        result = await model._run_with_fallbacks(attempt)
        assert result == "ok"
        specs = [call.args[0] for call in attempt.await_args_list]
        assert [s.name for s in specs] == [PRIMARY, FALLBACK]

    @pytest.mark.asyncio
    async def test_model_refusal_advances_to_fallback(self) -> None:
        model = _model()
        refusal = ModelRefusalError(
            "safety", model=PRIMARY, finish_reason="content_filter"
        )
        attempt = AsyncMock(side_effect=[refusal, "ok"])
        result = await model._run_with_fallbacks(attempt)
        assert result == "ok"
        specs = [call.args[0] for call in attempt.await_args_list]
        assert [s.name for s in specs] == [PRIMARY, FALLBACK]

    @pytest.mark.asyncio
    async def test_fatal_exception_propagates(self) -> None:
        model = _model()
        # An uncategorized exception (caller bug, malformed Message, etc.) is
        # neither retryable nor fallback-able; it propagates immediately.
        attempt = AsyncMock(side_effect=[ValueError("caller bug")])
        with pytest.raises(ValueError, match="caller bug"):
            await model._run_with_fallbacks(attempt)
        # Did not fall back to the second model.
        assert attempt.call_count == 1

    @pytest.mark.asyncio
    async def test_auth_error_advances_to_fallback(self) -> None:
        model = _model()
        auth_err = litellm.AuthenticationError(
            message="bad key", model=PRIMARY, llm_provider="openai"
        )
        attempt = AsyncMock(side_effect=[auth_err, "ok"])
        result = await model._run_with_fallbacks(attempt)
        assert result == "ok"
        specs = [call.args[0] for call in attempt.await_args_list]
        assert [s.name for s in specs] == [PRIMARY, FALLBACK]

    @pytest.mark.asyncio
    async def test_all_models_exhausted_raises(self) -> None:
        model = _model()
        # Every attempt rate-limited on both models.
        attempt = AsyncMock(
            side_effect=[
                _rate_limit_exc(),
                _rate_limit_exc(),
                _rate_limit_exc(),
                _rate_limit_exc(FALLBACK),
                _rate_limit_exc(FALLBACK),
            ]
        )
        with pytest.raises(AllModelsExhaustedError) as excinfo:
            await model._run_with_fallbacks(attempt)
        assert attempt.call_count == 5  # 3 on primary + 2 on fallback
        assert isinstance(excinfo.value.last_exc, litellm.RateLimitError)

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs_to_attempt(self) -> None:
        model = _model()
        attempt = AsyncMock(return_value="ok")
        await model._run_with_fallbacks(attempt, "a", "b", path="chat", extra=1)
        # First positional is the spec; the rest are the forwarded args.
        call = attempt.await_args_list[0]
        assert call.args[1:] == ("a", "b")
        assert call.kwargs == {"path": "chat", "extra": 1}


class TestCommitStream:
    @pytest.mark.asyncio
    async def test_replays_all_chunks(self) -> None:
        async def gen() -> AsyncIterator[LLMResult]:  # noqa: RUF029
            for chunk in ("a", "b", "c"):
                yield _chunk(chunk)

        committed = await _commit_stream(gen())
        collected = [r.text async for r in committed]
        assert collected == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_pre_first_chunk_error_propagates(self) -> None:
        async def gen() -> AsyncIterator[LLMResult]:  # noqa: RUF029
            raise litellm.RateLimitError(
                message="pre-first-chunk", model=PRIMARY, llm_provider="openai"
            )
            yield _chunk("unreachable")  # type: ignore[unreachable]  # pragma: no cover

        with pytest.raises(litellm.RateLimitError):
            await _commit_stream(gen())

    @pytest.mark.asyncio
    async def test_empty_stream_raises_runtime_error(self) -> None:
        async def gen() -> AsyncIterator[LLMResult]:  # noqa: RUF029
            return
            yield _chunk("unreachable")  # type: ignore[unreachable]  # pragma: no cover

        with pytest.raises(RuntimeError, match="Stream closed before"):
            await _commit_stream(gen())

    @pytest.mark.asyncio
    async def test_mid_stream_error_surfaces_unmodified(self) -> None:
        async def gen() -> AsyncIterator[LLMResult]:  # noqa: RUF029
            yield _chunk("a")
            raise litellm.APIConnectionError(
                message="mid-stream", model=PRIMARY, llm_provider="openai"
            )

        committed = await _commit_stream(gen())
        first = await anext(aiter(committed))
        # First chunk is delivered; the error only surfaces on the *next* pull.
        assert first.text == "a"
        with pytest.raises(litellm.APIConnectionError):
            async for _ in committed:
                pass


def _fake_chat_response(
    model: str = PRIMARY,
    text: str = "hello",
    finish_reason: str = "stop",
) -> ModelResponse:
    """Build a minimal litellm ModelResponse for non-streaming chat."""
    return ModelResponse(
        id="chatcmpl-test",
        model=model,
        choices=[
            Choices(
                finish_reason=finish_reason,
                message=LiteLLMMessage(content=text),
            )
        ],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


class TestCallEndToEnd:
    """Drive LiteLLMModel.call() with a mocked litellm.acompletion."""

    @pytest.mark.asyncio
    async def test_retries_then_succeeds_on_primary(self) -> None:
        model = _model()
        responses: list[Any] = [_rate_limit_exc(), _fake_chat_response()]
        with patch(
            "litellm.acompletion", AsyncMock(side_effect=responses)
        ) as mock_call:
            results = await model.call([Message(content="hi")])

        assert len(results) == 1
        assert results[0].text == "hello"
        # 2 attempts, both on primary.
        assert mock_call.await_count == 2
        assert all(
            call.kwargs["model"] == PRIMARY for call in mock_call.await_args_list
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_secondary_on_context_overflow(self) -> None:
        model = _model()
        responses: list[Any] = [
            _context_overflow_exc(),
            _fake_chat_response(model=FALLBACK),
        ]
        with patch(
            "litellm.acompletion", AsyncMock(side_effect=responses)
        ) as mock_call:
            results = await model.call([Message(content="hi")])

        assert len(results) == 1
        assert results[0].model == FALLBACK
        models_tried = [call.kwargs["model"] for call in mock_call.await_args_list]
        assert models_tried == [PRIMARY, FALLBACK]

    @pytest.mark.asyncio
    async def test_refusal_triggers_fallback(self) -> None:
        model = _model()
        refusal_resp = _fake_chat_response(finish_reason="content_filter", text="no")
        ok_resp = _fake_chat_response(model=FALLBACK, text="ok")
        with patch(
            "litellm.acompletion", AsyncMock(side_effect=[refusal_resp, ok_resp])
        ) as mock_call:
            results = await model.call([Message(content="hi")])

        assert len(results) == 1
        assert results[0].model == FALLBACK
        assert results[0].text == "ok"
        models_tried = [call.kwargs["model"] for call in mock_call.await_args_list]
        assert models_tried == [PRIMARY, FALLBACK]

    @pytest.mark.asyncio
    async def test_all_models_exhausted_raises(self) -> None:
        model = _model()
        # 3 attempts on primary (max_retries=2) + 2 on fallback (max_retries=1) = 5
        side_effects = [_rate_limit_exc()] * 5
        with (
            patch(
                "litellm.acompletion", AsyncMock(side_effect=side_effects)
            ) as mock_call,
            pytest.raises(AllModelsExhaustedError),
        ):
            await model.call([Message(content="hi")])
        assert mock_call.await_count == 5

    @pytest.mark.asyncio
    async def test_fatal_exception_propagates_unwrapped(self) -> None:
        model = _model()
        # Uncategorized exceptions (caller bug, Python-level error) propagate
        # without being wrapped in AllModelsExhaustedError.
        with (
            patch(
                "litellm.acompletion", AsyncMock(side_effect=[ValueError("caller bug")])
            ) as mock_call,
            pytest.raises(ValueError, match="caller bug"),
        ):
            await model.call([Message(content="hi")])
        assert mock_call.await_count == 1

    @pytest.mark.asyncio
    async def test_spec_extra_params_flow_to_litellm(self) -> None:
        cfg = LLMConfig(
            models=[
                ModelSpec(
                    name=PRIMARY,
                    max_retries=0,
                    timeout=7.5,
                    extra_params={"temperature": 0.2, "max_tokens": 42},
                )
            ]
        )
        model = LiteLLMModel(name=PRIMARY, llm_config=cfg)
        with patch(
            "litellm.acompletion",
            AsyncMock(return_value=_fake_chat_response()),
        ) as mock_call:
            await model.call([Message(content="hi")])

        call_kwargs = mock_call.await_args_list[0].kwargs
        assert call_kwargs["model"] == PRIMARY
        assert call_kwargs["timeout"] == 7.5
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 42


class TestDispatchPrimitiveSelection:
    """Verify `_dispatch` routes to the right primitive.

    Routing depends on `spec.responses_api` and whether `streaming=True` was passed.
    """

    @pytest.mark.asyncio
    async def test_chat_non_streaming(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _model()
        seen: dict[str, Any] = {}

        async def fake_acompletion(  # noqa: RUF029
            _self: LiteLLMModel,
            messages: Any,  # noqa: ARG001
            *,
            spec: ModelSpec,
            **_kwargs: Any,
        ) -> list[Any]:
            seen["primitive"] = "acompletion"
            seen["spec"] = spec
            return []

        monkeypatch.setattr(LiteLLMModel, "acompletion", fake_acompletion)
        await model.call([Message(content="hi")])
        assert seen["primitive"] == "acompletion"
        assert seen["spec"].name == PRIMARY

    @pytest.mark.asyncio
    async def test_chat_streaming(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = _model()
        seen: dict[str, Any] = {}

        async def one_chunk() -> AsyncIterator[LLMResult]:  # noqa: RUF029
            yield _chunk("ok")

        async def fake_acompletion_iter(  # noqa: RUF029
            _self: LiteLLMModel,
            messages: Any,  # noqa: ARG001
            *,
            spec: ModelSpec,  # noqa: ARG001
            **_kwargs: Any,
        ) -> AsyncIterator[LLMResult]:
            seen["primitive"] = "acompletion_iter"
            return one_chunk()

        monkeypatch.setattr(LiteLLMModel, "acompletion_iter", fake_acompletion_iter)
        await model.call([Message(content="hi")], callbacks=[lambda *_a, **_k: None])
        assert seen["primitive"] == "acompletion_iter"

    @pytest.mark.asyncio
    async def test_responses_non_streaming(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = LLMConfig(models=[ModelSpec(name=PRIMARY, responses_api=True)])
        model = LiteLLMModel(name=PRIMARY, llm_config=cfg)
        seen: dict[str, Any] = {}

        async def fake_aresponses(  # noqa: RUF029
            _self: LiteLLMModel,
            messages: Any,  # noqa: ARG001
            tools: Any,  # noqa: ARG001
            previous_response_id: Any = None,  # noqa: ARG001
            *,
            spec: ModelSpec,  # noqa: ARG001
            **_kwargs: Any,
        ) -> list[Any]:
            seen["primitive"] = "_aresponses"
            return []

        monkeypatch.setattr(LiteLLMModel, "_aresponses", fake_aresponses)
        await model.call([Message(content="hi")])
        assert seen["primitive"] == "_aresponses"
