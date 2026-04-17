import litellm
import pytest

from lmi.exceptions import ModelRefusalError
from lmi.retry import (
    BACKOFF_CAP,
    BACKOFF_INITIAL,
    backoff_seconds,
    should_fallback,
    should_retry,
)


def _litellm_exc(cls, message: str = "boom"):
    """Instantiate a litellm exception uniformly across variants."""
    return cls(message=message, model="gpt-4o-mini", llm_provider="openai")


class TestShouldRetry:
    @pytest.mark.parametrize(
        "cls",
        [
            litellm.RateLimitError,
            litellm.Timeout,
            litellm.APIConnectionError,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
        ],
    )
    def test_transient_errors_retry(self, cls) -> None:
        assert should_retry(_litellm_exc(cls)) is True

    @pytest.mark.parametrize(
        "cls",
        [
            litellm.ContextWindowExceededError,
            litellm.NotFoundError,
            litellm.ContentPolicyViolationError,
            litellm.AuthenticationError,
            litellm.BadRequestError,
        ],
    )
    def test_terminal_errors_dont_retry(self, cls) -> None:
        assert should_retry(_litellm_exc(cls)) is False

    def test_model_refusal_does_not_retry(self) -> None:
        exc = ModelRefusalError(
            "refused", model="gpt-4o-mini", finish_reason="content_filter"
        )
        assert should_retry(exc) is False

    def test_generic_exception_does_not_retry(self) -> None:
        assert should_retry(ValueError("nope")) is False


class TestShouldFallback:
    @pytest.mark.parametrize(
        "cls",
        [
            litellm.ContextWindowExceededError,
            litellm.NotFoundError,
            litellm.ContentPolicyViolationError,
        ],
    )
    def test_stable_errors_fallback(self, cls) -> None:
        assert should_fallback(_litellm_exc(cls)) is True

    def test_model_refusal_falls_back(self) -> None:
        exc = ModelRefusalError(
            "refused", model="gpt-4o-mini", finish_reason="content_filter"
        )
        assert should_fallback(exc) is True

    def test_anthropic_too_much_media_falls_back(self) -> None:
        exc = litellm.BadRequestError(
            message="Too much media: 0 document pages + 108 images > 100",
            model="claude-3-5-sonnet-20241022",
            llm_provider="anthropic",
        )
        assert should_fallback(exc) is True

    def test_generic_bad_request_does_not_fall_back(self) -> None:
        exc = litellm.BadRequestError(
            message="Malformed request",
            model="gpt-4o-mini",
            llm_provider="openai",
        )
        assert should_fallback(exc) is False

    @pytest.mark.parametrize(
        "cls",
        [
            litellm.RateLimitError,
            litellm.Timeout,
            litellm.AuthenticationError,
        ],
    )
    def test_retryable_or_fatal_does_not_fall_back(self, cls) -> None:
        assert should_fallback(_litellm_exc(cls)) is False


class TestBackoff:
    def test_returns_non_negative_float(self) -> None:
        for attempt in range(10):
            delay = backoff_seconds(attempt)
            assert 0.0 <= delay <= BACKOFF_CAP

    def test_first_attempt_bounded_by_initial(self) -> None:
        # attempt=0 -> ceiling = min(CAP, INITIAL * 2**0) = INITIAL
        for _ in range(50):
            assert backoff_seconds(0) <= BACKOFF_INITIAL

    def test_ceiling_grows_with_attempt_up_to_cap(self) -> None:
        # attempt large enough that 2**attempt * INITIAL exceeds CAP
        # => every draw is in [0, CAP]
        for _ in range(50):
            assert backoff_seconds(20) <= BACKOFF_CAP
