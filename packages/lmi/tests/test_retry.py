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
    kwargs: dict[str, object] = {
        "message": message,
        "model": "gpt-4o-mini",
        "llm_provider": "openai",
    }
    # PermissionDeniedError inherits from openai.PermissionDeniedError which
    # requires a `response` argument.
    if cls is litellm.PermissionDeniedError:
        import httpx

        kwargs["response"] = httpx.Response(403, request=httpx.Request("POST", "x"))
    return cls(**kwargs)


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
            litellm.PermissionDeniedError,
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
            litellm.AuthenticationError,
            litellm.PermissionDeniedError,
            litellm.BadRequestError,
        ],
    )
    def test_stable_errors_fallback(self, cls) -> None:
        assert should_fallback(_litellm_exc(cls)) is True

    def test_model_refusal_falls_back(self) -> None:
        exc = ModelRefusalError(
            "refused", model="gpt-4o-mini", finish_reason="content_filter"
        )
        assert should_fallback(exc) is True

    @pytest.mark.parametrize(
        "cls",
        [
            litellm.RateLimitError,
            litellm.Timeout,
        ],
    )
    def test_retryable_errors_do_not_fall_back(self, cls) -> None:
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
