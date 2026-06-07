import litellm
import pytest
from tenacity import stop_after_attempt, wait_random_exponential

from lmi.exceptions import ModelRefusalError
from lmi.retry import (
    BACKOFF_CAP,
    BACKOFF_INITIAL,
    model_retrying,
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
            litellm.BadGatewayError,
        ],
    )
    def test_transient_errors_retry(self, cls) -> None:
        assert should_retry(_litellm_exc(cls))

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
        assert not should_retry(_litellm_exc(cls))

    def test_model_refusal_does_not_retry(self) -> None:
        exc = ModelRefusalError(
            "refused", model="gpt-4o-mini", finish_reason="content_filter"
        )
        assert not should_retry(exc)

    def test_generic_exception_does_not_retry(self) -> None:
        assert not should_retry(ValueError("nope"))


class TestShouldFallback:
    @pytest.mark.parametrize(
        "cls",
        [
            litellm.ContextWindowExceededError,
            litellm.NotFoundError,
            litellm.ContentPolicyViolationError,
            litellm.AuthenticationError,
            litellm.PermissionDeniedError,
        ],
    )
    def test_stable_errors_fallback(self, cls) -> None:
        assert should_fallback(_litellm_exc(cls))

    def test_model_refusal_falls_back(self) -> None:
        exc = ModelRefusalError(
            "refused", model="gpt-4o-mini", finish_reason="content_filter"
        )
        assert should_fallback(exc)

    def test_generic_bad_request_does_not_fall_back(self) -> None:
        # A malformed request is a terminal client error, not a provider quirk.
        assert not should_fallback(_litellm_exc(litellm.BadRequestError))

    def test_provider_limit_bad_request_falls_back(self) -> None:
        # Anthropic's 100-image limit surfaces as a generic 400; it should fall
        # over to a sibling model that may accept the same input.
        exc = _litellm_exc(
            litellm.BadRequestError,
            message="Too much media: 0 document pages + 108 images > 100",
        )
        assert should_fallback(exc)

    @pytest.mark.parametrize(
        "cls",
        [
            litellm.RateLimitError,
            litellm.Timeout,
        ],
    )
    def test_retryable_errors_do_not_fall_back(self, cls) -> None:
        assert not should_fallback(_litellm_exc(cls))

    @staticmethod
    def _bad_request(message: str, status: int) -> litellm.BadRequestError:
        import httpx

        return litellm.BadRequestError(
            message=message,
            model="vertex_ai/gemini-2.0-flash",
            llm_provider="vertex_ai",
            response=httpx.Response(
                status_code=status,
                request=httpx.Request("POST", "https://example.test"),
            ),
        )

    def test_mislabelled_403_bad_request_falls_over(self) -> None:
        # litellm collapses a Vertex 403 into a bare BadRequestError; the true
        # status survives only on the response and should trigger fall-over.
        exc = self._bad_request(
            "Vertex_aiException BadRequestError - PERMISSION_DENIED 403", 403
        )
        assert should_fallback(exc)

    def test_generic_400_bad_request_is_terminal(self) -> None:
        exc = self._bad_request("invalid schema for function 'foo'", 400)
        assert not should_fallback(exc)

    def test_provider_limit_400_still_falls_over(self) -> None:
        exc = self._bad_request(
            "Too much media: 0 document pages + 108 images > 100", 400
        )
        assert should_fallback(exc)

    def test_content_safety_refusal_400_falls_over(self) -> None:
        # OpenAI's prompt-level safety block surfaces as a bare 400, not
        # ContentPolicyViolationError; a refusal may be a false positive, so a
        # sibling model should get a chance.
        exc = self._bad_request(
            "Invalid prompt: we've limited access to this content for safety reasons.",
            400,
        )
        assert should_fallback(exc)


class TestModelRetrying:
    def test_attempt_count_is_retries_plus_one(self) -> None:
        stop = model_retrying(3).stop
        assert isinstance(stop, stop_after_attempt)
        assert stop.max_attempt_number == 4
        stop = model_retrying(0).stop
        assert isinstance(stop, stop_after_attempt)
        assert stop.max_attempt_number == 1

    def test_uses_full_jitter_exponential_backoff(self) -> None:
        wait = model_retrying(3).wait
        assert isinstance(wait, wait_random_exponential)
        assert wait.multiplier == BACKOFF_INITIAL
        assert wait.max == BACKOFF_CAP

    def test_reraises_final_failure(self) -> None:
        assert model_retrying(3).reraise
