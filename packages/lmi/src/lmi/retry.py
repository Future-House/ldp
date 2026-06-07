"""Retry and fallback policy for LMI.

LMI attempts each model in an `LLMConfig.models` chain. Within a single model
it retries on transient errors (rate limits, timeouts, 5xx, transport blips)
using full-jitter exponential backoff (`model_retrying`, built on tenacity).
When retries are exhausted or the error indicates that another model might
succeed, it advances to the next entry in the chain.

Classification is delegated to litellm's exception hierarchy: `_RETRYABLE`
covers transient failures; `should_fallback` covers errors that are stable
against retry but may resolve on a different model.
"""

import logging

import litellm
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from lmi.exceptions import ModelRefusalError, ResponseValidationError

logger = logging.getLogger(__name__)

BACKOFF_INITIAL = 1.0
BACKOFF_CAP = 30.0


_RETRYABLE: tuple[type[BaseException], ...] = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.BadGatewayError,
    ResponseValidationError,
)

# 400s that are genuinely provider-specific (a sibling model may accept the same
# input) rather than a malformed request. litellm surfaces these as a plain
# `BadRequestError`, so we match on the message; see `should_fallback`.
_PROVIDER_LIMIT_PATTERNS = ("too much media",)

# Content-safety refusals that litellm surfaces as a bare `BadRequestError`
# instead of `ContentPolicyViolationError` (e.g. OpenAI's prompt-level safety
# block, error code "invalid_prompt"). A refusal can be a false positive, so a
# sibling model may legitimately answer — fall over rather than dead-end.
_CONTENT_REFUSAL_PATTERNS = ("limited access to this content for safety reasons",)

_FALLBACKABLE: tuple[type[BaseException], ...] = (
    litellm.NotFoundError,
    # Auth/permission failures: when a user configures a fallback chain, they
    # likely have distinct credentials per entry, so a 401/403 on one model
    # shouldn't block the others.
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    # Context-window and content-policy rejections differ across providers, so a
    # sibling model may accept the same input. Both subclass `BadRequestError`;
    # a *generic* `BadRequestError` is treated as a terminal client error (see
    # `should_fallback`) and propagates rather than falling over.
    litellm.ContextWindowExceededError,
    litellm.ContentPolicyViolationError,
    ModelRefusalError,
)


def should_retry(exc: BaseException) -> bool:
    """True if the same model might succeed on another try."""
    return isinstance(exc, _RETRYABLE)


def _response_status(exc: BaseException) -> int | None:
    """Best-effort recovery of the upstream HTTP status.

    litellm hardcodes ``BadRequestError.status_code = 400``, so the real status
    of a mislabelled error (e.g. a Vertex 403 caught by a string branch before
    litellm's status-code branch) survives only on the attached response.
    """
    return getattr(getattr(exc, "response", None), "status_code", None)


def should_fallback(exc: BaseException) -> bool:
    """True if another model might succeed where this one failed.

    Provider-specific 400s (e.g. Anthropic's 100-image limit) fall over to a
    sibling model, but a generic `BadRequestError` is a terminal client error
    (malformed request) and propagates. Auth/permission/not-found failures that
    litellm mislabels as a bare `BadRequestError` are recovered via the response
    status and treated as fall-over-able.
    """
    if isinstance(exc, _FALLBACKABLE):
        return True
    if isinstance(exc, litellm.BadRequestError):
        # litellm sometimes collapses auth/permission/not-found failures into a
        # bare BadRequestError instead of the dedicated subclass (e.g. a Vertex
        # 403 whose body contains "403" matches a string branch before the
        # status-code branch in exception_mapping_utils.py). Recover the true
        # status and fall over on these, mirroring the AuthenticationError /
        # PermissionDeniedError / NotFoundError entries already in _FALLBACKABLE.
        if _response_status(exc) in {401, 403, 404}:
            return True
        message = str(exc).lower()
        return any(
            pattern in message
            for pattern in (*_PROVIDER_LIMIT_PATTERNS, *_CONTENT_REFUSAL_PATTERNS)
        )
    return False


def model_retrying(max_retries: int) -> AsyncRetrying:
    """Tenacity policy for retries *within a single model*.

    `max_retries` retries after the first attempt, i.e. `max_retries + 1` total
    attempts, with full-jitter exponential backoff between them. Retries only on
    `_RETRYABLE` exceptions; with `reraise=True` the final failure propagates so
    the caller can decide whether to fall over to the next model.
    """
    return AsyncRetrying(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_random_exponential(multiplier=BACKOFF_INITIAL, max=BACKOFF_CAP),
        retry=retry_if_exception(should_retry),
        before_sleep=before_sleep_log(logger, logging.INFO),
        reraise=True,
    )
