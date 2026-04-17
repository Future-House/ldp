"""Retry and fallback policy for LMI.

LMI attempts each model in an `LLMConfig.models` chain. Within a single model
it retries on transient errors (rate limits, timeouts, 5xx, transport blips)
using `BACKOFF_INITIAL` / `BACKOFF_CAP` exponential backoff with full jitter.
When retries are exhausted or the error indicates that another model might
succeed, it advances to the next entry in the chain.

Classification is delegated to litellm's exception hierarchy: `_RETRYABLE`
covers transient failures; `_FALLBACKABLE` covers errors that are stable
against retry but may resolve on a different model.
"""

from __future__ import annotations

import random

import litellm

from lmi.exceptions import ModelRefusalError

BACKOFF_INITIAL = 1.0
BACKOFF_CAP = 30.0


_RETRYABLE: tuple[type[BaseException], ...] = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)

_FALLBACKABLE: tuple[type[BaseException], ...] = (
    litellm.ContextWindowExceededError,
    litellm.NotFoundError,
    litellm.ContentPolicyViolationError,
    # Auth/permission failures: when a user configures a fallback chain, they
    # likely have distinct credentials per entry, so a 401/403 on one model
    # shouldn't block the others.
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    # Providers reject requests for schema reasons that differ across
    # providers (e.g. image count limits, unsupported field combinations); a
    # sibling model may accept the same input.
    litellm.BadRequestError,
    ModelRefusalError,
)


def should_retry(exc: BaseException) -> bool:
    """True if the same model might succeed on another try."""
    return isinstance(exc, _RETRYABLE)


def should_fallback(exc: BaseException) -> bool:
    """True if another model might succeed where this one failed."""
    return isinstance(exc, _FALLBACKABLE)


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter. `attempt` is 0-indexed."""
    ceiling = min(BACKOFF_CAP, BACKOFF_INITIAL * (2**attempt))
    return random.uniform(0, ceiling)
