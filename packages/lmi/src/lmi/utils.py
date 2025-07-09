import asyncio
import contextlib
import logging
import logging.config
import os
from collections.abc import Awaitable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import parse_qs, urlencode, urlparse

import litellm

try:
    from tqdm.asyncio import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    import vcr.request


def configure_llm_logs() -> None:
    """Configure log levels."""
    # Set sane default LiteLLM logging configuration
    # SEE: https://docs.litellm.ai/docs/observability/telemetry
    litellm.telemetry = False
    if (
        logging.getLevelNamesMapping().get(
            os.environ.get("LITELLM_LOG", ""), logging.WARNING
        )
        < logging.WARNING
    ):
        # If LITELLM_LOG is DEBUG or INFO, don't change the LiteLLM log levels
        litellm_loggers_config: dict[str, Any] = {}
    else:
        litellm_loggers_config = {
            "LiteLLM": {"level": "WARNING"},
            "LiteLLM Proxy": {"level": "WARNING"},
            "LiteLLM Router": {"level": "WARNING"},
        }

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        # Lower level for httpx and LiteLLM
        "loggers": {"httpx": {"level": "WARNING"}} | litellm_loggers_config,
    })


def get_litellm_retrying_config(timeout: float = 60.0) -> dict[str, Any]:
    """Get retrying configuration for litellm.acompletion and litellm.aembedding."""
    return {"num_retries": 3, "timeout": timeout}


def partial_format(value: str, **formats: dict[str, Any]) -> str:
    """Partially format a string given a variable amount of formats."""
    for template_key, template_value in formats.items():
        with contextlib.suppress(KeyError):
            value = value.format(**{template_key: template_value})
    return value


T = TypeVar("T")


async def gather_with_concurrency(
    n: int | asyncio.Semaphore, coros: Iterable[Awaitable[T]], progress: bool = False
) -> list[T]:
    """
    Run asyncio.gather with a concurrency limit.

    SEE: https://stackoverflow.com/a/61478547/2392535
    """
    semaphore = asyncio.Semaphore(n) if isinstance(n, int) else n

    async def sem_coro(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    if progress:
        try:
            return await tqdm.gather(
                *(sem_coro(c) for c in coros), desc="Gathering", ncols=0
            )
        except AttributeError:
            raise ImportError(
                "Gathering with a progress bar requires 'tqdm' as a dependency, which"
                " is in the 'progress' extra."
                " Please run `pip install lmi[progress]`."
            ) from None
    return await asyncio.gather(*(sem_coro(c) for c in coros))


OPENAI_API_KEY_HEADER = "authorization"
ANTHROPIC_API_KEY_HEADER = "x-api-key"
CROSSREF_KEY_HEADER = "Crossref-Plus-API-Token"
SEMANTIC_SCHOLAR_KEY_HEADER = "x-api-key"

# SEE: https://github.com/kevin1024/vcrpy/blob/v6.0.1/vcr/config.py#L43
VCR_DEFAULT_MATCH_ON = "method", "scheme", "host", "port", "path", "query"


def filter_api_keys(request: "vcr.request.Request") -> "vcr.request.Request":
    """Filter out API keys from request URI query parameters."""
    parsed_uri = urlparse(request.uri)
    if parsed_uri.query:  # If there's a query that may contain API keys
        query_params = parse_qs(parsed_uri.query)

        # Filter out the Google Gemini API key, if present
        if "key" in query_params:
            query_params["key"] = ["<FILTERED>"]

        # Rebuild the URI, with filtered parameters
        filtered_query = urlencode(query_params, doseq=True)
        request.uri = parsed_uri._replace(query=filtered_query).geturl()

    return request


def update_litellm_max_callbacks(value: int = 1000) -> None:
    """Update litellm's MAX_CALLBACKS limit, can call with default to defeat this limit.

    SEE: https://github.com/BerriAI/litellm/issues/9792
    """
    litellm.litellm_core_utils.logging_callback_manager.LoggingCallbackManager.MAX_CALLBACKS = value
