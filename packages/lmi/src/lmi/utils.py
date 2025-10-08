import asyncio
import base64
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
    from typing import IO

    import vcr.request
    from PIL._typing import StrOrBytesPath


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
        "loggers": {
            "asyncio": {"level": "WARNING"},  # For selector_events selector
            "httpx": {"level": "WARNING"},
            "httpcore.connection": {"level": "WARNING"},  # For TCP connection events
            "httpcore.http11": {"level": "WARNING"},  # For request send/receive events
        }
        | litellm_loggers_config,
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


def bytes_to_string(value: bytes) -> str:
    """Convert bytes to a base64-encoded string."""
    # 1. Convert bytes to base64 bytes
    # 2. Convert base64 bytes to base64 string,
    #    using UTF-8 since base64 produces ASCII characters
    return base64.b64encode(value).decode("utf-8")


def string_to_bytes(value: str) -> bytes:
    """Convert a base64-encoded string to bytes."""
    # 1. Convert base64 string to base64 bytes (the noqa comment is to make this clear)
    # 2. Convert base64 bytes to original bytes
    return base64.b64decode(value.encode("utf-8"))  # noqa: FURB120


def validate_image(path: "StrOrBytesPath | IO[bytes]") -> None:
    """
    Validate that the file at the given path is a valid image.

    Raises:
        OSError: If the image file is truncated.
    """  # noqa: DOC502
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Image validation requires the 'image' extra for 'pillow'. Please:"
            " `pip install fhlmi[image]`."
        ) from exc

    with Image.open(path) as img:
        img.load()


def encode_image_as_url(image_type: str, image_data: bytes | str) -> str:
    """Convert image data to an RFC 2397 data URL format."""
    if isinstance(image_data, bytes):
        image_data = bytes_to_string(image_data)
    return f"data:image/{image_type};base64,{image_data}"


def is_encoded_image(image: str) -> bool:
    """Check if the given image is a GCS URL or a RFC 2397 data URL."""
    return image.startswith("gs://") or (
        image.startswith("data:image/") and ";base64," in image
    )
