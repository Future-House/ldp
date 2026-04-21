import asyncio
import base64
import contextlib
import json
import logging
import logging.config
import os
import re
from collections.abc import Awaitable, Iterable
from typing import TYPE_CHECKING, Any, TypeVar
from urllib.parse import parse_qs, urlencode, urlparse

import litellm
import litellm.constants

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
ANTHROPIC_ORGANIZATION_RESPONSE_HEADER = "anthropic-organization-id"
CROSSREF_KEY_HEADER = "Crossref-Plus-API-Token"
SEMANTIC_SCHOLAR_KEY_HEADER = "x-api-key"
OPENALEX_API_KEY_HEADER = "api_key"
OPENAI_ORGANIZATION_RESPONSE_HEADER = "openai-organization"
OPENAI_PROJECT_RESPONSE_HEADER = "openai-project"
# Scrubbed because Cloudflare sets its `__cf_bm` bot-management cookie on responses
# from vendor APIs served through its edge (e.g. api.openai.com)
# SEE: https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Set-Cookie
# SEE: https://developers.cloudflare.com/fundamentals/reference/policies-compliances/cloudflare-cookies/
SET_COOKIE_RESPONSE_HEADER = "Set-Cookie"
# Clients echo `__cf_bm` back on follow-up requests so Cloudflare can tie them to
# the same session and keep its per-request bot-likelihood score stable
# SEE: https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cookie
# SEE: https://developers.cloudflare.com/bots/concepts/bot-score/
COOKIE_HEADER = "Cookie"
FILTERED = "<FILTERED>"  # Could be header, token, or something else

# Returned in the response body by Google's OAuth 2.0/OpenID Connect token endpoint.
# E.g. when Vertex AI authenticates via a service account or `refresh_token` flow
# SEE: https://developers.google.com/identity/protocols/oauth2/openid-connect
OAUTH_RESPONSE_SECRETS = {"access_token", "id_token"}
# Sent in the POST body to Google's OAuth 2.0 token endpoint
# to authenticate the client and request a new access token
# SEE: https://developers.google.com/identity/protocols/oauth2/web-server#offline
OAUTH_POST_DATA_FILTER: list[tuple[str, str]] = [
    ("client_id", FILTERED),
    ("client_secret", FILTERED),
    ("refresh_token", FILTERED),
]

# SEE: https://github.com/kevin1024/vcrpy/blob/v6.0.1/vcr/config.py#L43
VCR_DEFAULT_MATCH_ON = "method", "scheme", "host", "port", "path", "query"

# Response headers to scrub via VCR `before_record_response`.
# VCR's `filter_headers` config only acts on request headers,
# so these enable scrubbing of response-side headers
# (org/project identifiers, Cloudflare bot-management cookies)
RESPONSE_FILTER_HEADERS: frozenset[str] = frozenset({
    ANTHROPIC_ORGANIZATION_RESPONSE_HEADER,
    OPENAI_ORGANIZATION_RESPONSE_HEADER,
    OPENAI_PROJECT_RESPONSE_HEADER,
    SET_COOKIE_RESPONSE_HEADER,
})


def filter_vcr_response(response: dict) -> dict:
    """Filter sensitive data from VCR response headers and body.

    Scrubs:
    - Response headers listed in `RESPONSE_FILTER_HEADERS`
      (`filter_headers` in `vcr_config` only covers the request side).
    - OAuth tokens (`access_token`, `id_token`) from JSON response bodies.

    Gzipped or otherwise non-UTF-8 bodies are left untouched.
    """
    headers = response.get("headers") or {}
    lowered_targets = {h.lower() for h in RESPONSE_FILTER_HEADERS}
    for header_name in list(headers):  # list() since we mutate
        if header_name.lower() in lowered_targets:
            headers[header_name] = [FILTERED]

    body = response.get("body", {}).get("string")
    if not body:
        return response
    if not isinstance(body, bytes):  # YAGNI
        raise NotImplementedError(f"Didn't yet handle body type {type(body).__name__}.")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return response  # No secrets to scrub when gzip'd or binary response body
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return response  # Body is an HTML string (e.g. not JSON)
    if isinstance(data, dict) and data.keys() & OAUTH_RESPONSE_SECRETS:
        for key in OAUTH_RESPONSE_SECRETS:
            if key in data:
                data[key] = FILTERED
        filtered = json.dumps(data)
        response["body"]["string"] = filtered.encode()
    return response


def filter_api_keys(request: "vcr.request.Request") -> "vcr.request.Request":
    """Filter out API keys from request URI query parameters."""
    parsed_uri = urlparse(request.uri)
    if parsed_uri.query:  # If there's a query that may contain API keys
        query_params = parse_qs(parsed_uri.query)

        # Filter out the Google Gemini API key, if present
        if "key" in query_params:
            query_params["key"] = [FILTERED]

        # Rebuild the URI, with filtered parameters
        filtered_query = urlencode(query_params, doseq=True)
        request.uri = parsed_uri._replace(query=filtered_query).geturl()

    return request


# SEE: https://regex101.com/r/Q461WZ/1
GCP_PROJECT_PATTERN = re.compile(r"/projects/[^/]+/")


def filter_gcp_project(request: "vcr.request.Request") -> "vcr.request.Request":
    """Scrub GCP project IDs from Vertex AI request URIs."""
    request.uri = GCP_PROJECT_PATTERN.sub(f"/projects/{FILTERED}/", request.uri)
    return request


def update_litellm_max_callbacks(value: int = 1000) -> None:
    """Update litellm's MAX_CALLBACKS limit, can call with default to defeat this limit.

    SEE: https://github.com/BerriAI/litellm/issues/9792
    """
    litellm.constants.MAX_CALLBACKS = value


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
