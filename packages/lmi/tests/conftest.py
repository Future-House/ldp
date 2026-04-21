import logging
import shutil
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx_aiohttp
import litellm.llms.custom_httpx.aiohttp_transport
import pytest
import vcr.request
import vcr.stubs.httpcore_stubs
from dotenv import load_dotenv
from google.cloud.storage import Client

from lmi.utils import (
    ANTHROPIC_API_KEY_HEADER,
    ANTHROPIC_ORGANIZATION_HEADER,
    COOKIE_HEADER,
    CROSSREF_KEY_HEADER,
    OAUTH_POST_DATA_FILTER,
    OPENAI_API_KEY_HEADER,
    OPENAI_ORGANIZATION_HEADER,
    OPENAI_PROJECT_HEADER,
    SEMANTIC_SCHOLAR_KEY_HEADER,
    SET_COOKIE_HEADER,
    filter_api_keys,
    filter_gcp_project,
    filter_vcr_response,
    update_litellm_max_callbacks,
)

TESTS_DIR = Path(__file__).parent
CASSETTES_DIR = TESTS_DIR / "cassettes"
STUB_DATA_DIR = TESTS_DIR / "stub_data"


@pytest.fixture(autouse=True, scope="session")
def _load_env() -> None:
    load_dotenv()


def _filter_vcr_request(request: vcr.request.Request) -> vcr.request.Request:
    """Scrub Gemini API keys (query params) and GCP project IDs from request URIs."""
    return filter_gcp_project(filter_api_keys(request))


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [
            CROSSREF_KEY_HEADER,
            SEMANTIC_SCHOLAR_KEY_HEADER,
            OPENAI_API_KEY_HEADER,
            OPENAI_ORGANIZATION_HEADER,
            OPENAI_PROJECT_HEADER,
            ANTHROPIC_API_KEY_HEADER,
            ANTHROPIC_ORGANIZATION_HEADER,
            SET_COOKIE_HEADER,
            COOKIE_HEADER,
        ],
        "filter_post_data_parameters": OAUTH_POST_DATA_FILTER,
        "before_record_request": _filter_vcr_request,
        "before_record_response": filter_vcr_response,
        "record_mode": "once",
        "allow_playback_repeats": True,
        "cassette_library_dir": str(CASSETTES_DIR),
        # "drop_unused_requests": True,  # Restore after https://github.com/kevin1024/vcrpy/issues/961
    }


@pytest.fixture(autouse=True, scope="session")
def _defeat_litellm_callbacks() -> None:
    update_litellm_max_callbacks()


@pytest.fixture
def tmp_path_cleanup(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path
    # Cleanup after the test
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture(scope="session", name="stub_data_dir")
def fixture_stub_data_dir() -> Path:
    return Path(__file__).parent / "stub_data"


@pytest.fixture(name="reset_log_levels")
def fixture_reset_log_levels(caplog) -> Iterator[None]:
    logging.getLogger().setLevel(logging.DEBUG)

    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True

    caplog.set_level(logging.DEBUG)

    yield

    for name in logging.root.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


@pytest.fixture(name="png_image", scope="session")
def fixture_png_image() -> bytes:
    with (STUB_DATA_DIR / "sf_districts.png").open("rb") as f:
        return f.read()


TMP_LMI_TEST_GCS_BUCKET = "tmp-lmi-test"


@pytest.fixture(name="png_image_gcs", scope="session")
def fixture_png_image_gcs(png_image: bytes) -> str:
    """Get or create a temporary GCS bucket, upload test image, and return GCS URL."""
    client = Client()
    bucket = client.bucket(TMP_LMI_TEST_GCS_BUCKET)
    if not bucket.exists():  # Get or create the bucket
        bucket = client.create_bucket(bucket)
    blob_name = "sf_districts.png"
    blob = bucket.blob(blob_name)
    if not blob.exists(client):
        blob.upload_from_string(png_image, content_type="image/png")
    return f"gs://{TMP_LMI_TEST_GCS_BUCKET}/{blob_name}"


class PreReadCompatibleAiohttpResponseStream(
    httpx_aiohttp.transport.AiohttpResponseStream
):
    """aiohttp-backed response stream that works if the response was pre-read."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        with httpx_aiohttp.transport.map_aiohttp_exceptions():
            if self._aiohttp_response._body is not None:
                # Happens if some intermediary called `await _aiohttp_response.read()`
                # TODO: take into account chunk size
                yield self._aiohttp_response._body
            else:
                async for chunk in self._aiohttp_response.content.iter_chunked(
                    self.CHUNK_SIZE
                ):
                    yield chunk


async def _vcr_handle_async_request(
    cassette,  # noqa: ARG001
    real_handle_async_request,
    self,
    real_request,
):
    """VCR handler that only sends, not possibly recording or playing back responses."""
    return await real_handle_async_request(self, real_request)


# Permanently patch the original response stream,
# to work around https://github.com/karpetrosyan/httpx-aiohttp/issues/23
# and https://github.com/BerriAI/litellm/issues/11724
httpx_aiohttp.transport.AiohttpResponseStream = (  # type: ignore[misc]
    litellm.llms.custom_httpx.aiohttp_transport.AiohttpResponseStream  # type: ignore[misc]
) = PreReadCompatibleAiohttpResponseStream  # type: ignore[assignment]

# Permanently patch vcrpy's async VCR recording functionality,
# to work around https://github.com/kevin1024/vcrpy/issues/944
vcr.stubs.httpcore_stubs._vcr_handle_async_request = _vcr_handle_async_request
