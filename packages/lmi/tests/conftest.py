from __future__ import annotations

import logging
import shutil
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx_aiohttp
import litellm.llms.custom_httpx.aiohttp_transport
import pytest
import vcr.stubs.httpx_stubs
from dotenv import load_dotenv
from google.cloud.storage import Client

from lmi.utils import (
    ANTHROPIC_API_KEY_HEADER,
    CROSSREF_KEY_HEADER,
    OPENAI_API_KEY_HEADER,
    SEMANTIC_SCHOLAR_KEY_HEADER,
    filter_api_keys,
    update_litellm_max_callbacks,
)

TESTS_DIR = Path(__file__).parent
CASSETTES_DIR = TESTS_DIR / "cassettes"
STUB_DATA_DIR = TESTS_DIR / "stub_data"


@pytest.fixture(autouse=True, scope="session")
def _load_env() -> None:
    load_dotenv()


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [
            CROSSREF_KEY_HEADER,
            SEMANTIC_SCHOLAR_KEY_HEADER,
            OPENAI_API_KEY_HEADER,
            ANTHROPIC_API_KEY_HEADER,
            "cookie",
        ],
        "before_record_request": filter_api_keys,
        "record_mode": "once",
        "allow_playback_repeats": True,
        "cassette_library_dir": str(CASSETTES_DIR),
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


async def _async_vcr_send(cassette, real_send, *args, **kwargs):  # noqa: ARG001
    """VCR send that only sends, not possibly recording or playing back responses."""
    return await real_send(*args, **kwargs)


# Permanently patch the original response stream,
# to work around https://github.com/karpetrosyan/httpx-aiohttp/issues/23
# and https://github.com/BerriAI/litellm/issues/11724
httpx_aiohttp.transport.AiohttpResponseStream = (  # type: ignore[misc]
    litellm.llms.custom_httpx.aiohttp_transport.AiohttpResponseStream  # type: ignore[misc]
) = PreReadCompatibleAiohttpResponseStream  # type: ignore[assignment]

# Permanently patch vcrpy's async VCR recording functionality,
# to work around https://github.com/kevin1024/vcrpy/issues/944
vcr.stubs.httpx_stubs._async_vcr_send = _async_vcr_send
