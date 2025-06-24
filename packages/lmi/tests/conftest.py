from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

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
