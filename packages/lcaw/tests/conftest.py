from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# from unittest.mock import patch
import pytest
from dotenv import load_dotenv

TESTS_DIR = Path(__file__).parent
CASSETTES_DIR = TESTS_DIR / "cassettes"


@pytest.fixture(autouse=True, scope="session")
def _load_env() -> None:
    load_dotenv()


OPENAI_API_KEY_HEADER = "authorization"
ANTHROPIC_API_KEY_HEADER = "x-api-key"
CROSSREF_HEADER_KEY = "Crossref-Plus-API-Token"
SEMANTIC_SCHOLAR_HEADER_KEY = "x-api-key"
# SEE: https://github.com/kevin1024/vcrpy/blob/v6.0.1/vcr/config.py#L43
VCR_DEFAULT_MATCH_ON = "method", "scheme", "host", "port", "path", "query"


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [
            CROSSREF_HEADER_KEY,
            SEMANTIC_SCHOLAR_HEADER_KEY,
            OPENAI_API_KEY_HEADER,
            ANTHROPIC_API_KEY_HEADER,
            "cookie",
        ],
        "record_mode": "once",
        "allow_playback_repeats": True,
        "cassette_library_dir": str(CASSETTES_DIR),
    }


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
