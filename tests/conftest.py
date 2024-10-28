import os
import random
from typing import Any

import numpy as np
import pytest
import torch
from aviary.core import DummyEnv

from ldp.utils import configure_log_levels

from . import CASSETTES_DIR

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(name="dummy_env")
def fixture_dummy_env() -> DummyEnv:
    return DummyEnv()


@pytest.fixture(scope="session", autouse=True)
def _fixture_set_up_environment() -> None:
    configure_log_levels()


def set_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@pytest.fixture(name="seed_zero")
def fixture_seed_zero() -> None:
    """Set a 0 seed to minimize the chances of test flakiness."""
    set_seed(0)


OPENAI_API_KEY_HEADER = "authorization"
ANTHROPIC_API_KEY_HEADER = "x-api-key"
# SEE: https://github.com/kevin1024/vcrpy/blob/v6.0.1/vcr/config.py#L43
VCR_DEFAULT_MATCH_ON = "method", "scheme", "host", "port", "path", "query"


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [OPENAI_API_KEY_HEADER, ANTHROPIC_API_KEY_HEADER, "cookie"],
        "record_mode": "once",
        "match_on": ["method", "host", "path", "query"],
        "allow_playback_repeats": True,
        "cassette_library_dir": str(CASSETTES_DIR),
    }
