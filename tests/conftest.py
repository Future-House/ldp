import os
import random
from typing import Any

import numpy as np
import pytest
import torch
from aviary.core import DummyEnv
from lmi import configure_llm_logs
from lmi.utils import update_litellm_max_callbacks

from ldp.nn.handlers.transformer_handler import ExecutionMode, ParallelModeConfig

from . import CASSETTES_DIR

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(name="dummy_env")
def fixture_dummy_env() -> DummyEnv:
    return DummyEnv()


@pytest.fixture(scope="session", autouse=True)
def _fixture_set_up_environment() -> None:
    configure_llm_logs()


@pytest.fixture(autouse=True, scope="session")
def _defeat_litellm_callbacks() -> None:
    update_litellm_max_callbacks()


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


ENABLED = {"true", "1", "yes"}
TEST_GPUS: bool = os.getenv("TEST_GPUS", "").lower() in ENABLED
TEST_SLURM: bool = os.getenv("TEST_SLURM", "").lower() in ENABLED

PARALLEL_MODE_CONFIGS = [
    pytest.param(None, id="cpu-only"),
    pytest.param(
        ParallelModeConfig(num_workers=2, num_cpus_per_worker=1),
        id="two-gpu",
        marks=pytest.mark.skipif(not TEST_GPUS, reason="Requires GPUs"),
    ),
    pytest.param(
        ParallelModeConfig(num_workers=2, num_cpus_per_worker=1, offload_cpu=True),
        id="two-gpu-offload",
        marks=pytest.mark.skipif(not TEST_GPUS, reason="Requires GPUs"),
    ),
    pytest.param(
        ParallelModeConfig(
            num_workers=2,
            num_cpus_per_worker=1,
            offload_cpu=True,
            activation_checkpointing=True,
            cpu_ram_efficient_loading=True,
        ),
        id="two-gpu-all-enabled",
        marks=pytest.mark.skipif(not TEST_GPUS, reason="Requires GPUs"),
    ),
    pytest.param(
        ParallelModeConfig(
            num_workers=2,
            num_cpus_per_worker=1,
            execution_mode=ExecutionMode.SLURM_CLUSTER,
        ),
        id="two-gpus-slurm",
        marks=pytest.mark.skipif(
            not TEST_GPUS or not TEST_SLURM, reason="Requires GPUs and SLURM"
        ),
    ),
    pytest.param(
        ParallelModeConfig(
            num_workers=2,
            num_cpus_per_worker=1,
            execution_mode=ExecutionMode.SLURM_CLUSTER,
            offload_cpu=True,
        ),
        id="two-gpus-slurm-offload",
        marks=pytest.mark.skipif(
            not TEST_GPUS or not TEST_SLURM, reason="Requires GPUs and SLURM"
        ),
    ),
]
