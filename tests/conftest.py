import os
import random

import numpy as np
import pytest
import torch
from aviary.env import DummyEnv

IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(name="dummy_env")
def fixture_dummy_env() -> DummyEnv:
    return DummyEnv()


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
