import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


REPO_ROOT = Path(__file__).parent.parent.parent
