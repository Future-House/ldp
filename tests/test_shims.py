import importlib
import os
from unittest.mock import patch

import ldp.shims


def test_tqdm_import() -> None:
    assert ldp.shims.tqdm.__module__ == "tqdm.std"
    with patch.dict(os.environ, {"LDP_TQDM_USE_RICH": "1"}):
        importlib.reload(ldp.shims)
        assert ldp.shims.tqdm.__module__ == "tqdm.rich"
