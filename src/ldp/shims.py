__all__ = ["tqdm", "trange"]

import logging
import os

from tqdm import tqdm, trange

logger = logging.getLogger(__name__)

if os.environ.get("LDP_TQDM_USE_RICH", "").lower() in {"1", "true", "yes"}:
    # TODO: remove after https://github.com/tqdm/tqdm/issues/1618
    try:
        # pylint: disable-next=reimported
        from tqdm.rich import tqdm, trange  # type: ignore[no-redef]
    except ModuleNotFoundError:
        logger.warning(
            "User opted into rich progress via the environment variable"
            " LDP_TQDM_USE_RICH, but did not have 'rich' installed."
            " Please run `pip install ldp[rich]`.",
            exc_info=True,
        )
