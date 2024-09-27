import logging
import logging.config
import os
from typing import Any

import litellm


def configure_log_levels() -> None:
    """Configure log levels."""
    # Set sane default LiteLLM logging configuration
    # SEE: https://docs.litellm.ai/docs/observability/telemetry
    litellm.telemetry = False
    if (
        logging.getLevelNamesMapping().get(
            os.environ.get("LITELLM_LOG", ""), logging.WARNING
        )
        < logging.WARNING
    ):
        # If LITELLM_LOG is DEBUG or INFO, don't change the LiteLLM log levels
        litellm_loggers_config: dict[str, Any] = {}
    else:
        litellm_loggers_config = {
            "LiteLLM": {"level": "WARNING"},
            "LiteLLM Proxy": {"level": "WARNING"},
            "LiteLLM Router": {"level": "WARNING"},
        }

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        # Lower level for httpx and LiteLLM
        "loggers": {"httpx": {"level": "WARNING"}} | litellm_loggers_config,
    })


def configure_stdout_logs(
    level: int | str = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure root logger to log to stdout.

    Args:
        level: Log level to be emitted to stdout.
        fmt: Optional format string.
    """
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"standard": {"format": fmt}},
        "handlers": {
            "stdout": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {"level": level, "handlers": ["stdout"]},
    })
