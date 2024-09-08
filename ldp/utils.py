import logging.config

import litellm


def configure_log_levels() -> None:
    """Configure log levels."""
    # Set sane default LiteLLM logging configuration
    # SEE: https://docs.litellm.ai/docs/observability/telemetry
    litellm.telemetry = False

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        # Lower level for httpx and LiteLLM
        "loggers": {
            "httpx": {"level": "WARNING"},
            # SEE: https://github.com/BerriAI/litellm/issues/2256
            "LiteLLM": {"level": "WARNING"},
        },
    })


def configure_stdout_logs(
    stdout_level: int | str = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure root logger to log to stdout.

    Args:
        stdout_level: Log level to be emitted to stdout.
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
        "root": {"level": stdout_level, "handlers": ["stdout"]},
    })
