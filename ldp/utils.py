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
    name: str = "root",
    level: int | str = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure root logger to log to stdout.

    Args:
        name: Optional logger name, if unspecified the 'root' logger is configured.
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
        name: {"level": level, "handlers": ["stdout"]},  # type: ignore[misc]
    })


def discounted_returns(
    rewards: list[float], terminated: list[bool], discount: float = 1.0
) -> list[float]:
    r"""
    Calculate the discounted returns for a list of rewards, considering termination flags and a discount factor.

    The discounted return represents the future discounted rewards from each time step onwards, taking into account
    whether an episode has terminated at each step.

    The discounted return \( G_t \) is given by:

    .. math::
        G_t = \sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k}

        where:
        - \( G_t \) is the discounted return starting from time step \( t \).
        - \( \gamma \) is the discount factor.
        - \( R_{t+k} \) is the reward received at time step \( t+k \).

    NOTE: this could live in ldp.alg, but it's here to avoid circular imports.

    Args:
        rewards: A list of rewards at each time step.
        terminated: A list of boolean flags indicating whether the episode terminated at each time step.
        discount: Discount factor to apply to future rewards. Defaults to 1.0 which means no discounting is applied.

    Returns:
        A list of discounted returns (rewards to go), with each element representing the
            total discounted reward from that step onwards.

    Example:
        >>> rewards = [1.0, 2.0, 3.0]
        >>> terminated = [False, False, True]
        >>> discounted_returns(rewards, terminated, discount=0.9)
        [5.23, 4.7, 3.0]
    """
    returns = []
    r = 0.0
    for reward, term in zip(reversed(rewards), reversed(terminated), strict=False):
        # 1 - term is 0 if the episode has terminated
        r = reward + discount * r * (1 - term)
        returns.append(r)
    returns.reverse()
    return returns
