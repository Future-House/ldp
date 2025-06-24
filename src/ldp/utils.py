import logging
import logging.config
from typing import Any

logger = logging.getLogger(__name__)


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
    config: dict[str, Any] = {name: {"level": level, "handlers": ["stdout"]}}
    if name != "root":  # Non-root loggers need to be in a "loggers" key
        config["loggers"] = config
    logging.config.dictConfig(
        {
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
        }
        | config
    )


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


def format_error_details(error: Exception) -> str:
    """Format detailed error information from an exception.

    Specially handles HTTP errors that have response attributes with status codes
    and JSON details, but works with any exception type.

    Args:
        error: The exception to format

    Returns:
        A formatted error string with available details
    """
    error_details = f"{error!s}"

    if hasattr(error, "response"):
        error_details += f"\nStatus code: {error.response.status_code}"
        try:
            response_data = error.response.json()
            if "detail" in response_data:
                error_details += "\nServer Traceback:\n"
                for line in response_data["detail"].split("\n"):
                    error_details += f"    {line}\n"
        except Exception:
            error_details += f"\nResponse body: {error.response.text}"

    return error_details
