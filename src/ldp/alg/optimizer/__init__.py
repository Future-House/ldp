from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

from ldp.agent import Agent, MemoryAgent, ReActAgent
from ldp.alg.optimizer.ape import APEOpt, APEScoreFn
from ldp.alg.optimizer.memory import MemoryFactory, MemoryOpt, PositiveMemoryOpt
from ldp.alg.optimizer.opt import _OPTIMIZER_REGISTRY, ChainedOptimizer, Optimizer

_DEFAULT_OPTIMIZER_ERROR_MSG = (
    "Didn't yet implement an optimizer of type {opt_type} for {agent_type}."
)
AUTOSELECT_BEST_OPTIMIZER = None


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    optimizer_type: str | None = AUTOSELECT_BEST_OPTIMIZER
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)


_DEFAULT_OPTIMIZER_MAP: dict[type[Agent], type[Optimizer]] = {
    MemoryAgent: MemoryOpt,
    ReActAgent: APEOpt,
}


def default_optimizer_factory(
    agent: Agent,
    optimizer_cls: str | type[Optimizer] | None = AUTOSELECT_BEST_OPTIMIZER,
    **optimizer_kwargs,
) -> Optimizer:
    """A method that constructs a default optimizer for commonly-used agents.

    Args:
        agent: Agent to construct the optimizer for.
        optimizer_cls: The optimizer class to use. If not specified, we will try a
            default based on the provided agent.
        optimizer_kwargs: Arguments forwarded to optimizer_cls.

    Returns:
        Instantiated optimizer.
    """
    if isinstance(optimizer_cls, str):
        try:
            optimizer_cls = _OPTIMIZER_REGISTRY[optimizer_cls]
        except KeyError:
            raise TypeError(
                "Optimizer class not supported by default_optimizer_factory:"
                f" {optimizer_cls}"
            ) from None

    if optimizer_cls is None:
        optimizer_cls = _DEFAULT_OPTIMIZER_MAP.get(agent.__class__)

    # convince mypy that optimizer_cls is a type from here on
    optimizer_cls = cast("type", optimizer_cls)

    if isinstance(agent, MemoryAgent):
        if optimizer_cls != MemoryOpt:
            raise NotImplementedError(
                _DEFAULT_OPTIMIZER_ERROR_MSG.format(
                    opt_type=optimizer_cls.__name__, agent_type=MemoryAgent.__name__
                )
            )
        return MemoryOpt.from_agent(agent, **optimizer_kwargs)
    if isinstance(agent, ReActAgent):
        if optimizer_cls != APEOpt:
            raise NotImplementedError(
                _DEFAULT_OPTIMIZER_ERROR_MSG.format(
                    opt_type=optimizer_cls.__name__, agent_type=ReActAgent.__name__
                )
            )
        return APEOpt.from_agent(agent, **optimizer_kwargs)
    raise TypeError(f"Unsupported agent type: {agent.__class__.__name__}")


__all__ = [
    "APEOpt",
    "APEScoreFn",
    "ChainedOptimizer",
    "MemoryFactory",
    "MemoryOpt",
    "Optimizer",
    "OptimizerConfig",
    "PositiveMemoryOpt",
    "default_optimizer_factory",
]
