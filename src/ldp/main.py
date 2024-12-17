import argparse
import asyncio
import pickle
from contextlib import suppress
from os import PathLike
from pathlib import Path

from aviary.core import Environment

from ldp.agent import Agent
from ldp.alg.callbacks import TerminalPrintingCallback
from ldp.alg.rollout import RolloutManager


def get_or_make_agent(agent: Agent | str | PathLike) -> Agent:
    if isinstance(agent, Agent):
        return agent

    if isinstance(agent, str):
        with suppress(KeyError):
            return Agent.from_name(agent)

    path = Path(agent)
    if not path.exists():
        raise ValueError(f"Could not resolve agent: {agent}")

    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def get_or_make_environment(environment: Environment | str, task: str) -> Environment:
    if isinstance(environment, Environment):
        return environment

    if isinstance(environment, str):
        with suppress(KeyError):
            return Environment.from_name(environment, task=task)

    raise ValueError(
        f"Could not resolve environment: {environment}. Available environments:"
        f" {Environment.available()}"
    )


async def main(
    task: str,
    environment: Environment | str,
    agent: Agent | str | PathLike = "SimpleAgent",
):
    agent = get_or_make_agent(agent)

    callback = TerminalPrintingCallback()
    rollout_manager = RolloutManager(agent=agent, callbacks=[callback])

    _ = await rollout_manager.sample_trajectories(
        environment_factory=lambda: get_or_make_environment(environment, task)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", help="Task to prompt environment with.")
    parser.add_argument(
        "--env", required=True, help="Environment to sample trajectories from."
    )
    parser.add_argument(
        "--agent", default="SimpleAgent", help="Agent to sample trajectories with."
    )
    args = parser.parse_args()

    asyncio.run(main(args.task, args.env, args.agent))
