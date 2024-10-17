import argparse
import asyncio
import pickle
from os import PathLike
from pathlib import Path

from aviary.env import Environment

from ldp.agent import Agent
from ldp.alg.callbacks import TerminalLoggingCallback
from ldp.alg.rollout import RolloutManager


def agent_factory(agent: Agent | str | PathLike) -> Agent:
    if isinstance(agent, Agent):
        return agent

    if isinstance(agent, str):
        try:
            return Agent.from_name(agent)
        except KeyError:
            pass

    path = Path(agent)
    if not path.exists():
        raise ValueError(f"Could not resolve agent: {agent}")

    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def environment_factory(environment: Environment | str, task: str) -> Environment:
    if isinstance(environment, Environment):
        return environment

    if isinstance(environment, str):
        try:
            return Environment.from_name(environment, task=task)
        except ValueError:
            pass

    raise ValueError(
        f"Could not resolve environment: {environment}. Available environments: {Environment.available()}"
    )


async def main(
    task: str,
    environment: Environment | str,
    agent: Agent | str | PathLike = "SimpleAgent",
):
    agent = agent_factory(agent)

    callback = TerminalLoggingCallback()
    rollout_manager = RolloutManager(agent=agent, callbacks=[callback])

    _ = await rollout_manager.sample_trajectories(
        environment_factory=lambda: environment_factory(environment, task)
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
