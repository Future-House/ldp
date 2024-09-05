import asyncio
import itertools
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from aviary.message import Message
from aviary.utils import is_coroutine_callable

from ldp.agent import Agent
from ldp.alg.callbacks import Callback
from ldp.alg.rollout import (
    AgentError,
    CaughtError,
    EnvError,
    RolloutManager,
    TEnv,
    reraise_exc_as,
)
from ldp.data_structures import Trajectory

logger = logging.getLogger(__name__)


class TreeSearchRollout(RolloutManager):
    def __init__(
        self,
        agent: Agent,
        branching_factor: int,
        env_clone_fn: Callable[[TEnv], Awaitable[TEnv]] | Callable[[TEnv], TEnv],
        catch_agent_failures: bool = True,
        catch_env_failures: bool = True,
        callbacks: Sequence[Callback] | None = None,
        concurrency_limit: int | None = None,
        target_reward: float | None = None,
    ):
        super().__init__(
            agent,
            catch_agent_failures=catch_agent_failures,
            catch_env_failures=catch_env_failures,
            callbacks=callbacks,
            concurrency_limit=concurrency_limit,
        )

        self.branching_factor = branching_factor
        self.target_reward = (
            target_reward if target_reward is not None else float("inf")
        )
        self.target_reward_hit: set[str] = set()

        self.env_clone_fn = env_clone_fn

    async def sample_trees(
        self,
        environments: Sequence[TEnv],
        max_depth: int | None = None,
    ) -> list[list[Trajectory]]:
        return await asyncio.gather(*[
            self.sample_tree(env, max_depth) for env in environments
        ])

    async def sample_tree(self, env: TEnv, max_depth: int | None) -> list[Trajectory]:
        max_depth_f = max_depth if max_depth is not None else float("inf")

        try:
            with reraise_exc_as(EnvError, enabled=self.catch_env_failures):
                obs, tools = await env.reset()

            with reraise_exc_as(AgentError, enabled=self.catch_agent_failures):
                agent_state = await self.agent.init_state(tools)

            root_traj = Trajectory(traj_id=str(uuid.uuid4()))
            return await self._descend(root_traj, env, agent_state, obs, max_depth_f)

        except CaughtError:
            return []

    async def _descend(
        self,
        branch: Trajectory,
        env: TEnv,
        agent_state: Any,
        obs: list[Message],
        max_depth: float,
    ) -> list[Trajectory]:
        # Descend one level in the tree, by adding branching_factor children to the branch
        # Then, recurse on each child
        root_traj_id = cast(str, branch.traj_id).split(":")[0]
        if root_traj_id in self.target_reward_hit:
            return [branch]

        timestep = len(branch.steps)

        async def inner_descend(idx: int) -> list[Trajectory]:
            if is_coroutine_callable(self.env_clone_fn):
                cloned_env = await self.env_clone_fn(env)  # type: ignore[arg-type, misc]
            else:
                cloned_env = self.env_clone_fn(env)  # type: ignore[arg-type]

            # Descend one step
            traj_id = f"{branch.traj_id}:{idx}"
            try:
                step = await self._take_step(
                    timestep, traj_id, cloned_env, agent_state, obs
                )
            except CaughtError:
                # If we failed, do not extend the branch - just return an empty list
                return []

            await asyncio.gather(*[
                callback.after_transition(traj_id, self.agent, cloned_env, step)
                for callback in self.callbacks
            ])

            # The original branch plus one step
            extended_branch = Trajectory(traj_id=traj_id, steps=[*branch.steps, step])

            if (
                step.done  # Trajectory is over
                or len(extended_branch.steps) >= max_depth  # Hit max depth
            ):
                return [extended_branch]

            if (
                sum(step_.reward for step_ in extended_branch.steps)
                >= self.target_reward
            ):
                # signal other descents to stop too
                self.target_reward_hit.add(root_traj_id)
                return [extended_branch]

            # Recurse
            return await self._descend(
                extended_branch,
                cloned_env,
                step.next_agent_state,
                step.next_observation,
                max_depth,
            )

        # Add branching_factory children
        branches = await asyncio.gather(*[
            inner_descend(idx) for idx in range(self.branching_factor)
        ])

        return list(itertools.chain.from_iterable(branches))
