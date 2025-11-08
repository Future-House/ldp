import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import Any, TypeAlias

from aviary.core import Message, is_coroutine_callable
from tqdm.asyncio import tqdm

from ldp.agent import Agent
from ldp.data_structures import TransitionTree

from .callbacks import Callback
from .rollout import (
    AgentError,
    CaughtError,
    EnvError,
    RolloutManager,
    TEnv,
    reraise_exc_as,
)

logger = logging.getLogger(__name__)

TEnvCloneFn: TypeAlias = Callable[[TEnv], Awaitable[TEnv]] | Callable[[TEnv], TEnv]


class TreeSearchRollout(RolloutManager):
    def __init__(
        self,
        agent: Agent,
        branching_factor: int,
        env_clone_fn: TEnvCloneFn,
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
        disable_pbar: bool = False,
    ) -> list[TransitionTree]:
        return await tqdm.gather(
            *[self.sample_tree(env, max_depth) for env in environments],
            desc="Sampling Trees",
            ncols=0,
            disable=disable_pbar,
        )

    async def sample_tree(self, env: TEnv, max_depth: int | None) -> TransitionTree:
        max_depth_f = max_depth if max_depth is not None else float("inf")
        tree = TransitionTree(root_id=str(uuid.uuid4()))

        try:
            await asyncio.gather(*[
                c.before_rollout(tree.root_id, env) for c in self.callbacks
            ])

            with reraise_exc_as(EnvError, enabled=self.catch_env_failures):
                obs, tools = await env.reset()
            await asyncio.gather(*[
                c.after_env_reset(tree.root_id, obs, tools) for c in self.callbacks
            ])

            with reraise_exc_as(AgentError, enabled=self.catch_agent_failures):
                agent_state = await self.agent.init_state(tools)
            await asyncio.gather(*[
                c.after_agent_init_state(tree.root_id, agent_state)
                for c in self.callbacks
            ])
        except CaughtError:
            return tree
        finally:
            # We need to close the env here (before descending) to avoid requiring
            # the resources of this env while making new ones in the next node.
            # Also, we double-suppress EnvError because a tree search crash here kills
            # everything and denies us passing the tree to the next step
            with (
                suppress(EnvError),
                reraise_exc_as(EnvError, enabled=self.catch_env_failures),
            ):
                await env.close()

        await self._descend(
            tree=tree,
            prev_step_id=tree.root_id,
            env=env,
            agent_state=agent_state,
            obs=obs,
            prev_timestep=-1,
            prev_cumulative_reward=0.0,
            max_depth=max_depth_f,
        )

        return tree

    async def _descend(
        self,
        tree: TransitionTree,
        prev_step_id: str,
        env: TEnv,
        agent_state: Any,
        obs: list[Message],
        prev_timestep: int,
        prev_cumulative_reward: float,
        max_depth: float,
    ) -> None:
        # Descend one level in the tree, by adding branching_factor children to the branch
        # Then, recurse on each child

        if tree.root_id in self.target_reward_hit:
            # If at least one branch hit the target reward, stop descending
            return

        timestep = prev_timestep + 1

        async def inner_descend(idx: int) -> None:
            if tree.root_id in self.target_reward_hit:
                # Check again in case the target reward was hit while waiting to start
                # this step
                return

            if is_coroutine_callable(self.env_clone_fn):
                cloned_env = await self.env_clone_fn(env)
            else:
                cloned_env = self.env_clone_fn(env)

            # Descend one step
            step_id = f"{prev_step_id}:{idx}"
            try:
                step = await self._take_step(
                    timestep, step_id, cloned_env, agent_state, obs
                )
            except CaughtError:
                # If we failed, do not extend the branch - just give up on this path
                return

            if timestep + 1 == max_depth and not step.done:
                # Mark as truncated if we hit max_steps and the state is not terminal.
                step.truncated = True

            await asyncio.gather(*[
                callback.after_transition(step_id, self.agent, cloned_env, step)
                for callback in self.callbacks
            ])

            tree.add_transition(step_id, step)

            cumulative_reward = prev_cumulative_reward + step.reward
            if cumulative_reward >= self.target_reward:
                # signal other descents to stop too
                self.target_reward_hit.add(tree.root_id)
                return

            if step.done or step.truncated:
                return

            # Recurse
            await self._descend(
                tree=tree,
                prev_step_id=step_id,
                env=cloned_env,
                agent_state=step.next_agent_state,
                obs=step.next_observation,
                prev_timestep=timestep,
                prev_cumulative_reward=cumulative_reward,
                max_depth=max_depth,
            )

        # Add branching_factor children
        await asyncio.gather(*[
            inner_descend(idx) for idx in range(self.branching_factor)
        ])
