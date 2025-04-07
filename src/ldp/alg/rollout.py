import asyncio
import itertools
import logging
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, TypeVar, overload

from aviary.core import Environment, Message

from ldp.agent import Agent
from ldp.data_structures import Trajectory, Transition
from ldp.utils import format_error_details

from .callbacks import Callback

logger = logging.getLogger(__name__)


TEnv = TypeVar("TEnv", bound=Environment)


class CaughtError(Exception):
    """Base class for reraised exceptions when catching is enabled."""

    def __init__(self, original_exc: Exception):
        self.original_exc = original_exc

    exc_type = "undefined"


class AgentError(CaughtError):
    exc_type = "agent"


class EnvError(CaughtError):
    exc_type = "env"


@contextmanager
def reraise_exc_as(reraise: type[CaughtError], enabled: bool) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        if enabled:
            error_details = format_error_details(e)
            logger.exception(f"Caught {reraise.exc_type} exception:\n{error_details}")
            raise reraise(e) from None
        raise


class RolloutManager:
    def __init__(
        self,
        agent: Agent,
        catch_agent_failures: bool = True,
        catch_env_failures: bool = True,
        callbacks: Sequence[Callback] | None = None,
        concurrency_limit: int | None = None,
    ):
        self.agent = agent

        self.catch_agent_failures = catch_agent_failures
        self.catch_env_failures = catch_env_failures

        self.concurrency_limiter = (
            asyncio.Semaphore(concurrency_limit) if concurrency_limit else nullcontext()
        )

        self.traj_buffer: dict[str, Trajectory] = {}
        self.callbacks = callbacks or []

    @overload
    async def sample_trajectories(  # noqa: D418
        self,
        environment_factory: Callable[[], TEnv],
        batch_size: int = 1,
        max_steps: int | None = None,
    ) -> list[tuple[Trajectory, TEnv]]:
        """Run rollouts in parallel, using a factory to construct environments.

        We will construct `batch_size` environments and run rollouts on each of them.
        If `max_steps` is set, rollouts will be truncated at this value. If a rollout
        has fewer than `max_steps`, then a new environment will be constructed and another
        rollout will be started until `max_steps` is reached.

        Args:
            environment_factory: A no-argument callable that returns
                an environment instance
            batch_size (int, optional): Defaults to 1.
            max_steps (int | None, optional): Max steps per rollout. Defaults to None (see above).

        Returns:
            list[tuple[Trajectory, Environment]]: A list of (trajectory, environment) tuples: one per rollout.
        """

    @overload
    async def sample_trajectories(  # noqa: D418
        self,
        environments: Sequence[Environment],
        max_steps: int | None = None,
    ) -> list[Trajectory]:
        """Run rollouts in parallel on a list of provided environments.

        Args:
            environments: A list of environments to run rollouts on.
            max_steps: Max steps per rollout. Defaults to None, in which case the rollouts are run
                until environment returns done.
        """

    async def sample_trajectories(self, **kwargs):
        if "environment_factory" in kwargs:
            assert "environments" not in kwargs, (
                "Cannot use environment_factory with environments"
            )

            return await self._sample_trajectories_from_env_factory(
                kwargs["environment_factory"],
                kwargs.get("batch_size", 1),
                kwargs.get("max_steps"),
            )

        if "environments" in kwargs:
            assert "environment_factory" not in kwargs, (
                "Cannot use environments with environment_factory"
            )
            return await self._sample_trajectories_from_envs(
                kwargs["environments"], kwargs.get("max_steps")
            )

        raise TypeError(
            "sample_trajectories() missing required "
            "arguments 'environment_factory' or 'environments'"
        )

    async def _sample_trajectories_from_env_factory(
        self,
        environment_factory: Callable[[], Environment],
        batch_size: int = 1,
        max_steps: int | None = None,
    ) -> list[tuple[Trajectory, Environment]]:
        self.traj_buffer.clear()

        async def rollout_with_args(idx: int, **rollout_kwargs):
            return idx, await self._rollout(**rollout_kwargs), rollout_kwargs

        accumulated_steps = [0] * batch_size
        # submit initial batch of tasks
        tasks = [
            asyncio.create_task(
                rollout_with_args(
                    idx,
                    traj_id=uuid.uuid4().hex,
                    env=environment_factory(),
                    max_steps=max_steps,
                )
            )
            for idx in range(batch_size)
        ]

        results = []
        while tasks:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            new_tasks = []
            for task in done:
                idx, traj, kwargs = await task
                results.append((traj, kwargs["env"]))
                accumulated_steps[idx] += len(traj.steps)
                if (
                    max_steps is not None
                    and (remaining_steps := max_steps - accumulated_steps[idx]) > 0
                ):
                    # submit another task if we haven't reached max_steps
                    new_task = asyncio.create_task(
                        rollout_with_args(
                            idx,
                            traj_id=uuid.uuid4().hex,
                            env=environment_factory(),
                            max_steps=remaining_steps,
                        )
                    )
                    new_tasks.append(new_task)

            tasks = list(pending) + new_tasks

        return results

    async def _sample_trajectories_from_envs(
        self,
        environments: Sequence[Environment],
        max_steps: int | None = None,
    ) -> list[Trajectory]:
        self.traj_buffer.clear()

        traj_ids = [uuid.uuid4().hex for _ in range(len(environments))]
        await asyncio.gather(
            *(
                self._rollout(*args, max_steps=max_steps)
                for args in zip(traj_ids, environments, strict=True)
            )
        )
        return [self.traj_buffer[traj_id] for traj_id in traj_ids]

    async def _rollout(
        self,
        traj_id: str,
        env: Environment,
        max_steps: int | None,
    ) -> Trajectory:
        trajectory = Trajectory(traj_id=traj_id)

        async def store_step(step: Transition):
            await asyncio.gather(*[
                callback.after_transition(traj_id, self.agent, env, step)
                for callback in self.callbacks
            ])
            trajectory.steps.append(step)

        # Set default values to store in the buffer in case reset/init_state fail
        obs: list[Message] = []
        agent_state: Any = None

        try:
            await asyncio.gather(*[
                c.before_rollout(traj_id, env) for c in self.callbacks
            ])

            with reraise_exc_as(EnvError, enabled=self.catch_env_failures):
                obs, tools = await env.reset()
            await asyncio.gather(*[
                c.after_env_reset(traj_id, obs, tools) for c in self.callbacks
            ])

            with reraise_exc_as(AgentError, enabled=self.catch_agent_failures):
                agent_state = await self.agent.init_state(tools)
            await asyncio.gather(*[
                c.after_agent_init_state(traj_id, agent_state) for c in self.callbacks
            ])

            for timestep in itertools.count():
                step = await self._take_step(timestep, traj_id, env, agent_state, obs)

                if timestep + 1 == max_steps and not step.done:
                    # Mark as truncated if we hit max_steps and the state is not terminal.
                    # Do it before store_step(), so that callbacks can access this info
                    step.truncated = True

                # We assume the below won't throw a CaughtError
                await store_step(step)

                # set things up for the next iteration
                agent_state = step.next_agent_state
                obs = step.next_observation

                if step.done or step.truncated:
                    break

        except CaughtError as e:
            # NOTE: This trajectory should not be used for regular training.
            # We save the last transition here for debugging, etc.
            await store_step(
                Transition(
                    timestep=len(trajectory.steps),
                    agent_state=agent_state,
                    next_agent_state=None,
                    observation=obs,
                    next_observation=[],
                    action=None,
                    done=True,
                    metadata={"exception": repr(e.original_exc)},
                )
            )

        self.traj_buffer[traj_id] = trajectory
        return trajectory

    async def _take_step(
        self,
        timestep: int,
        traj_id: str,
        env: Environment,
        agent_state: Any,
        obs: list[Message],
    ) -> Transition:
        async with self.concurrency_limiter:
            await asyncio.gather(*[
                callback.before_transition(traj_id, self.agent, env, agent_state, obs)
                for callback in self.callbacks
            ])

            with reraise_exc_as(AgentError, enabled=self.catch_agent_failures):
                (
                    action,
                    next_agent_state,
                    value,
                ) = await self.agent.get_asv(agent_state, obs)
            await asyncio.gather(*[
                callback.after_agent_get_asv(traj_id, action, next_agent_state, value)
                for callback in self.callbacks
            ])

            with reraise_exc_as(EnvError, enabled=self.catch_env_failures):
                next_obs, reward, done, trunc = await env.step(action.value)
            await asyncio.gather(*[
                callback.after_env_step(traj_id, next_obs, reward, done, trunc)
                for callback in self.callbacks
            ])

            return Transition(
                timestep=timestep,
                agent_state=agent_state,
                next_agent_state=next_agent_state,
                action=action,
                reward=reward,
                value=value,
                observation=obs,
                next_observation=next_obs,
                done=done,
                truncated=trunc,
            )
