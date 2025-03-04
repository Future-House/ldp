import asyncio
import itertools
import logging
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from typing import Any, TypeVar

from aviary.core import Environment, Message
from tqdm import tqdm

from ldp.agent import Agent
from ldp.data_structures import Trajectory, Transition

from .callbacks import Callback

logger = logging.getLogger(__name__)


TEnv = TypeVar("TEnv", bound=Environment)


class CaughtError(Exception):
    """Base class for reraised exceptions when catching is enabled."""

    def __init__(self, original_exc: Exception):
        super().__init__(str(original_exc))
        self.original_exc = original_exc

    exc_type = "undefined"


class AgentError(CaughtError):
    exc_type = "agent"


class EnvError(CaughtError):
    exc_type = "env"


@contextmanager
def reraise_exc_as(reraise: type[CaughtError], enabled: bool) -> Iterator[None]:
    """Context manager that reraises exceptions as a custom CaughtError type if enabled."""
    try:
        yield
    except Exception as e:
        if enabled:
            # Minimal logging instead of spamming. Detailed error stored in the trajectory's metadata.
            logger.debug(f"Reraising {reraise.exc_type} exception.")
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

    async def sample_trajectories(
        self,
        environment_factory: Callable[[], TEnv] | None = None,
        environments: Sequence[TEnv] | None = None,
        batch_size: int = 1,
        max_steps: int | None = None,
        *,
        log_exceptions_immediately: bool = False,
    ) -> list[tuple[Trajectory, Environment]] | list[Trajectory]:
        """Sample trajectories from environments, either via factory or pre-created.

        There are two main ways to use this method:

        1. Using an environment factory:
            Run rollouts in parallel, using a factory to construct environments.
            We will construct `batch_size` environments and run rollouts on each of them.
            If `max_steps` is set, rollouts will be truncated at this value. If a rollout
            has fewer than `max_steps`, then a new environment will be constructed and another
            rollout will be started until `max_steps` is reached.

            In this case, returns a list of (trajectory, environment) tuples.

        2. Using a sequence of environments:
            Run rollouts in parallel on a list of provided environments.
            In this case, returns a list of trajectories.

        Args:
            environment_factory: A no-argument callable that returns an environment instance
            environments: A list of environments to run rollouts on
            batch_size: Number of parallel environments to run when using environment_factory. Defaults to 1.
            max_steps: Max steps per rollout. Defaults to None, in which case the rollouts are run
                until environment returns done.
            log_exceptions_immediately: Whether to log exceptions as they occur
                or only at the end of the rollouts. Defaults to False.

        Returns:
            Either list[tuple[Trajectory, Environment]] or list[Trajectory] depending on whether
            environment_factory or environments is provided.

        Raises:
            TypeError: If neither environment_factory nor environments is provided.
        """
        if environment_factory is not None:
            assert environments is None, (
                "Cannot use environment_factory with environments"
            )
            return await self._sample_trajectories_from_env_factory(
                environment_factory,
                batch_size,
                max_steps,
                log_exceptions_immediately=log_exceptions_immediately,
            )

        if environments is not None:
            assert environment_factory is None, (
                "Cannot use environments with environment_factory"
            )
            return await self._sample_trajectories_from_envs(
                environments,
                max_steps,
                log_exceptions_immediately=log_exceptions_immediately,
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
        *,
        log_exceptions_immediately: bool = False,
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
                    log_exceptions_immediately=log_exceptions_immediately,
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
                            log_exceptions_immediately=log_exceptions_immediately,
                        )
                    )
                    new_tasks.append(new_task)

            tasks = list(pending) + new_tasks

        return results

    async def _sample_trajectories_from_envs(
        self,
        environments: Sequence[Environment],
        max_steps: int | None = None,
        *,
        log_exceptions_immediately: bool = False,
    ) -> list[Trajectory]:
        self.traj_buffer.clear()
        exception_counter: Counter = Counter()

        traj_ids = [uuid.uuid4().hex for _ in environments]

        # Create all tasks first
        tasks = [
            asyncio.create_task(
                self._rollout(
                    traj_id,
                    env,
                    max_steps=max_steps,
                    log_exceptions_immediately=log_exceptions_immediately,
                )
            )
            for traj_id, env in zip(traj_ids, environments, strict=True)
        ]

        with tqdm(
            total=len(tasks),
            desc="Rollouts",
            unit="rollout",
            ncols=0,
        ) as pbar:
            for task in asyncio.as_completed(tasks):
                trajectory = await task
                pbar.update(1)
                # Check if this trajectory ended with an exception
                if trajectory.steps:
                    last_step = trajectory.steps[-1]
                    if last_step.metadata.get("exception"):
                        # We'll keep it short but still have something to categorize
                        exc_str: str = str(last_step.metadata["exception"])[
                            :500
                        ].replace('"', "'")
                        exception_counter[exc_str] += 1
                        num_exceptions = sum(exception_counter.values())
                        pbar.set_postfix({"num_exceptions": num_exceptions})

        # Final summary of exceptions (if any)
        if exception_counter and not log_exceptions_immediately:
            summary = ["Caught exceptions:", "Count  Exception"]
            summary.extend(
                f"{count:<6d} {exc:<50s}" for exc, count in exception_counter.items()
            )
            logger.info("\n".join(summary))

        return [self.traj_buffer[traj_id] for traj_id in traj_ids]

    async def _rollout(
        self,
        traj_id: str,
        env: Environment,
        max_steps: int | None,
        *,
        log_exceptions_immediately: bool = False,
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
            if log_exceptions_immediately:
                logger.exception(f"Exception in rollout {traj_id}: {e.original_exc}")

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
