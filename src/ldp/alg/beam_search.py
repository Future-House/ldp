import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from copy import deepcopy
from typing import NamedTuple, cast

from aviary.core import Environment

from ldp.agent.agent import Agent, TAgentState
from ldp.data_structures import Trajectory, Transition

from .callbacks import Callback
from .rollout import AgentError, EnvError, TEnv, reraise_exc_as


class Beam(NamedTuple):
    # An ongoing beam contains two things: the trajectory up to now
    # and the environment that the last action was sampled from. We
    # need both to continue sampling the next step.
    traj: Trajectory
    env: Environment


class BeamSearchRollout:
    def __init__(
        self,
        agent: Agent,
        beam_width: int,
        samples_per_beam: int,
        env_clone_fn: Callable[[TEnv], Awaitable[TEnv]],
        agent_clone_fn: Callable[[TAgentState], TAgentState],
        scoring_fn: Callable[[Trajectory], Awaitable[float]],
        replay_actions_on_clone: bool = False,
        callbacks: Sequence[Callback] | None = None,
        catch_agent_failures: bool = True,
        catch_env_failures: bool = True,
        verbose: bool = False,
    ):
        self.agent = agent

        self.catch_agent_failures = catch_agent_failures
        self.catch_env_failures = catch_env_failures

        self.verbose = verbose

        self.traj_buffer: dict[str, Trajectory] = {}
        self.search_buffer: dict[str, list[Trajectory]] = {}

        self.beam_width = beam_width
        self.samples_per_beam = samples_per_beam

        self.env_clone_fn = env_clone_fn
        self.agent_clone_fn = agent_clone_fn
        self.scoring_fn = scoring_fn
        self.replay_actions_on_clone = replay_actions_on_clone

        self.callbacks = callbacks or []

    async def sample_trajectories(
        self,
        environments: Sequence[Environment],
        max_steps: int | None = None,
    ) -> list[Trajectory]:
        self.traj_buffer.clear()
        traj_ids = [uuid.uuid4().hex for _ in environments]

        tasks = [
            self._rollout(traj_id, env, max_steps)
            for traj_id, env in zip(traj_ids, environments, strict=True)
        ]
        await asyncio.gather(*tasks)

        return [self.traj_buffer[traj_id] for traj_id in traj_ids]

    async def _rollout(
        self, traj_id: str, env: Environment, max_steps: int | None
    ) -> None:
        with suppress(AgentError, EnvError):
            # for samples_per_beam==1. we want to ensemble and pick the highest-scoring one
            n_seeds = 1 if self.samples_per_beam > 1 else self.beam_width

            done_beams: list[Beam] = []
            beams = [
                Beam(
                    traj=await Trajectory.from_env(env, traj_id=f"{traj_id}:{i}"),
                    env=env,
                )
                for i in range(n_seeds)
            ]
            # will be replaced if rollout is successful
            self.traj_buffer[traj_id] = beams[0].traj
            self.search_buffer[traj_id] = []

            await asyncio.gather(*[
                c.before_rollout(traj_id, env) for c in self.callbacks
            ])

            with reraise_exc_as(EnvError, self.catch_env_failures):
                init_obs, tools = await env.reset()
            await asyncio.gather(*[
                c.after_env_reset(traj_id, init_obs, tools) for c in self.callbacks
            ])

            with reraise_exc_as(AgentError, self.catch_agent_failures):
                seed_agent_states = await asyncio.gather(
                    *(self.agent.init_state(tools) for _ in range(n_seeds))
                )
            # TODO: implement after_agent_init_state callback

            while len(done_beams) < self.beam_width and beams:
                new_beams = []
                for beam, seed_agent_state in zip(
                    beams, seed_agent_states, strict=True
                ):
                    for i_sample in range(self.samples_per_beam):
                        new_env = await self._clone_env(beam)
                        if new_env is None:
                            continue

                        agent_state = self.agent_clone_fn(seed_agent_state)
                        obs = (
                            beam.traj.steps[-1].next_observation
                            if beam.traj.steps
                            else init_obs.copy()
                        )

                        await asyncio.gather(*[
                            callback.before_transition(
                                traj_id, self.agent, env, agent_state, obs
                            )
                            for callback in self.callbacks
                        ])

                        with reraise_exc_as(AgentError, self.catch_agent_failures):
                            (
                                action,
                                next_agent_state,
                                vhat,
                            ) = await self.agent.get_asv(agent_state, obs)
                        await asyncio.gather(*[
                            callback.after_agent_get_asv(
                                traj_id, action, next_agent_state, vhat
                            )
                            for callback in self.callbacks
                        ])

                        with reraise_exc_as(EnvError, self.catch_env_failures):
                            next_obs, reward, done, trunc = await new_env.step(
                                action.value
                            )
                        await asyncio.gather(*[
                            callback.after_env_step(
                                traj_id, next_obs, reward, done, trunc
                            )
                            for callback in self.callbacks
                        ])

                        step = Transition(
                            timestep=len(beam.traj.steps),
                            agent_state=agent_state,
                            next_agent_state=next_agent_state,
                            observation=obs,
                            next_observation=next_obs,
                            action=action,
                            reward=reward,
                            done=done,
                            truncated=trunc,
                            value=0.0,  # will be filled in
                        )
                        await asyncio.gather(*[
                            callback.after_transition(traj_id, self.agent, env, step)
                            for callback in self.callbacks
                        ])

                        new_beam = Beam(
                            traj=Trajectory(
                                traj_id=cast(str, beam.traj.traj_id) + f":{i_sample}",
                                steps=[*beam.traj.steps, step],
                                metadata=deepcopy(beam.traj.metadata),
                            ),
                            env=new_env,
                        )
                        step.value = await self.scoring_fn(new_beam.traj)
                        self.search_buffer[traj_id].append(new_beam.traj)

                        if (
                            not new_beam.traj.done
                            and max_steps is not None
                            and len(new_beam.traj.steps) >= max_steps
                        ):
                            last_step = new_beam.traj.steps[-1]
                            last_step.done = last_step.truncated = True

                        if new_beam.traj.done:
                            done_beams.append(new_beam)
                        else:
                            new_beams.append(new_beam)

                new_beams.sort(key=lambda b: b.traj.steps[-1].value, reverse=True)
                beams, discarded = (
                    new_beams[: self.beam_width],
                    new_beams[self.beam_width :],
                )
                seed_agent_states = [b.traj.steps[-1].next_agent_state for b in beams]
                await asyncio.gather(*[d.env.close() for d in discarded])

            await asyncio.gather(*[b.env.close() for b in beams])

            self.traj_buffer[traj_id] = max(
                done_beams,
                key=lambda b: (b.traj.steps[-1].truncated, b.traj.steps[-1].value),
            ).traj

    async def _clone_env(self, beam: Beam) -> Environment | None:
        try:
            with reraise_exc_as(EnvError, self.catch_env_failures):
                # I'm not sure how to type hint this
                env = await self.env_clone_fn(beam.env)  # type: ignore[arg-type]
                if self.replay_actions_on_clone:
                    # Some envs can't be cloned, so instead replay.
                    # We rely on env_clone_fn to properly reset the env if needed.
                    # We assume a deterministic env, so the return values are discarded.
                    for step in beam.traj.steps:
                        if step.action is not None:
                            _ = await env.step(step.action.value)
                return env
        except EnvError:
            return None
