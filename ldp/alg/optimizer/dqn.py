from __future__ import annotations

import asyncio
import itertools
import random
from collections.abc import Sequence
from copy import deepcopy
from enum import StrEnum
from math import ceil
from typing import Any, Self, cast
from uuid import UUID

import torch
import torch.optim
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.nn import functional as F  # noqa: N812

from ldp.agent import DQNAgent
from ldp.alg.algorithms import discounted_returns
from ldp.alg.optimizer.opt import Optimizer
from ldp.alg.optimizer.replay_buffers import CircularReplayBuffer
from ldp.data_structures import Trajectory, Transition
from ldp.graph.modules import DQNOp, DQNPolicyModule
from ldp.graph.op_utils import eval_mode
from ldp.graph.ops import OpResult

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


class DQNTarget(StrEnum):
    Q = "Q"  # Standard Bellman target
    SARSA = "SARSA"  # SARSA target
    MC_SARSA = "MC_SARSA"  # Monte Carlo SARSA target (discounted cumulative return)


class DQNOptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 0.001
    num_update_epochs: int = Field(
        1,
        description="Number of passes through the train buffer "
        "for every update() call.",
    )
    batch_size: int = 32
    gradient_checkpointing: bool = Field(
        default=False,
        description="Only supported for transformers models.",
    )
    target: DQNTarget = DQNTarget.Q

    eval_every: int | float | None = Field(
        None,
        description="Check validation loss every eval_every steps. "
        "If None, don't run validation. If a float in (0, 1], "
        "run validation every eval_every fraction of the train buffer.",
    )
    early_stopping_tolerance: float = Field(
        0.1,
        description="If the validation loss increases by more than this much, "
        "early stop. Default is 10%. ",
    )

    ignore_truncated: bool = Field(
        default=False,
        description="If True, do not train on any transitions from a "
        "truncated trajectory. This can make sense if rewards are not emitted "
        "until the trajectory finishes.",
    )
    train_buffer_size: int | None = Field(
        1_000,
        description="Size of the replay buffer used to train the Q network. "
        "If None, the buffer can grow arbitrarily large. ",
    )
    val_buffer_size: int | None = Field(
        100,
        description="Size of the replay buffer used for evaluating the Q network."
        "If None, the buffer can grow arbitrarily large. ",
    )
    val_frac: float | None = Field(
        None,
        description="Fraction of aggregated trajectories that will be added to the "
        "val buffer. If not set, will attempt to infer from [val,train]_buffer_sizes.",
    )

    continual_training: bool = Field(
        default=True,
        description="If True (default), will continually train the DQN "
        "across update() calls. If False, will reset the DQN on each "
        "update() call and retrain from scratch using the train buffer.",
    )

    reward_discount: float = 1.0
    soft_update_tau: float = Field(
        1.0,
        description="Update coefficient for the target network if performing Q learning. "
        "1.0 means we use the policy network at the end of the previous update() as the "
        "target network for the next update().",
    )

    optimizer: str = "SGD"
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_val_sizes(self):
        if self.eval_every is None:
            # don't need to store anything in the val buffer if we're
            # not running eval
            self.val_buffer_size = 0
            self.val_frac = 0.0

        if self.val_frac is None:
            if self.val_buffer_size is None or self.train_buffer_size is None:
                raise ValueError(
                    "If val_frac is not set, val_buffer_size and train_buffer_size "
                    "must be set."
                )
            self.val_frac = self.val_buffer_size / (
                self.val_buffer_size + self.train_buffer_size
            )

        return self


class DQNOptimizer(Optimizer):
    def __init__(
        self,
        policy: DQNPolicyModule,
        config: DQNOptimizerConfig | None = None,
        log_to_wandb: bool = False,
    ):
        if config is None:
            config = DQNOptimizerConfig()
        self.config = config

        if log_to_wandb and wandb is None:
            raise ImportError(
                "If logging to wandb, please install its package via `pip install wandb`."
            )
        self.log_to_wandb = log_to_wandb

        self.policy = policy

        # A buffer element consists of:
        # - s_t, a_t
        # - r_{t+1}
        # - all the (s_{t+1}, a_{t+1}) candidates considered for the next transition
        # - other metadata like the value estimate when sampling
        self.train_buffer = CircularReplayBuffer()
        self.val_buffer = CircularReplayBuffer()

        if not self.config.continual_training:
            # Store a copy of the state dict so we can reset the model
            self.init_state_dict = deepcopy(self.policy.dqn.network.state_dict())

        optimizer_cls = getattr(torch.optim, self.config.optimizer)
        self.optimizer = optimizer_cls(
            self.dqn_op.network.parameters(),
            lr=self.config.lr,
            **self.config.optimizer_kwargs,
        )
        # Define the model's device/dtype from a random parameter - we assume they're all the same
        param = next(self.dqn_op.network.parameters())
        self.device = param.device
        self.dtype = param.dtype

        if self.config.gradient_checkpointing:
            try:
                self.dqn_op.network.gradient_checkpointing_enable()
            except AttributeError as e:
                raise ValueError(
                    "Gradient checkpointing was requested, but we do not "
                    f"support it for {self.dqn_op.network.__class__.__name__}"
                ) from e

    @classmethod
    def from_agent(cls, agent: DQNAgent, **kwargs) -> Self:
        return cls(policy=agent._dqn_policy, **kwargs)

    @property
    def dqn_op(self) -> DQNOp:
        return self.policy.dqn

    def _update_target_network(self) -> None:
        tau = self.config.soft_update_tau
        target_state_dict = self.dqn_op.target_network.state_dict()
        for name, param in self.dqn_op.network.named_parameters():
            target_param = target_state_dict[name]
            # write it it out this way so we're doing in-place operations
            # The actual math is target_param = tau * param + (1 - tau) * target_param
            target_param.data *= 1 - tau
            target_param.data += tau * param.data

    def aggregate_trajectory(self, trajectory: Trajectory) -> None:  # noqa: C901
        """Add training examples to the replay buffers."""
        if trajectory.failed or not trajectory.steps:
            return

        if self.config.ignore_truncated and trajectory.steps[-1].truncated:
            return

        # use the same buffer for all transitions in this trajectory to avoid
        # train/val leakage. Remember, we want to generalize to unseen trajectories.
        buffer = (
            self.val_buffer
            if random.uniform(0, 1) < cast(float, self.config.val_frac)
            else self.train_buffer
        )

        steps = cast(list[Transition | None], [*trajectory.steps, None])
        d_returns = discounted_returns(
            [step.reward for step in trajectory.steps],
            [step.done for step in trajectory.steps],
            self.config.reward_discount,
        )

        for (step, next_step), d_return in zip(
            itertools.pairwise(steps), d_returns, strict=True
        ):
            step = cast(Transition, step)

            if step.truncated:
                break

            if next_step is None:
                assert step.done

            # checking trajectory.failed ensures this is an OpResult
            run_id = cast(OpResult, step.action).call_id.run_id

            if self.config.target == DQNTarget.MC_SARSA:
                # Don't need next_state_action_cands - avoid ctx lookup
                next_state_action_cands: list[str] | None = None
            elif step.done:
                # Last step, so the target is just r_t+1
                next_state_action_cands = []
            else:
                next_state_action_cands = self._get_state_action_cands(
                    cast(OpResult, cast(Transition, next_step).action).call_id.run_id
                )

            dqn_call_ids = self.dqn_op.get_call_ids({run_id})
            for dqn_call_id in dqn_call_ids:
                if (
                    self.dqn_op.ctx.get(dqn_call_id, "grad_output", default=None)
                    is None
                ):
                    # This op call was pruned from backward compute graph - skip.
                    continue

                tensor_args, tensor_kwargs = self.dqn_op.ctx.get(
                    dqn_call_id, DQNOp.CTX_TENSOR_INPUT_KEY
                )

                buffer.append({
                    # (s_t, a_t)
                    "input_args": tensor_args,
                    "input_kwargs": tensor_kwargs,
                    # r_{t+1}
                    "reward": step.reward,
                    # all the (s_{t+1}, a_{t+1}) candidates considered for the next transition
                    "next_state_action_cands": next_state_action_cands,
                    # other metadata like the value estimate when sampling
                    "q": step.value,
                    "discounted_return": d_return,
                })

        if self.config.train_buffer_size:
            self.train_buffer.resize(self.config.train_buffer_size)
        if self.config.val_buffer_size:
            self.val_buffer.resize(self.config.val_buffer_size)

    async def update(self):  # noqa: C901
        num_samples = len(self.train_buffer)
        if num_samples < self.config.batch_size:
            return

        if self.log_to_wandb:
            wandb.log({"dqn/train_buffer_size": num_samples})

        num_steps_per_epoch = ceil(num_samples / self.config.batch_size)
        num_train_steps = self.config.num_update_epochs * num_steps_per_epoch

        val_every = self.config.eval_every or 0
        if 0 < val_every <= 1:
            val_every = int(val_every * num_steps_per_epoch)

        best_val_loss = float("inf")
        best_ckpt: dict | None = None

        if not self.config.continual_training:
            self.dqn_op.network.load_state_dict(self.init_state_dict)

        for batch_num, batch in enumerate(
            self.train_buffer.batched_iter(
                self.config.batch_size,
                infinite=True,  # will count the number of batches manually
            ),
            start=1,
        ):
            tensor_args, tensor_kwargs = self._collate_fn(
                batch["input_args"], batch["input_kwargs"]
            )
            targets = await self._compute_targets(batch)

            tensor_args, tensor_kwargs, targets = self._move_tensors(
                tensor_args, tensor_kwargs, targets
            )

            self.optimizer.zero_grad()
            self.dqn_op.network.train()
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                loss = self._compute_loss(tensor_args, tensor_kwargs, targets)
            loss.backward()
            self.optimizer.step()

            if self.log_to_wandb:
                qs = torch.tensor(batch["q"])
                returns = torch.tensor(batch["discounted_return"])
                wandb.log({
                    "dqn/minibatch_loss": loss.item(),
                    "dqn/minibatch_empirical_error": F.mse_loss(qs, returns).item(),
                })

            if val_every and self.val_buffer and batch_num % val_every == 0:
                val_loss = await self.run_val_loop(_internal_call=True)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = self.dqn_op.network.state_dict()
                elif (
                    val_loss
                    > (1 + self.config.early_stopping_tolerance) * best_val_loss
                ):
                    # early stopping. best_ckpt will not be None if best_val_loss is finite.
                    self.dqn_op.network.load_state_dict(cast(dict, best_ckpt))
                    if self.log_to_wandb:
                        n_epochs = (
                            batch_num * self.config.batch_size / len(self.train_buffer)
                        )
                        wandb.log({"dqn/early_stopping_epoch": n_epochs})
                    break

            if batch_num == num_train_steps:
                break

        if best_ckpt is not None:
            # Reset to best ckpt, in case it was from an earlier epoch and ES didn't kick in
            self.dqn_op.network.load_state_dict(best_ckpt)

        # TODO: should we do this here? Or per-epoch? Or per-batch? Unclear what to do in this
        # semi-offline setting. Maybe experiment.
        self._update_target_network()

    @torch.no_grad()
    async def run_val_loop(self, _internal_call: bool = False) -> float:
        # _internal_call is set to True if this is being called by the optimizer, in which case
        # we want to log things.
        losses = []

        if _internal_call and self.log_to_wandb:
            wandb.log({"dqn/val_buffer_size": len(self.val_buffer)})

        for batch in self.val_buffer.batched_iter(self.config.batch_size):
            tensor_args, tensor_kwargs = self._collate_fn(
                batch["input_args"], batch["input_kwargs"]
            )
            targets = await self._compute_targets(batch)

            tensor_args, tensor_kwargs, targets = self._move_tensors(
                tensor_args, tensor_kwargs, targets
            )

            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                loss = self._compute_loss(tensor_args, tensor_kwargs, targets)

            losses.append(loss.item())

        losses_tensor = torch.tensor(losses)
        val_loss, val_loss_std = losses_tensor.mean().item(), losses_tensor.std().item()
        if _internal_call and self.log_to_wandb:
            wandb.log({
                "dqn/val_loss": val_loss,
                "dqn/val_loss_std": val_loss_std,
            })

        return val_loss

    def _get_state_action_cands(self, run_id: UUID) -> list[str]:
        actor_op = self.policy.action_selector
        call_id, *extra = actor_op.get_call_ids({run_id})
        if extra:
            raise RuntimeError(
                "DQNOptimizer cannot handle a actor op that was called more than once "
                "in a compute graph."
            )

        state_actions = cast(list[str], actor_op.ctx.get(call_id, "state_actions"))

        if self.config.target == DQNTarget.Q:
            # Off-policy: need to see the whole action space (or a set of unbiased samples
            # from it).
            return state_actions

        # On-policy: only need the selected action
        i_selected = cast(int, actor_op.ctx.get(call_id, "i_selected"))
        return [state_actions[i_selected]]

    async def _compute_targets(self, batch: dict[str, list[Any]]) -> torch.Tensor:
        with self.dqn_op.use_target_network():
            targets = await asyncio.gather(
                *list(
                    itertools.starmap(
                        self._compute_target,
                        zip(
                            batch["reward"],
                            batch["discounted_return"],
                            batch["next_state_action_cands"],
                            strict=True,
                        ),
                    )
                )
            )

        return torch.tensor(targets, dtype=torch.float32)[:, None]

    @eval_mode()
    async def _compute_target(
        self,
        r_tp1: float,
        discounted_return: float,
        next_state_action_cands: Sequence[str],
    ) -> float:
        if self.config.target == DQNTarget.MC_SARSA:
            return discounted_return

        if not next_state_action_cands:
            # This is the last step, so there is no next state
            return r_tp1

        # We use the Bellman equation to compute the target
        # Q(s_t, a_t) = r_{t+1} + gamma * max_a Q(s_{t+1}, a)
        # where a is from our set of candidates
        q_tp1s = await asyncio.gather(*[
            self.dqn_op(s_a) for s_a in next_state_action_cands
        ])

        # TODO: add DDQN
        q_tp1 = max(q.value for q in q_tp1s)

        return r_tp1 + self.config.reward_discount * q_tp1

    def _move_tensors(
        self, tensor_args, tensor_kwargs, target_tensor
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        tensor_args = [arg.to(device=self.device) for arg in tensor_args]
        tensor_kwargs = {
            k: kwarg.to(device=self.device) for k, kwarg in tensor_kwargs.items()
        }
        target_tensor = target_tensor.to(device=self.device)

        return tensor_args, tensor_kwargs, target_tensor

    def _collate_fn(
        self,
        input_args: list[list[torch.Tensor]],
        input_kwargs: list[dict[str, torch.Tensor]],
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        tensor_args = [
            torch.stack([inp[i] for inp in input_args], dim=0)
            # use the first element to find the number of args
            for i in range(len(input_args[0]))
        ]

        tensor_kwargs = {
            key: torch.stack([inp[key] for inp in input_kwargs], dim=0)
            # use the first element to find the keys
            for key in input_kwargs[0]
        }

        return tensor_args, tensor_kwargs

    def _compute_loss(
        self,
        tensor_args: list[torch.Tensor],
        tensor_kwargs: dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.dqn_op.network(*tensor_args, **tensor_kwargs)
        return F.mse_loss(pred, targets)
