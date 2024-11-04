from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from contextlib import nullcontext
from typing import ClassVar, Protocol, Self, cast, runtime_checkable

from aviary.core import Message, Messages, ToolRequestMessage
from aviary_internal.utils.config import ConfigModel
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ldp.agent import MemoryAgent
from ldp.alg.optimizer.opt import Optimizer
from ldp.data_structures import Trajectory
from ldp.graph import CallID, Memory, MemoryOp, Op, OpResult
from ldp.graph.common_ops import LLMCallOp
from ldp.graph.ops import TOutput

logger = logging.getLogger(__name__)


@runtime_checkable
class MemoryFactory(Protocol):
    async def __call__(
        self,
        memory_op: MemoryOp,
        output_op: Op[TOutput],
        memory_template: str,
        example_buffer: Iterable[tuple[CallID, CallID, float, JsonValue]],
    ) -> list[Memory]:
        """
        Create new memories from the example buffer.

        Args:
            memory_op: MemoryOp whose context contains the MemoryModel's query and input.
            output_op: Op whose context contains an output that can be correlated with
                how good the outcome was.
            memory_template: Template used for the Memory's string representation.
            example_buffer: Buffer of 4-tuples containing the memory_op's call ID, the
                output_op's call ID, the current discounted return, and arbitrary JSON
                metadata (which can be used to add task-specific metadata the memory).

        Returns:
            New Memories created.
        """


async def _default_memory_factory(
    memory_op: MemoryOp,
    output_op: Op[TOutput],
    memory_template: str,
    example_buffer: Iterable[tuple[CallID, CallID, float, JsonValue]],
) -> list[Memory]:
    return [
        Memory.from_ops(
            memory_op,
            mem_call_id,
            output_op,
            output_call_id,
            d_return,
            template=memory_template,
            metadata=metadata,
        )
        for mem_call_id, output_call_id, d_return, metadata in example_buffer
    ]


class MemoryOpt(BaseModel, Optimizer):
    """Trainer for memory agents. By default it is a minimizer.

    This optimizer simply adds memories to the MemoryOp using a memory factory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Working around https://github.com/pydantic/pydantic/issues/10551
    default_memory_factory: ClassVar[MemoryFactory] = _default_memory_factory

    ### Configuration
    memory_op: MemoryOp
    output_op: Op
    reward_discount: float = 1.0
    memory_factory: MemoryFactory = Field(
        default=default_memory_factory,
        description=(
            "Async function to make Memories from an example buffer. It's async so this"
            " can involve an LLM completion if desired."
        ),
        exclude=True,
    )
    memory_template: str = Field(
        default="Input: {input}\nOutput: {output}\nReward: {value}",
        description="Template for a Memory's string representation.",
    )

    ### State
    steps: int = 0
    example_buffer: list[tuple[CallID, CallID, float, JsonValue]] = Field(
        default_factory=list
    )

    @classmethod
    def from_agent(cls, agent: MemoryAgent, **kwargs) -> Self:
        return cls(memory_op=agent._memory_op, output_op=agent._llm_call_op, **kwargs)

    def _memory_filter(
        self, call_id: CallID, memory_op: MemoryOp, d_return: float
    ) -> bool:
        # only keep memories that backprop reached, i.e. those that were used in
        # achieving the reward
        return memory_op.ctx.get(call_id, "grad_output", default=None) is not None

    def aggregate_trajectory(self, trajectory: Trajectory) -> None:
        if trajectory.failed:
            return

        d_returns = trajectory.compute_discounted_returns(self.reward_discount)

        for step, d_return in zip(trajectory.steps, d_returns, strict=True):
            output_run_id = cast(OpResult, step.action).call_id.run_id
            output_call_ids = self.output_op.get_call_ids({output_run_id})

            for output_call_id in output_call_ids:
                output = cast(
                    OpResult, self.output_op.ctx.get(output_call_id, "output")
                )
                mem_call_ids = {
                    m.call_id
                    for m in output.get_upstream_results(self.memory_op)
                    if self._memory_filter(m.call_id, self.memory_op, d_return)
                }

                self.example_buffer.extend(
                    (
                        mem_call_id,
                        output_call_id,
                        d_return,
                        {
                            "timestep": step.timestep,
                            "done": step.done,
                            "truncated": step.truncated,
                        },
                    )
                    for mem_call_id in mem_call_ids
                )

    async def update(self) -> None:
        """Create new memories from the example buffer and add them to MemoryOp."""
        for memory in await self.memory_factory(  # pylint: disable=too-many-function-args
            self.memory_op, self.output_op, self.memory_template, self.example_buffer
        ):
            await self.memory_op.memory_model.add_memory(memory)
        self.steps += 1
        self.example_buffer.clear()


class PositiveMemoryOpt(MemoryOpt):
    def _memory_filter(
        self, call_id: CallID, memory_op: MemoryOp, d_return: float
    ) -> bool:
        # only keep positive memories
        return d_return > 0 and super()._memory_filter(call_id, memory_op, d_return)


class MemoryOptConfig(ConfigModel):
    reward_discount: float = 1.0
    return_threshold: float = Field(
        default=0.0,
        description="Only keep memories with a return above this threshold.",
    )
    concurrency_limit: int | None = Field(
        default=None,
        description="If set, imposes a concurrency limit on memory construction. "
        "Can be useful if using an LLM to construct memories and the buffer is very large.",
    )


class MemoryOptBufferElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Messages
    action: ToolRequestMessage
    d_return: float


class MemoryOpt2(BaseModel, Optimizer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: MemoryOptConfig = Field(default_factory=MemoryOptConfig)
    memory_op: MemoryOp = Field(
        description="A memory Op that lives upstream of action_op in the compute graph."
    )
    action_op: LLMCallOp = Field(
        description="The Op that produces the action of the compute graph. "
        "This is typically going to be a terminal Op (i.e. no children)."
    )

    ### Optimizer state
    example_buffer: list[MemoryOptBufferElement] = Field(
        default_factory=list, description="Buffer of memories to add to the MemoryOp."
    )

    def aggregate_trajectory(self, trajectory: Trajectory) -> None:
        if trajectory.failed:
            return

        d_returns = trajectory.compute_discounted_returns(self.config.reward_discount)

        for step, d_return in zip(trajectory.steps, d_returns, strict=True):
            if d_return < self.config.return_threshold:
                continue

            # Note that we don't really need to traverse the compute graph here, since we
            # only need the inputs to action_op,
            action = cast(OpResult[ToolRequestMessage], step.action)
            _, input_kwargs = action.inputs
            state = cast(Messages, input_kwargs["msgs"])

            self.example_buffer.append(
                MemoryOptBufferElement(
                    state=state, action=action.value, d_return=d_return
                )
            )

    async def update(self):
        await asyncio.gather(*[self._add_memory(el) for el in self.example_buffer])
        self.example_buffer.clear()

    async def _add_memory(self, el: MemoryOptBufferElement):
        async with self._concurrency_limiter:
            await self.memory_op.add_memory(
                state=el.state, action=el.action, value=el.d_return
            )

    @model_validator(mode="after")
    def set_concurrency_limit(self):
        self._concurrency_limiter = (
            asyncio.Semaphore(self.config.concurrency_limit)
            if self.config.concurrency_limit
            else nullcontext()
        )
