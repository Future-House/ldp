from __future__ import annotations

import logging
from itertools import product
from typing import Protocol, Self, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ldp.agent import MemoryAgent
from ldp.alg.optimizer.opt import Optimizer
from ldp.data_structures import Trajectory
from ldp.graph.common_ops import MemoryOp
from ldp.graph.memory import Memory
from ldp.graph.op_utils import CallID
from ldp.graph.ops import Op, OpResult, TOutput

logger = logging.getLogger(__name__)


@runtime_checkable
class MemoryFactory(Protocol):
    def __call__(
        self,
        mem_op: MemoryOp,
        mem_call_id: CallID,
        output_op: Op[TOutput],
        output_call_id: CallID,
        value: float,
        **kwargs,
    ) -> Memory: ...


class MemoryOpt(BaseModel, Optimizer):
    """Trainer for memory agents. By default it is a minimizer.

    This optimizer simply adds memories to the MemoryOp using a memory factory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ### Configuration
    memory_op: MemoryOp
    output_op: Op
    reward_discount: float = 1.0
    memory_factory: MemoryFactory = Field(
        default=Memory.from_ops, description="Function to make a Memory.", exclude=True
    )
    memory_template: str = Field(
        default="Input: {input}\nOutput: {output}\nReward: {value}",
        description="Template for a Memory's string representation.",
    )

    ### State
    steps: int = 0
    example_buffer: list[tuple[CallID, CallID, float]] = Field(default_factory=list)

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
        # NOTE: this is a little dangerous. This optimizer currently
        # does not check which memory op calls are upstream of output op calls,
        # besides making sure they belong to the same run.
        # This is not a problem if we have no branching in the compute graph
        # between the memory op and the *final* output op.
        # TODO: fix the above using OpResult.traverse() to find the upstream calls

        if trajectory.failed:
            return

        d_returns = trajectory.compute_discounted_returns(self.reward_discount)

        for step, d_return in zip(trajectory.steps, d_returns, strict=True):
            output = cast(OpResult, step.action)
            mem_call_ids = self.memory_op.get_call_ids({output.call_id.run_id})
            mem_call_ids = {
                m
                for m in mem_call_ids
                if self._memory_filter(m, self.memory_op, d_return)
            }
            output_call_ids = self.output_op.get_call_ids({output.call_id.run_id})
            if len(mem_call_ids) > 1 and len(output_call_ids) > 1:
                raise ValueError(
                    "Multiple memory or output calls in a single run - this violates"
                    " our 1-1 correspondence assumption."
                )

            self.example_buffer.extend(
                (*x, d_return) for x in product(mem_call_ids, output_call_ids)
            )

    async def update(self) -> None:
        """Create new memories from the example buffer and add them to MemoryOp."""
        new_memories = [
            self.memory_factory(
                self.memory_op,
                mem_call_id,
                self.output_op,
                output_call_id,
                d_return,
                template=self.memory_template,
            )
            for mem_call_id, output_call_id, d_return in self.example_buffer
        ]
        for memory in new_memories:
            await self.memory_op.memory_model.add_memory(memory)
        self.steps += 1
        self.example_buffer.clear()


class PositiveMemoryOpt(MemoryOpt):
    def _memory_filter(
        self, call_id: CallID, memory_op: MemoryOp, d_return: float
    ) -> bool:
        # only keep positive memories
        return d_return > 0 and super()._memory_filter(call_id, memory_op, d_return)
