"""This module defines the MemoryAgent class, which extends a base agent model with memory.

capabilities. The MemoryAgent can pick and invoke tools based on the stored and retrieved
memories, formatted using specified prompts. A memory is typically a set of previous trajectories
"""

from typing import cast

from aviary.message import Message
from aviary.tools import ToolRequestMessage
from pydantic import ConfigDict, Field

from ldp.graph.common_ops import FxnOp, MemoryOp, PromptOp
from ldp.graph.memory import Memory, MemoryModel
from ldp.graph.op_utils import compute_graph
from ldp.graph.ops import OpResult
from ldp.llms.prompts import indent_xml

from .simple_agent import SimpleAgent, SimpleAgentState


class MemoryAgent(SimpleAgent):
    """
    Simple agent that can pick and invoke tools with memory.

    NOTE: the MemoryAgent does not maintain an explicit value estimate,
    it simply supplies previous trajectories via the prompt.
    As such, the value estimate vhat will always be zero.
    """

    prompt: str = Field(
        default=(
            "<episode-memories>\n<description>\n"
            "These are relevant memories from previous attempts at similar tasks, "
            "along with the action taken and the discounted cumulative reward from that action. "
            "A negative reward is failure, a positive reward is success.\n"
            "</description>{memories}</episode-memories>\n\n"
            "Considering the memories, choose the next action."
        ),
        description="Prompt that includes the memories.",
    )
    memory_prompt: str = Field(
        default="<memory><obs>{input}</obs><action>{output}</action><reward>{value}</reward></memory>",
        description="Prompt for formatting an individual memory. "
        "Use XML instead of JSON to avoid potential escaping issues.",
    )
    num_memories: int = Field(
        default=MemoryModel.DEFAULT_MEMORY_MATCHES,
        description="Number of memories to retrieve from MemoryOp",
    )
    # Freeze to ensure the only mutation happens in either the agent state (which is
    # passed around) or in the internal Ops
    model_config = ConfigDict(frozen=True)

    @staticmethod
    def _parse_memory(prompt: str, memories: list[Memory]) -> str:
        return indent_xml(
            "\n".join([
                prompt.format(**m.model_dump(exclude={"call_id"})) for m in memories
            ])
        )

    @staticmethod
    def _package_messages(
        msgs: list[Message], memory_prompt: str, use_memories: bool
    ) -> list[Message]:
        if use_memories:
            return [*msgs, Message(content=memory_prompt)]
        return msgs

    def __init__(self, memory_model: MemoryModel | None = None, **kwargs):
        super().__init__(**kwargs)
        self._memory_op = MemoryOp(memory_model)
        self._format_memory_op = FxnOp(self._parse_memory)
        self._prompt_op = PromptOp(self.prompt)
        self._package_op = FxnOp(self._package_messages)

    @compute_graph()
    async def get_asv(
        self, agent_state: SimpleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        next_state = agent_state.get_next_state(obs)

        query = "\n\n".join([str(m) for m in next_state.messages if m.role != "system"])
        memories = await self._memory_op(query, matches=self.num_memories)
        packaged_messages = await self._package_op(
            next_state.messages,
            memory_prompt=await self._prompt_op(
                memories=await self._format_memory_op(self.memory_prompt, memories)
            ),
            use_memories=bool(memories.value),
        )
        result = cast(
            OpResult[ToolRequestMessage],
            await self._llm_call_op(
                await self._config_op(), msgs=packaged_messages, tools=next_state.tools
            ),
        )
        next_state.messages = [*next_state.messages, result.value]
        return result, next_state, 0.0
