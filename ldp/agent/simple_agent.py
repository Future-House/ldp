from __future__ import annotations

from typing import Any, Self, cast

from aviary.message import Message
from aviary.tools import Tool, ToolRequestMessage, ToolResponseMessage
from pydantic import BaseModel, ConfigDict, Field

from ldp.graph import ConfigOp, LLMCallOp, OpResult, compute_graph
from ldp.llms import prepend_sys

from . import DefaultLLMModelNames
from .agent import Agent


class SimpleAgentState(BaseModel):
    """Simple bucket for an Agent to access tools and store messages."""

    tools: list[Tool] = Field(default_factory=list)
    messages: list[ToolRequestMessage | ToolResponseMessage | Message] = Field(
        default_factory=list
    )

    def get_next_state(
        self,
        obs: list[Message] | None = None,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> Self:
        """
        Get the next agent state without mutating the optional prior state.

        Do not mutate self here, just read from it.

        Args:
            obs: Optional observation messages to use in creating the next state.
            tools: Optional list of tools available to the agent. If unspecified, these
                should be pulled from the prior_state.
            kwargs: Additional keyword arguments to pass to this class's constructor.

        Returns:
            The next agent state (which is not an in-place change to self).
        """
        return type(self)(
            tools=tools if tools is not None else self.tools,
            messages=self.messages + (obs or []),
            **kwargs,
        )


class SimpleAgent(BaseModel, Agent[SimpleAgentState]):
    """Simple agent that can pick and invoke tools with a language model.

    It does not have a system prompt because it's meant to be lightweight.
    """

    # Freeze to ensure the only mutation happens in either the agent state (which is
    # passed around) or in the internal Ops
    model_config = ConfigDict(frozen=True)

    llm_model: dict[str, Any] = Field(
        default={"model": DefaultLLMModelNames.OPENAI.value, "temperature": 0.1},
        description="Starting configuration for the LLM model. Trainable.",
    )
    sys_prompt: str | None = Field(
        default=None,
        description=(
            "Opt-in system prompt. If one is passed, the system prompt is not set up to"
            " be trainable, because this class is meant to be quite simple as far as"
            " possible hyperparameters."
        ),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config_op = ConfigOp[dict](config=self.llm_model)
        self._llm_call_op = LLMCallOp()

    async def init_state(self, tools: list[Tool]) -> SimpleAgentState:
        return SimpleAgentState(tools=tools)

    @compute_graph()
    async def get_asv(
        self, agent_state: SimpleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        next_state = agent_state.get_next_state(obs)

        messages = (
            prepend_sys(next_state.messages, sys_content=self.sys_prompt)
            if self.sys_prompt is not None
            else next_state.messages
        )
        result = cast(
            OpResult[ToolRequestMessage],
            await self._llm_call_op(
                await self._config_op(), msgs=messages, tools=next_state.tools
            ),
        )
        next_state.messages = [*next_state.messages, result.value]
        return result, next_state, 0.0
