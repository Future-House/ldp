from __future__ import annotations

from collections.abc import Awaitable, Callable
from itertools import chain
from typing import Any, Self, cast

from aviary.core import Message, Tool, ToolRequestMessage, ToolResponseMessage
from aviary.message import EnvStateMessage
from lmi import CommonLLMNames
from pydantic import BaseModel, ConfigDict, Field

from ldp.graph import ConfigOp, FxnOp, LLMCallOp, OpResult, compute_graph
from ldp.llms import prepend_sys
from ldp.utils import split_message_transitions

from . import DEFAULT_LLM_COMPLETION_TIMEOUT
from .agent import Agent


class HiddenEnvStateMessage(EnvStateMessage):
    content: str = "[Previous environment state - hidden]"


def hide_action_content(msg: ToolRequestMessage) -> ToolRequestMessage:
    return msg.model_copy(update={"content": None})


class SimpleAgentState(BaseModel):
    """Simple bucket for an Agent to access tools and store messages."""

    model_config = ConfigDict(extra="forbid")

    tools: list[Tool] = Field(default_factory=list)
    messages: list[ToolRequestMessage | ToolResponseMessage | Message] = Field(
        default_factory=list
    )
    hide_old_env_states: bool = Field(
        default=False,
        description="Whether to hide old EnvStateMessages.",
    )
    hide_old_action_content: bool = Field(
        default=False,
        description="If True, will hide the content of old ToolRequestMessages.",
    )
    sliding_window: int | None = Field(
        default=None,
        description="Number of previous trajectory transitions to keep. None means all previous transitions.",
    )

    def get_next_state(
        self,
        obs: list[Message] | None = None,
        tools: list[Tool] | None = None,
        hide_old_env_states: bool | None = None,
        **kwargs,
    ) -> Self:
        """
        Get the next agent state without mutating the optional prior state.

        Do not mutate self here, just read from it.

        Args:
            obs: Optional observation messages to use in creating the next state.
            tools: Optional list of tools available to the agent. If unspecified, these
                should be pulled from the prior_state.
            hide_old_env_states: Optional override of self.hide_old_env_states.
                TODO: do we still need this override?
            kwargs: Additional keyword arguments to pass to this class's constructor.

        Returns:
            The next agent state (which is not an in-place change to self).
        """
        old_messages = self.messages
        hide_old_env_states = (
            hide_old_env_states
            if hide_old_env_states is not None
            else self.hide_old_env_states
        )
        if self.sliding_window is not None and self.sliding_window > 0 and old_messages:
            hide_message = "[Previous messages - hidden]"
            msg_blocks = split_message_transitions(old_messages)

            # Do not duplicate hide_message if it's already in the first block
            if msg_blocks[0][-1].content != hide_message:
                msg_blocks[0].append(Message(content=hide_message))

            old_messages = (
                msg_blocks[0]  # keep system messages + user message + hide_message
                + list(chain.from_iterable(msg_blocks[1:][-self.sliding_window :]))
            )

        if hide_old_env_states:
            old_messages = [
                HiddenEnvStateMessage() if isinstance(m, EnvStateMessage) else m
                for m in old_messages
            ]
        if self.hide_old_action_content:
            old_messages = [
                hide_action_content(m) if isinstance(m, ToolRequestMessage) else m
                for m in old_messages
            ]

        return type(self)(
            tools=tools if tools is not None else self.tools,
            messages=old_messages + (obs or []),
            hide_old_env_states=hide_old_env_states,
            hide_old_action_content=self.hide_old_action_content,
            sliding_window=self.sliding_window,
            **kwargs,
        )


class SimpleAgent(BaseModel, Agent[SimpleAgentState]):
    """Simple agent that can pick and invoke tools with a language model."""

    # Freeze to ensure the only mutation happens in either the agent state (which is
    # passed around) or in the internal Ops
    model_config = ConfigDict(frozen=True)

    llm_model: dict[str, Any] = Field(
        default={
            "name": CommonLLMNames.GPT_4O.value,
            "temperature": 0.1,
            "timeout": DEFAULT_LLM_COMPLETION_TIMEOUT,
        },
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

    hide_old_env_states: bool = Field(
        default=False,
        description="See SimpleAgentState.hide_old_env_states.",
    )

    hide_old_action_content: bool = Field(
        default=False,
        description="See SimpleAgentState.hide_old_action_content.",
    )

    sliding_window: int | None = Field(
        default=None,
        description="See SimpleAgentState.sliding_window.",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config_op = ConfigOp[dict](config=self.llm_model)
        self._llm_call_op = LLMCallOp()

    async def init_state(self, tools: list[Tool]) -> SimpleAgentState:
        return SimpleAgentState(
            tools=tools,
            hide_old_env_states=self.hide_old_env_states,
            hide_old_action_content=self.hide_old_action_content,
            sliding_window=self.sliding_window,
        )

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
            "OpResult[ToolRequestMessage]",
            await self._llm_call_op(
                await self._config_op(), msgs=messages, tools=next_state.tools
            ),
        )
        next_state.messages = [*next_state.messages, result.value]
        return result, next_state, 0.0


class NoToolsSimpleAgent(SimpleAgent):
    """Agent whose action is parsed out of a LLM prompt without tool schemae."""

    def __init__(
        self,
        cast_tool_request: (
            Callable[[Message], ToolRequestMessage]
            | Callable[[Message], Awaitable[ToolRequestMessage]]
        ),
        **kwargs,
    ):
        """Initialize.

        Args:
            cast_tool_request: Function that is given a plain text message and produces
                a tool request message.
            **kwargs: Keyword arguments to pass to SimpleAgent's constructor.
        """
        super().__init__(**kwargs)
        self._cast_tool_request_op = FxnOp[ToolRequestMessage](cast_tool_request)

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
        result = await self._cast_tool_request_op(
            # NOTE: this call has no tools specified to the LLM
            await self._llm_call_op(await self._config_op(), msgs=messages)
        )
        next_state.messages = [*next_state.messages, result.value]
        return result, next_state, 0.0
