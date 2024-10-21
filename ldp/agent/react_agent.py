import logging
from typing import Any, Self

from aviary.message import Message
from aviary.tools import Tool, ToolRequestMessage
from pydantic import BaseModel, ConfigDict, Field

from ldp.graph import OpResult, compute_graph
from ldp.graph.modules.react import (
    ACT_DEFAULT_PROMPT_TEMPLATE,
    REACT_DEFAULT_PROMPT_TEMPLATE,
    ReActModule,
    ToolDescriptionMethods,
)

from . import DefaultLLMModelNames
from .agent import Agent
from .simple_agent import SimpleAgentState

logger = logging.getLogger(__name__)


class ReActAgent(BaseModel, Agent[SimpleAgentState]):
    """An Act or ReAct Agent built to work with chat models.

    Paper: https://arxiv.org/abs/2210.03629

    The ReAct style is like so, and note Act style has no 'Thought: ' entries:
    System:
        Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
    User:
        {questions}
    Assistant:
        Thought:
        Action:
        Action Input:
    User:
        Observation:
    Assistant:
        Thought:
        Action:
        Action Input:
    ...

    One notable design decision is that ReAct's state does not necessarily
    track ToolRequestMessage. Recall that aviary is in a partially observable
    domain, meaning we don't need to have perfect symmetry with Environments.
    Instead, ReActAgent's state stores a ReAct-style message history, where the
    messages are plain Message (and not a ToolRequestMessage).
    """

    # Freeze to ensure the only mutation happens in either the agent state (which is
    # passed around) or in the internal Ops
    model_config = ConfigDict(frozen=True)

    llm_model: dict[str, Any] = Field(
        default={
            "model": DefaultLLMModelNames.OPENAI.value,
            "temperature": 0.1,
            "logprobs": True,
            "top_logprobs": 1,
        },
        description="Starting configuration for the LLM model.",
    )
    sys_prompt: str = Field(
        default=REACT_DEFAULT_PROMPT_TEMPLATE,
        description="Learnable system prompt template, defaults to ReAct.",
    )
    tool_description_method: ToolDescriptionMethods = Field(
        default=ToolDescriptionMethods.STR,
        description="Method used to describe the tools, defaults to 'str' description.",
    )

    @classmethod
    def make_act_agent(cls, **kwargs) -> Self:
        return cls(sys_prompt=ACT_DEFAULT_PROMPT_TEMPLATE, **kwargs)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._react_module = ReActModule(
            self.llm_model, self.sys_prompt, self.tool_description_method
        )

    async def init_state(self, tools: list[Tool]) -> SimpleAgentState:
        return SimpleAgentState(tools=tools)

    @compute_graph()
    async def get_asv(
        self, agent_state: SimpleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        next_state = agent_state.get_next_state(
            obs=obs,
        )
        final_result, react_message = await self._react_module(
            messages=next_state.messages, tools=next_state.tools
        )
        next_state.messages = [*next_state.messages, react_message, final_result.value]
        return final_result, next_state, 0.0
