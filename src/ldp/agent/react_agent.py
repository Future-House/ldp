import logging
from typing import Any, Self, cast

from aviary.core import (
    MalformedMessageError,
    Message,
    Tool,
    ToolRequestMessage,
    ToolResponseMessage,
)
from lmi import CommonLLMNames
from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    Future,
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from ldp.graph import OpResult, compute_graph
from ldp.graph.modules.react import (
    ACT_DEFAULT_PROMPT_TEMPLATE,
    ACT_DEFAULT_SINGLE_PROMPT_TEMPLATE,
    REACT_DEFAULT_PROMPT_TEMPLATE,
    REACT_DEFAULT_SINGLE_PROMPT_TEMPLATE,
    ReActModule,
    ReActModuleSinglePrompt,
    ToolDescriptionMethods,
)

from . import DEFAULT_LLM_COMPLETION_TIMEOUT
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
            "name": CommonLLMNames.GPT_4O.value,
            "temperature": 0.1,
            "logprobs": True,
            "top_logprobs": 1,
            "timeout": DEFAULT_LLM_COMPLETION_TIMEOUT,
        },
        description="Starting configuration for the LLM model.",
    )
    sys_prompt: str = Field(
        description=(
            "Learnable system prompt template. If not provided, a default ReAct prompt "
            "template will be assigned, depending on the single_prompt setting."
        ),
    )
    tool_description_method: ToolDescriptionMethods = Field(
        default=ToolDescriptionMethods.STR,
        description="Method used to describe the tools, defaults to 'str' description.",
    )
    single_prompt: bool = Field(
        default=False,
        description=(
            "Specifies whether to use a single prompt for both reasoning and action"
            " selection, or to use 2 sequential prompts. When set to True, a single API"
            " call is made, and ldp handles the message parsing to extract the action."
            " If set to False, a second API call is made specifically to request the"
            " action, with parsing done on the API side. Defaults to False, as it"
            " results in fewer action selection failures."
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

    @classmethod
    def make_act_agent(cls, **kwargs) -> Self:
        single_prompt = kwargs.pop("single_prompt", False)
        return cls(
            sys_prompt=(
                ACT_DEFAULT_SINGLE_PROMPT_TEMPLATE
                if single_prompt
                else ACT_DEFAULT_PROMPT_TEMPLATE
            ),
            **kwargs,
        )

    def __init__(self, **kwargs):
        # set sys_prompt if not provided
        if "sys_prompt" not in kwargs:
            single_prompt = kwargs.get("single_prompt", False)
            kwargs["sys_prompt"] = (
                REACT_DEFAULT_SINGLE_PROMPT_TEMPLATE
                if single_prompt
                else REACT_DEFAULT_PROMPT_TEMPLATE
            )

        super().__init__(**kwargs)
        if self.single_prompt:
            self._react_module = ReActModuleSinglePrompt(
                self.llm_model, self.sys_prompt, self.tool_description_method
            )
        else:
            self._react_module = ReActModule(
                self.llm_model, self.sys_prompt, self.tool_description_method
            )

    async def init_state(self, tools: list[Tool]) -> SimpleAgentState:
        return SimpleAgentState(
            tools=tools,
            hide_old_env_states=self.hide_old_env_states,
            hide_old_action_content=self.hide_old_action_content,
            sliding_window=self.sliding_window,
        )

    @staticmethod
    def after_retry_failure_log(retry_state: RetryCallState):
        logger.error(
            f"Failed across {retry_state.attempt_number} attempts to run get_asv given"
            f" arguments {retry_state.args} and kwargs {retry_state.kwargs}."
        )
        # NOTE: this blows up with the underlying exception... it isn't wrapped in a
        # RetryError like normal tenacity
        return cast("Future", retry_state.outcome).result()

    @retry(
        retry=retry_if_exception_type(MalformedMessageError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        stop=stop_after_attempt(5),
        retry_error_callback=after_retry_failure_log,
    )
    @compute_graph()
    async def get_asv(
        self, agent_state: SimpleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        obs = obs.copy()  # Keep original obs, as we edit the content below
        if self.single_prompt:
            for i, m in enumerate(obs):
                if isinstance(m, ToolResponseMessage):
                    obs[i] = Message(content=f"Observation: {m.content}")
        else:
            for i, m in enumerate(obs):
                if isinstance(m, ToolResponseMessage):
                    obs[i] = m.model_copy(
                        update={"content": f"Observation: {m.content}"}
                    )
        next_state = agent_state.get_next_state(obs=obs)
        action_selection_result, new_messages = await self._react_module(
            messages=next_state.messages, tools=next_state.tools
        )
        next_state.messages = [*next_state.messages, *new_messages]
        return action_selection_result, next_state, 0.0
