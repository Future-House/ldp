import asyncio
import logging
from collections.abc import Sequence
from typing import Any, Self, cast

from aviary.message import EnvStateMessage, Message
from aviary.tools import Tool, ToolRequestMessage
from pydantic import BaseModel, ConfigDict, Field

from ldp.agent import Agent, DefaultLLMModelNames, SimpleAgentState
from ldp.graph import (
    DQNOp,
    FxnOp,
    LLMCallOp,
    OpResult,
    ResultOrValue,
    compute_graph,
    get_call_id,
)
from ldp.graph.modules import DQNPolicyModule
from ldp.llms import MultipleCompletionLLMModel, prepend_sys

logger = logging.getLogger(__name__)


class MultipleCompletionLLMCallOp(LLMCallOp):
    """Like LLMCallOp, but samples multiple completions (n>1) for a given input."""

    async def forward(  # type: ignore[override]
        self,
        config: dict,
        msgs: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: (
            Tool | str | None
        ) = MultipleCompletionLLMModel.TOOL_CHOICE_REQUIRED,
    ) -> list[Message]:
        model = MultipleCompletionLLMModel(config=config)

        results = await model.call(messages=msgs, tools=tools, tool_choice=tool_choice)
        if not results:
            raise ValueError("No completions returned from the model.")

        # All completions have the same config, so check the first one
        temperature: float = (results[0].config or {}).get("temperature", 1.0)

        # Compute a Monte Carlo estimate of the logprob of this sequence at the given temperature.
        logprobs = await asyncio.gather(*[
            self.compute_logprob(
                raw_log_p=result.logprob,
                temperature=temperature,
                model=model,
                messages=msgs,
                tools=tools,
                tool_choice=tool_choice,
            )
            for result in results
        ])

        call_id = get_call_id()
        self.ctx.update(call_id, "results", results)
        # This is the logprob of this sequence according to the raw model, without
        # any temperature/top-p distribution shaping.
        self.ctx.update(call_id, "raw_logprobs", [result.logprob for result in results])

        self.ctx.update(call_id, "temperature", temperature)
        self.ctx.update(call_id, "logprob", logprobs)

        return [cast(list[Message], result.messages)[0] for result in results]

    # set the return type to list[Message], not LLMCallOp's inherited Message
    async def __call__(  # type: ignore[override]
        self, *args, **kwargs
    ) -> OpResult[list[Message]]:
        return await super().__call__(*args, **kwargs)  # type: ignore[return-value]

    async def compute_logprob(
        self,
        raw_log_p: float | None,
        temperature: float,
        model: MultipleCompletionLLMModel,
        **model_kwargs,
    ) -> float | None:
        """This method computes a Monte Carlo estimate of logprob for a given temperature."""
        return None  # TODO: finish implementing this using n>1


class HiddenEnvStateMessage(Message):
    content: str = "[Previous environment state - hidden]"


class CollapsibleAgentState(SimpleAgentState):
    """Like SimpleAgentState, but offers the possibility of collapsing EnvStateMessages."""

    def get_next_state(
        self,
        obs: list[Message] | None = None,
        tools: list[Tool] | None = None,
        hide_old_env_states: bool = False,
        **kwargs,
    ) -> Self:
        old_messages = self.messages
        if hide_old_env_states:
            old_messages = [
                HiddenEnvStateMessage() if isinstance(m, EnvStateMessage) else m
                for m in old_messages
            ]

        return type(self)(
            tools=tools if tools is not None else self.tools,
            messages=old_messages + (obs or []),
            **kwargs,
        )


class DQNAgent(BaseModel, Agent[CollapsibleAgentState]):
    """An agent that uses a state-action value function Q(s,a) to select actions.

    This a modification of traditional DQNs [1]. When using a vanilla DQN, the action
    space is assumed to be enumerable, enabling greedy action selection. When sampling
    from an LLM, the action space is large. So instead, we sample from the LLM and use
    the Q network to score the sampled actions. We are then ε-greedy among the sampled
    actions.

    [1] https://arxiv.org/abs/1312.5602
    """

    # Not frozen because we want to change num_actions_to_sample
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

    num_actions_to_sample: int = Field(
        default=1,
        description=(
            "Number of actions to sample from the LLM. "
            "If >1, the Q function will be used to select the "
            "highest-scoring action."
        ),
    )

    llm_model: dict[str, Any] = Field(
        default={"model": DefaultLLMModelNames.OPENAI.value, "temperature": 1.0},
        description=(
            "Model configuration (not trained). "
            "Setting a high default temperature to encourage exploration."
        ),
    )
    sys_prompt: str = Field(
        default="Using tools, complete the given task.",
        description="System prompt. Trainable",
    )
    last_observation_only: bool = Field(
        default=False,
        description=(
            "Whether to only include the last observation when passing "
            "agent state to the DQN."
        ),
    )
    hide_old_env_states: bool = Field(
        default=False,
        description="Whether to hide previous environment states in the agent state.",
    )

    def __init__(
        self,
        dqn: DQNOp | None = None,
        epsilon: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._prepend = FxnOp(prepend_sys)
        self._llm_call = MultipleCompletionLLMCallOp()
        self._action_splitter = FxnOp[ToolRequestMessage](lambda x, i: x[i])
        self._dqn_policy = DQNPolicyModule[ToolRequestMessage](dqn=dqn, epsilon=epsilon)

    async def init_state(self, tools: list[Tool]) -> CollapsibleAgentState:
        return CollapsibleAgentState(tools=tools)

    @staticmethod
    def prepare_state_messages(
        messages: ResultOrValue[Sequence[Message]], last_observation_only: bool
    ) -> Sequence[Message]:
        messages = OpResult.unwrap_value(messages)
        if not last_observation_only:
            return messages

        i_last_agent_message = 0
        for i, m in enumerate(messages):
            if m.role == "assistant":
                i_last_agent_message = i

        return messages[i_last_agent_message:]

    @compute_graph()
    async def get_asv(
        self, agent_state: CollapsibleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], CollapsibleAgentState, float]:
        new_state = agent_state.get_next_state(
            obs, hide_old_env_states=self.hide_old_env_states
        )

        msgs = await self._prepend(new_state.messages, sys_content=self.sys_prompt)
        sampled_actions = cast(
            OpResult[list[ToolRequestMessage]],
            await self._llm_call(
                # Override config's n. Also make sure caching=False, since we always want
                # stochastic samples from the LLM for the DQN.
                self.llm_model | {"n": self.num_actions_to_sample, "caching": False},
                msgs=msgs,
                tools=agent_state.tools,
            ),
        )
        split_actions: list[OpResult[ToolRequestMessage]] = await asyncio.gather(*[
            self._action_splitter(sampled_actions, i)
            for i in range(self.num_actions_to_sample)
        ])

        state_msgs = self.prepare_state_messages(
            new_state.messages, self.last_observation_only
        )
        best_q, best_action = await self._dqn_policy(state_msgs, *split_actions)
        new_state.messages = [*new_state.messages, best_action.value]

        return best_action, new_state, best_q
