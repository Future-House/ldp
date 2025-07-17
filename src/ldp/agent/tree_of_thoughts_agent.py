"""Module for the Tree of Thoughts agent.

This module defines the Tree of Thoughts agent which uses a language model to generate and evaluate possible
steps in a puzzle or problem-solving environment. The agent employs a tree search mechanism to explore different
solutions and selects the most promising ones based on evaluations.

This module is based on the following paper: https://openreview.net/forum?id=5Xc1ecxO1h

Note: TreeofThoughtsAgent is currently tested as a baseline agent for Game of 24. It does not yet support tool calls
that operate on the intermediate reasoning steps. This would probably entail a redefinition of the POMDP to
undertake intermediate reasoning steps as environment steps.
"""

import logging
import operator
from collections.abc import Callable
from typing import Any

from aviary.core import Message, Tool, ToolCall, ToolRequestMessage
from lmi import CommonLLMNames
from pydantic import BaseModel, ConfigDict, Field

from ldp.graph import FxnOp, LLMCallOp, OpResult, compute_graph
from ldp.llms import prepend_sys

from . import DEFAULT_LLM_COMPLETION_TIMEOUT
from .agent import Agent
from .simple_agent import SimpleAgentState

logger = logging.getLogger(__name__)


class TreeofThoughtsAgent(BaseModel, Agent[SimpleAgentState]):
    """Tree of Thoughts Agent.

    This agent uses a tree search mechanism combined with an LLM to generate and evaluate
    possible steps in a problem-solving environment. It is designed to explore different solutions
    and select the most promising ones based on a heuristic evaluation function.
    """

    # Freeze to ensure the only mutation happens in either the agent state (which is
    # passed around) or in the internal Ops
    model_config = ConfigDict(frozen=True)

    llm_model: dict[str, Any] = Field(
        default={
            "name": CommonLLMNames.GPT_4O.value,
            "temperature": 0.1,
            "timeout": DEFAULT_LLM_COMPLETION_TIMEOUT,
        },
        description="Starting configuration for the LLM model.",
    )
    value_prompt_func: Callable[[str, str], str] = Field(
        default=lambda x, y: f"Value prompt for input: {x}, current path: {y}",
        description="Function to format value prompt template.",
    )
    proposal_prompt_func: Callable[[str, str], str] = Field(
        default=lambda x, y: f"Proposal prompt for input: {x}, current path: {y}",
        description="Function to format proposal prompt template.",
    )
    hide_old_env_states: bool = Field(
        default=False,
        description="See SimpleAgentState.hide_old_env_states.",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prepend_op = FxnOp(prepend_sys)
        self._llm_call_op = LLMCallOp()

    async def init_state(self, tools: list[Tool]) -> SimpleAgentState:
        return SimpleAgentState(
            tools=tools, hide_old_env_states=self.hide_old_env_states
        )

    @compute_graph()
    async def get_asv(  # type: ignore[override]
        self,
        agent_state: SimpleAgentState,
        obs: list[Message],
        eval_function: Callable[[str, list[str]], float],
        n_steps: int = 0,
        n_select_samples: int = 0,
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        """Generate and evaluate possible steps in the problem-solving process.

        Args:
            agent_state: The current state of the agent.
            obs: The observations provided to the agent.
            eval_function: Function to evaluate the generated paths in the tree.
            n_steps: Number of steps to generate. Defaults to 0. Dictated by the environment.
            n_select_samples: Number of tree nodes to select to explore in each step. Defaults to 0.

        Returns:
            The result of the operation, the new state of the agent, and the number representing the value (0).
        """
        new_state = agent_state.get_next_state()

        x = str(obs[0].content)  # Current problem input
        current_paths = [""]  # current candidate paths through the tree

        for step in range(n_steps):
            logger.info(f"Step {step}")

            # propose candidate paths
            candidate_paths = []
            for path in current_paths:
                proposal_prompt_init = self.proposal_prompt_func(x, path)
                proposal_msgs = await self._prepend_op(
                    new_state.messages, sys_content=proposal_prompt_init
                )
                proposal = await self._llm_call_op(self.llm_model, msgs=proposal_msgs)
                # Append candidate paths to the current paths
                candidate_paths += [
                    path + _ + "\n"
                    for _ in (proposal.value.content or "").split("\n")
                    if _
                ]

            # score candidate paths
            values = []
            for path in candidate_paths:
                value_prompt_init = self.value_prompt_func(x, path)
                value_msgs = await self._prepend_op(
                    new_state.messages, sys_content=value_prompt_init
                )
                value_outputs = await self._llm_call_op(self.llm_model, msgs=value_msgs)
                values.append(eval_function(path, [value_outputs.value.content or ""]))

            # greedy selection
            values_with_index = [(v, i) for i, v in enumerate(values)]
            sorted_values = sorted(
                values_with_index, key=operator.itemgetter(0), reverse=True
            )
            select_ids = [i for _, i in sorted_values[:n_select_samples]]
            select_new_paths = [candidate_paths[select_id] for select_id in select_ids]
            current_paths = select_new_paths

        # Generate tool calls for the selected answer
        tool_calls = [
            ToolCall.from_tool(tool, *[current_paths[0]]) for tool in new_state.tools
        ]
        result = ToolRequestMessage(content=current_paths[0], tool_calls=tool_calls)

        new_state.messages = [*new_state.messages, result]
        return await Agent.wrap_action(result), new_state, 0.0
