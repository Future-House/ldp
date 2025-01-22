from typing import cast

import torch
import torch.distributed as dist
from aviary.core import Message, Tool, ToolRequestMessage

from ldp.agent import Agent, SimpleAgentState
from ldp.graph import OpResult
from ldp.graph.op_utils import compute_graph
from ldp.llms import prepend_sys

from ..graph.llm_call_op import LocalLLMCallOp  # noqa: TID252
from ..handlers.chunking import TensorChunker  # noqa: TID252
from ..handlers.transformer_handler import (  # noqa: TID252
    ParallelModeConfig,
    logits_to_logprobs,
)
from ..lm_config import LMConfig as _LMConfig  # noqa: TID252


class AgentLMConfig(_LMConfig):
    """Adds some additional configuration options for running an LM in an Op."""

    # async batching parameters
    batch_size: int = 1
    max_wait_interval: float = 0.1

    # distribution
    parallel_mode: ParallelModeConfig | None = None

    # sampling parameters
    temperature: float = 1.0
    max_new_tokens: int = 50
    parallel_tool_calls: bool = False


# TODO: consider merging with SimpleAgent
class SimpleLocalLLMAgent(Agent[SimpleAgentState]):
    """Simple agent that can pick and invoke tools with a local language model."""

    def __init__(
        self,
        llm_model: AgentLMConfig,
        sys_prompt: str | None = None,
        hide_old_env_states: bool = False,
    ):
        self.llm_model = llm_model
        self.sys_prompt = sys_prompt
        self.hide_old_env_states = hide_old_env_states

        # Initialize the local LLM operation with configuration specifics
        self._llm_call_op = LocalLLMCallOp(
            model_config=self.llm_model,
            batch_size=self.llm_model.batch_size,
            max_wait_interval=self.llm_model.max_wait_interval,
            parallel_mode_config=self.llm_model.parallel_mode,
        )

    async def init_state(self, tools: list[Tool]) -> SimpleAgentState:
        """Initialize the state with provided tools."""
        return SimpleAgentState(
            tools=tools, hide_old_env_states=self.hide_old_env_states
        )

    @compute_graph()
    async def get_asv(
        self, agent_state: SimpleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        """Generates an action-state-value tuple using the local LLM model."""
        next_state = agent_state.get_next_state(obs)

        # If system prompt exists, prepend it; otherwise, use the existing messages
        messages = (
            prepend_sys(next_state.messages, sys_content=self.sys_prompt)
            if self.sys_prompt is not None
            else next_state.messages
        )

        # Execute the LLM operation call
        result = cast(
            OpResult[Message | ToolRequestMessage],
            await self._llm_call_op(
                xi=messages,
                temperature=self.llm_model.temperature,
                max_new_tokens=self.llm_model.max_new_tokens,
                tools=next_state.tools,
            ),
        )

        # Type-checking for expected output
        if not isinstance(result.value, ToolRequestMessage):
            raise TypeError(
                f"Expected ToolRequestMessage, got {type(result.value)}: {result.value}, history: {messages}"
            )

        # Update state messages with result and return the new state
        next_state.messages = [*next_state.messages, result.value]
        return cast(OpResult[ToolRequestMessage], result), next_state, 0.0

    # TODO: maybe remove these recomputation methods. I added them to debug some things. But idk,
    # maybe they'll come in handy later.
    @staticmethod
    def recompute_logprobs_forward(
        handler, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        num_chunks = dist.get_world_size()
        curr_chunk = dist.get_rank()
        chunker = TensorChunker(num_chunks=num_chunks)
        split_args_list, *_ = chunker.chunkify(input_ids, attention_mask)
        input_ids, attention_mask = split_args_list[curr_chunk]

        input_ids, attention_mask = (
            input_ids.to(handler.module.device),
            attention_mask.to(handler.module.device),
        )
        with torch.no_grad():
            handler.module.eval()
            out = handler.module(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits.float()

    def recompute_logprobs(self, messages_: list[Message], tools_: list[Tool]):
        op = self._llm_call_op
        handler = op.model_handler
        messages = op.prep_messages_for_tokenizer(messages_)
        tools = op.prep_tools_for_tokenizer(tools_)

        encoded = handler.tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            [messages],
            tools=tools,
            return_tensors="pt",
            return_dict=True,
            return_assistant_tokens_mask=True,
            tokenizer_kwargs={
                "return_special_tokens_mask": True,
            },
        )
        am = torch.tensor(encoded["assistant_masks"])
        action_start = (1 - am[0]).argwhere()[-1].item() + 1

        out = handler.wrap_func(
            SimpleLocalLLMAgent.recompute_logprobs_forward,
            worker_agg_fn=torch.concatenate,
        )(encoded["input_ids"], encoded["attention_mask"]).squeeze(0)

        logprobs_reco = logits_to_logprobs(
            out[:-1], encoded["input_ids"].squeeze(0)[1:]
        )
        # +3 to remove the assistant prompt tokens
        return logprobs_reco[action_start + 3 :]
