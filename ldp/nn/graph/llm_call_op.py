from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, ClassVar

import tree
from aviary.core import (
    MalformedMessageError,
    Message,
    Messages,
    ToolCall,
    ToolRequestMessage,
    Tools,
)
from transformers import LogitsProcessorList

from ldp.graph.gradient_estimators import assign_constant_grads
from ldp.graph.op_utils import CallID, get_call_id, get_training_mode
from ldp.graph.ops import GradInType, Op, OpCtx, ResultOrValue

from ..handlers.transformer_handler import (  # noqa: TID252
    AsyncTransformerInterface,
    LMType,
    ParallelModeConfig,
    TransformerHandlerConfig,
    collate_fn_transformer_left_pad,
    decollate_fn_transformer_decoder,
)
from ..lm_config import LMConfig  # noqa: TID252

logger = logging.getLogger(__name__)


class MessageAndToolParser(ABC):
    """Base class to define how we translate between (messages, tools) and strings."""

    supported_templates: ClassVar[set[str]] = set()

    @abstractmethod
    @classmethod
    def get_message_content(cls, msg: Message) -> str | None:
        """Represents a message as a string."""

    @abstractmethod
    @classmethod
    def prep_tools_for_tokenizer(cls, tools: Tools | None) -> list[dict] | None:
        """Prepares tools for tokenization."""

    @abstractmethod
    @classmethod
    def parse_tool_request_message(
        cls, out_text: str, tools: Tools
    ) -> ToolRequestMessage:
        """Parses the output text from a tool request message."""

    @classmethod
    def prep_messages_for_tokenizer(cls, msgs: Messages) -> list[dict]:
        """Prepares message history for tokenization."""
        result: list[dict] = []
        for msg in msgs:
            content = cls.get_message_content(msg)
            assert content is not None, f"Content should not be None: {msg!r}"
            result.append({"role": msg.role, "content": content})
        return result


class Llama31Parser(MessageAndToolParser):
    """Follows the Llama 3.1 syntax.

    See details:
    https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#-tool-calling-(8b/70b/405b)-
    """

    supported_templates: ClassVar[set[str]] = {
        "llama2_chat_template_ori.jinja",
        "llama3.1_chat_template_hf.jinja",
        "llama3.1_chat_template_nothought.jinja",
        "llama3.1_chat_template_thought.jinja",
        "llama3.1_chat_template_vllm.jinja",
        "llama3_chat_template_ori.jinja",
    }

    @classmethod
    def get_message_content(cls, msg: Message) -> str | None:
        if isinstance(msg, ToolRequestMessage):
            assert len(msg.tool_calls) == 1, (
                "Support parsing only single tool call for now"
            )
            tool_call = msg.tool_calls[0]
            content_dict = {
                "name": tool_call.function.name,
                "parameters": tool_call.function.arguments,
                "thought": msg.content,
            }
            return json.dumps(content_dict)

        return msg.content

    @classmethod
    def prep_tools_for_tokenizer(cls, tools: Tools | None) -> list[dict] | None:
        if not tools:
            return None

        # TODO: should be able to switch to tool.info.model_dump() here
        return [
            {
                "name": tool.info.name,
                "description": tool.info.description,
                "parameters": {
                    "type": tool.info.parameters.type,
                    "properties": {
                        prop_name: {
                            "type": prop_details.get("type"),
                            "description": prop_details.get("description"),
                            "title": prop_details.get("title"),
                        }
                        for prop_name, prop_details in tool.info.parameters.properties.items()
                    },
                    "required": tool.info.parameters.required,
                },
            }
            for tool in tools
        ]

    @classmethod
    def parse_tool_request_message(
        cls, out_text: str, tools: Tools
    ) -> ToolRequestMessage:
        try:
            tool_request = json.loads(out_text)
            tool_name = tool_request["name"]
            tool = next(t for t in tools if t.info.name == tool_name)
            tool_thought = tool_request.get("thought", "")
            tool_parameters = tool_request.get("parameters", {})
            return ToolRequestMessage(
                tool_calls=[ToolCall.from_tool(tool, **tool_parameters)],
                content=tool_thought,
            )
        except StopIteration as exc:
            raise MalformedMessageError(
                f"Tool {tool_name} not found in tools."
            ) from exc
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse tools call message: {out_text}") from err


class LocalLLMCallOp(Op[Message]):
    """An Op that samples a token sequence from a local language model."""

    CTX_INPUTS_PREP_KEY: ClassVar[str] = "inputs_prepared"
    CTX_TOOLS_PREP_KEY: ClassVar[str] = "tools_prepared"
    CTX_OUTPUT_PREP_KEY: ClassVar[str] = "outputs_prepared"

    model_name: str

    def __init__(
        self,
        model_config: LMConfig,
        batch_size: int = 1,
        max_wait_interval: float = 0.1,
        parallel_mode_config: ParallelModeConfig | None = None,
        parser: type[MessageAndToolParser] = Llama31Parser,
    ) -> None:
        super().__init__()

        pad_token_id = model_config.get_tokenizer().pad_token_id

        handler_config = TransformerHandlerConfig(
            # configurable
            lm_config=model_config,
            batch_size=batch_size,
            max_wait_interval=max_wait_interval,
            parallel_mode_config=parallel_mode_config,
            # constant configuration
            lm_type=LMType.GENERATION,
            module_call_fn=AsyncTransformerInterface.model_generate,
            collate_fn=partial(
                collate_fn_transformer_left_pad, pad_token_id=pad_token_id
            ),
            decollate_fn=decollate_fn_transformer_decoder,
        )
        self.model_handler = handler_config.make_async_module()
        self.model_name = model_config.model

        self.prep_messages_for_tokenizer = parser.prep_messages_for_tokenizer
        self.prep_tools_for_tokenizer = parser.prep_tools_for_tokenizer
        self.parse_tool_request_message = parser.parse_tool_request_message
        if model_config.chat_template not in parser.supported_templates:
            logger.warning(
                f"Chat template {model_config.chat_template!r} not in "
                f"{parser.__class__.__name__}.supported templates."
            )

        self.llm_call_kwargs = {"logits_processor": LogitsProcessorList()}

    async def forward(
        self,
        msgs: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int = 10,
        tools: Tools | None = None,
        **kwargs: dict[str, Any],
    ) -> Message:
        call_id = get_call_id()
        inputs = self.prep_messages_for_tokenizer(msgs)
        tools_json = self.prep_tools_for_tokenizer(tools)
        if get_training_mode():
            self.ctx.update(call_id, LocalLLMCallOp.CTX_INPUTS_PREP_KEY, inputs)
            self.ctx.update(call_id, LocalLLMCallOp.CTX_TOOLS_PREP_KEY, tools_json)

        out_text, logprobs = await self.model_handler(
            inputs,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            tools_json=tools_json,
            output_scores=True,
            return_legacy_cache=False,
            return_dict_in_generate=True,
            do_sample=temperature > 0,
            **self.llm_call_kwargs,
            **kwargs,
        )

        out_msg = Message(role="assistant", content=out_text)
        if tools and out_text.startswith("{"):
            out_msg = self.parse_tool_request_message(out_text, tools)

        if get_training_mode():
            self.ctx.update(
                call_id,
                LocalLLMCallOp.CTX_OUTPUT_PREP_KEY,
                self.prep_messages_for_tokenizer([out_msg])[0],
            )
            self.ctx.update(call_id, "logprob", logprobs.cpu().tolist())

        return out_msg

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args: list[ResultOrValue],
        input_kwargs: dict[str, ResultOrValue],
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        return assign_constant_grads(input_args, input_kwargs, 0.0, descend=False)
