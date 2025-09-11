from __future__ import annotations

import json
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

from aviary.core import MalformedMessageError, Message, Messages
from aviary.tools import Tool, ToolCall, ToolRequestMessage
from transformers import LogitsProcessorList

from ldp.graph.gradient_estimators import assign_constant_grads
from ldp.graph.op_utils import CallID, get_call_id, get_training_mode
from ldp.graph.ops import GradInType, Op, OpCtx, ResultOrValue
from ldp.nn.handlers.transformer_handler import (
    AsyncTransformerInterface,
    LMType,
    ParallelModeConfig,
    TransformerHandlerConfig,
    collate_fn_transformer_left_pad,
    decollate_fn_transformer_decoder,
)
from ldp.nn.lm_config import LMConfig

if TYPE_CHECKING:
    import tree


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

        self.llm_call_kwargs = {"logits_processor": LogitsProcessorList()}

    @staticmethod
    def prep_messages_for_tokenizer(xi: Messages) -> list[dict]:
        result: list[dict] = []
        for msg in xi:
            content = msg.content
            if isinstance(msg, ToolRequestMessage):
                assert len(msg.tool_calls) == 1, (
                    "Support parsing only single tool call for now"
                )
                tool_call = msg.tool_calls[0]
                # TODO: document where this format is coming from. Is this a Huggingface chat template syntax?
                content_dict = {
                    "name": tool_call.function.name,
                    "parameters": tool_call.function.arguments,
                    "thought": msg.content,
                }
                content = json.dumps(content_dict)
            assert content is not None, "content is None, doesn't make sense"

            result.append({"role": msg.role, "content": content})
        return result

    @staticmethod
    def prep_tools_for_tokenizer(
        tools: list[Tool] | None,
    ) -> list[dict[Any, Any]] | None:
        """Prepare tools for the tokenizer by transforming them into a JSON schema."""
        if not tools:
            return None

        # TODO: should be able to switch to tool.info.model_dump() here
        tools_list: list[dict[Any, Any]] = []
        for tool in tools:
            if tool.info.parameters is None:
                raise NotImplementedError(
                    "Didn't yet handle serializing tools without parameters."
                )

            tools_list.append({
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
                        for prop_name, prop_details in tool.info.get_properties().items()
                    },
                    "required": tool.info.parameters.required,
                },
            })
        return tools_list

    @staticmethod
    def _parse_tool_request(out_text: str, tools: list[Tool]) -> ToolRequestMessage:
        """Parse the output text to extract the tool request.

        TODO: see if this needs to be configurable, e.g. for different model
        output formats that we want to experiment with.
        """
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

    async def forward(
        self,
        xi: list[Message],
        temperature: float = 1.0,
        max_new_tokens: int = 10,
        tools: list[Tool] | None = None,
        **kwargs: dict[str, Any],
    ) -> Message:
        call_id = get_call_id()
        inputs = self.prep_messages_for_tokenizer(xi)
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
            out_msg = self._parse_tool_request(out_text, tools)

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
