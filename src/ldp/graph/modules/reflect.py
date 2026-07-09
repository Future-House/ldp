from typing import Any

from aviary.core import Message
from pydantic import BaseModel, Field

from ldp.graph import ConfigOp, FxnOp, LLMCallOp, PromptOp, compute_graph
from ldp.graph.ops import ResultOrValue
from ldp.llms import append_to_sys, indent_xml


class ReflectModuleConfig(BaseModel):
    """Configuration for the ReflectModuleConfig."""

    llm_model: dict[str, Any] = Field(
        default={"name": "gpt-3.5-turbo"},
        description="Starting configuration for the LLM model.",
    )


class ReflectModule:
    """A module that simply gives an LLM to reflect on an input."""

    def __init__(self, start_config: ReflectModuleConfig):
        self.llm_call_op = LLMCallOp()
        self.prompt_op = PromptOp(
            "Consider a proposed response based on context. Reflect on the response"
            " within <thought> tags then conclude with a possibly revised response"
            " within <final-response> tags."
        )
        self.config_op = ConfigOp[ReflectModuleConfig](config=start_config)
        self.llm_config_op = FxnOp[dict](lambda c: c.llm_model)
        self.package_fxn = FxnOp(append_to_sys)

        def extract_msg(msg: Message, backup_response: str) -> str:
            msg_str = msg.content
            if msg_str and "<final-response>" in msg_str:
                return msg_str.split("<final-response>")[1].split("</final-response>")[
                    0
                ]
            if msg_str and "<response>" in msg_str:
                return msg_str.split("<response>")[1].split("</response>")[0]
            return backup_response

        self.extract_msg = FxnOp(extract_msg)

    @compute_graph()
    async def __call__(
        self, context: ResultOrValue[str], response: ResultOrValue[str]
    ) -> ResultOrValue[str]:
        llm_config = await self.llm_config_op(await self.config_op())
        sys_str = await self.prompt_op()
        user_str = indent_xml(
            f"<context>{context}</context><response>{response}</response>"
        )
        msg = await self.package_fxn(user_str, sys_str)
        llm_result = await self.llm_call_op(llm_config, msg)
        return await self.extract_msg(llm_result, response)
