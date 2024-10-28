from collections.abc import Iterable, Mapping
from typing import Any

from aviary.core import Message, ToolRequestMessage

from ldp.graph import FxnOp, OpResult, PromptOp, compute_graph
from ldp.llms import prepend_sys_and_append_sys

from .llm_call import ParsedLLMCallModule


class ThoughtModule:
    @staticmethod
    def _downcast_to_message(message: Message | ToolRequestMessage) -> Message:
        if isinstance(message, ToolRequestMessage):
            # Downcast into a normal Message if the LLM tried to call tools
            return Message(role=message.role, content=message.content)
        return message

    def __init__(
        self, llm_model: dict[str, Any], first_sys_prompt: str, second_sys_prompt: str
    ):
        self.first_sys_prompt_op = PromptOp(first_sys_prompt)
        self.second_sys_prompt_op = PromptOp(second_sys_prompt)
        self.package_msg_op = FxnOp(prepend_sys_and_append_sys)
        self.llm_call = ParsedLLMCallModule[Message](
            llm_model, parser=self._downcast_to_message
        )

    @compute_graph()
    async def __call__(
        self,
        messages: Iterable[Message],
        first_prompt_kwargs: Mapping[str, Any],
        second_prompt_kwargs: Mapping[str, Any],
    ) -> OpResult[Message]:
        packaged_msgs = await self.package_msg_op(
            messages,
            initial_sys_content=await self.first_sys_prompt_op(**first_prompt_kwargs),
            final_sys_content=await self.second_sys_prompt_op(**second_prompt_kwargs),
        )
        return (await self.llm_call(packaged_msgs))[0]  # type: ignore[arg-type]
