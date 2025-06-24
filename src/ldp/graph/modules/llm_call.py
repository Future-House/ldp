from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar

from aviary.core import Message

from ldp.graph import ConfigOp, FxnOp, LLMCallOp, OpResult, compute_graph

TParsedMessage = TypeVar("TParsedMessage", bound=Message)


class ParsedLLMCallModule(Generic[TParsedMessage]):
    """Module for a processing-based tool selection, with a learnable configuration."""

    def __init__(
        self, llm_model: dict[str, Any], parser: Callable[..., TParsedMessage]
    ):
        self.config_op = ConfigOp[dict](config=llm_model)
        self.llm_call_op = LLMCallOp()
        self.parse_msg_op = FxnOp(parser)

    @compute_graph()
    async def __call__(
        self, messages: Iterable[Message], *parse_args, **parse_kwargs
    ) -> tuple[OpResult[TParsedMessage], Message]:
        raw_result = await self.llm_call_op(await self.config_op(), msgs=messages)
        return (
            await self.parse_msg_op(raw_result, *parse_args, **parse_kwargs),
            raw_result.value,
        )
