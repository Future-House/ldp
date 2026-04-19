__all__ = [
    "CommonLLMNames",
    "LLMModel",
    "LiteLLMModel",
    "PassThroughRouter",
    "estimate_message_tokens",
    "extract_top_logprobs",
    "parse_cached_usage",
    "rate_limited",
    "request_limited",
    "sum_logprobs",
    "validate_json_completion",
]

import asyncio
import contextlib
import functools
import json
import logging
from abc import ABC
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Mapping,
    Sequence,
)
from enum import StrEnum
from inspect import isasyncgenfunction, isawaitable, signature
from typing import Any, ClassVar, ParamSpec, TypeAlias, cast, overload

import litellm
from aviary.core import (
    Message,
    MessagesAdapter,
    Tool,
    ToolCall,
    ToolCallFunction,
    ToolRequestMessage,
    ToolResponseMessage,
    ToolsAdapter,
    ToolSelector,
    is_coroutine_callable,
)
from aviary.message import MalformedMessageError
from litellm import completion_cost
from litellm.types.llms.openai import (
    ErrorEvent,
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIResponse,
)
from litellm.types.utils import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from lmi.config import LLMConfig, ModelSpec
from lmi.constants import (
    CHARACTERS_PER_TOKEN_ASSUMPTION,
    DEFAULT_VERTEX_SAFETY_SETTINGS,
    IS_PYTHON_BELOW_312,
)
from lmi.cost_tracker import track_costs, track_costs_iter
from lmi.exceptions import (
    AllModelsExhaustedError,
    JSONSchemaValidationError,
    ModelRefusalError,
    ResponseValidationError,
)
from lmi.rate_limiter import GLOBAL_LIMITER
from lmi.retry import (
    backoff_seconds,
    should_fallback,
    should_retry,
)
from lmi.types import LLMResult

from . import (
    litellm_patches as _litellm_patches,  # noqa: F401 - In-place apply patches at import
)

logger = logging.getLogger(__name__)


def _convert_content_block_for_responses(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a single Chat Completions content block to Responses API format.

    Aviary format: {"type": "image_url", "image_url": {"url": "..."}}
    Responses API: {"type": "input_image", "image_url": "..."}
    """
    if block.get("type") == "image_url":
        return {
            "type": "input_image",
            "image_url": block.get("image_url", {}).get("url", ""),
        }
    if block.get("type") == "text":
        return {
            "type": "input_text",
            "text": block.get("text", ""),
        }
    return block


def _convert_tool_response_for_responses(
    msg: ToolResponseMessage,
) -> str | list[dict[str, Any]]:
    """Convert tool response content to Responses API format."""
    if not msg.content:
        return ""
    if not msg.content_is_json_str:
        return msg.content
    items = json.loads(msg.content)
    if not isinstance(items, list):
        return msg.content
    return [_convert_content_block_for_responses(item) for item in items]


def _convert_multimodal_content_for_responses(msg: Message) -> list[dict[str, Any]]:
    """Convert multimodal message content to Responses API format."""
    if msg.content is None:
        raise TypeError("Multimodal content cannot be None.")
    items = json.loads(msg.content)
    return [_convert_content_block_for_responses(item) for item in items]


def _convert_to_responses_input(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert aviary Messages to Responses API input format."""
    result = []
    for msg in messages:
        if isinstance(msg, ToolResponseMessage):
            result.append({
                "type": "function_call_output",
                "call_id": msg.tool_call_id,
                "output": _convert_tool_response_for_responses(msg),
            })
        elif isinstance(msg, ToolRequestMessage):
            if msg.content:
                result.append({
                    "type": "message",
                    "role": msg.role,
                    "content": msg.content,
                })
            result.extend(
                {
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.dumps(tc.function.arguments),
                }
                for tc in msg.tool_calls
            )
        elif msg.is_multimodal:
            result.append({
                "type": "message",
                "role": msg.role,
                "content": _convert_multimodal_content_for_responses(msg),
            })
        else:
            result.append({
                "type": "message",
                "role": msg.role,
                "content": msg.content or "",
            })
    return result


def _convert_tools_for_responses(tools: list[dict] | None) -> list[dict] | None:
    """Convert Chat Completions tools to Responses API format."""
    if not tools:
        return None
    result = []
    for tool in tools:
        if tool["type"] == "function":
            func = tool["function"]
            result.append({
                "type": "function",
                "name": func["name"],
                "description": func.get("description"),
                "parameters": func.get("parameters"),
                "strict": False,
            })
        else:
            result.append(tool)
    return result


def _parse_responses_output(
    output: list[ResponseOutputMessage | ResponseFunctionToolCall],
) -> tuple[str | None, list[Message | ToolRequestMessage]]:
    """Convert Responses API output to aviary Messages."""
    text_parts: list[str] = []
    tool_calls = []

    for item in output:
        if item.type == "message":
            text_parts.extend(c.text for c in item.content if c.type == "output_text")
        elif item.type == "function_call":
            arguments = json.loads(item.arguments)
            tool_calls.append(
                ToolCall(
                    id=item.call_id,
                    type="function",
                    function=ToolCallFunction(name=item.name, arguments=arguments),
                )
            )

    if len(text_parts) > 1:
        logger.warning(
            f"Responses API returned {len(text_parts)} output_text parts;"
            " concatenating with newlines."
        )
    text_content = "\n".join(text_parts) if text_parts else None

    messages: list[Message | ToolRequestMessage] = []
    if tool_calls:
        messages.append(
            ToolRequestMessage(
                role="assistant", content=text_content, tool_calls=tool_calls
            )
        )
    elif text_content is not None:
        messages.append(Message(role="assistant", content=text_content))

    return text_content, messages


def _extract_previous_response_id(
    messages: list[Message],
) -> tuple[str | None, list[Message]]:
    """Scan messages for the last response_id in Message.info and split into delta.

    Returns:
        (previous_response_id, messages_to_send): If a response_id is found,
        returns it and only the messages after it. Otherwise returns (None, messages).
    """
    for i in range(len(messages) - 1, -1, -1):
        rid = (messages[i].info or {}).get("response_id")
        if rid is not None:
            return rid, messages[i + 1 :]
    return None, messages


# List of possible refusal flags in finish_reason
REFUSAL_REASON = (
    "content_filter",  # litellm normalizes provider safety signals (including Anthropic "refusal") to this
)


def parse_cached_usage(usage: Usage | None) -> tuple[int | None, int | None]:
    """Parse cached token counts from LiteLLM usage object.

    Args:
        usage: LiteLLM usage object containing token usage metadata.
            None for streaming intermediate chunks (only final chunk includes usage).

    Returns:
        Tuple of (cache_read_tokens, cache_creation_tokens):
        - (None, None) if usage is None or provider doesn't support caching
        - (int, int) where values can be 0 if caching is supported but had no cache hits/creation

    Provider support:
        - OpenAI: cache_read via prompt_tokens_details.cached_tokens (no creation tracking)
        - Anthropic: cache_read and cache_creation via dedicated fields
    """
    if not usage:
        return None, None

    cache_read: int | None = None
    cache_creation: int | None = None

    # Cache reads: Both OpenAI and Anthropic use prompt_tokens_details.cached_tokens
    if hasattr(usage, "prompt_tokens_details"):
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details:
            cached_val = (
                prompt_details.get("cached_tokens")
                if isinstance(prompt_details, dict)
                else getattr(prompt_details, "cached_tokens", None)
            )
            if isinstance(cached_val, int):
                cache_read = cached_val

    # Cache creation: Anthropic-only field (OpenAI doesn't report cache writes)
    if hasattr(usage, "cache_creation_input_tokens"):
        cached_val = getattr(usage, "cache_creation_input_tokens", None)
        if isinstance(cached_val, int):
            cache_creation = cached_val

    return cache_read, cache_creation


def estimate_message_tokens(
    messages: Iterable[Message] | Sequence[dict[str, Any]], model: str, **kwargs
) -> int:
    """Estimate total token count for a list of messages using ``litellm.token_counter``."""
    messages_list = list(messages)
    return litellm.token_counter(
        model=model,
        messages=(
            MessagesAdapter.dump_python(cast(list[Message], messages_list))
            if messages_list and isinstance(messages_list[0], Message)
            else messages_list
        ),
        **kwargs,
    )


if not IS_PYTHON_BELOW_312:
    _DeploymentTypedDictValidator = TypeAdapter(
        list[litellm.DeploymentTypedDict],
        config=ConfigDict(arbitrary_types_allowed=True),
    )

# Yes, this is a hack, it mostly matches
# https://github.com/python-jsonschema/referencing/blob/v0.35.1/referencing/jsonschema.py#L20-L21
JSONSchema: TypeAlias = Mapping[str, Any]


class CommonLLMNames(StrEnum):
    """When you don't want to think about models, just use one from here."""

    # Use these to avoid thinking about exact versions
    GPT_5 = "gpt-5-2025-08-07"
    GPT_5_MINI = "gpt-5-mini-2025-08-07"
    GPT_41 = "gpt-4.1-2025-04-14"
    GPT_4O = "gpt-4o-2024-11-20"
    GPT_35_TURBO = "gpt-3.5-turbo-0125"
    CLAUDE_45_HAIKU = "claude-haiku-4-5-20251001"
    CLAUDE_37_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_45_SONNET = "claude-sonnet-4-5-20250929"
    CLAUDE_46_SONNET = "claude-sonnet-4-6"
    CLAUDE_45_OPUS = "claude-opus-4-5-20251101"
    CLAUDE_46_OPUS = "claude-opus-4-6"

    # Use these when trying to think of a somewhat opinionated default
    OPENAI_BASELINE = "gpt-4o-2024-11-20"  # Fast and decent

    # Use these in unit testing
    OPENAI_TEST = "gpt-4o-mini-2024-07-18"  # Cheap, fast, and not OpenAI's cutting edge
    ANTHROPIC_TEST = (  # Cheap, fast, and not Anthropic's cutting edge
        "claude-haiku-4-5-20251001"
    )


async def _commit_stream(gen: AsyncIterable[LLMResult]) -> AsyncIterable[LLMResult]:
    """Advance `gen` to its first yield and return an iterator that replays it.

    Exceptions raised before the first yield propagate to the caller. Once the
    first chunk has been produced the returned iterator yields it and then
    forwards the rest of `gen` verbatim; any mid-stream error surfaces
    unmodified to the consumer.
    """
    iterator = aiter(gen)
    try:
        first = await anext(iterator)
    except StopAsyncIteration as exc:
        raise RuntimeError("Stream closed before producing any output.") from exc

    async def replay() -> AsyncIterable[LLMResult]:
        yield first
        async for item in iterator:
            yield item

    return replay()


def sum_logprobs(choice: litellm.utils.Choices | list[float]) -> float | None:
    """Calculate the sum of the log probabilities of an LLM completion (a Choices object).

    Args:
        choice: A sequence of choices from the completion or an iterable with logprobs.

    Returns:
        The sum of the log probabilities of the choice.
    """
    if isinstance(choice, litellm.utils.Choices):
        logprob_obj = getattr(choice, "logprobs", None)
        if not logprob_obj:
            return None

        if isinstance(
            logprob_obj, dict | litellm.types.utils.ChoiceLogprobs
        ) and logprob_obj.get("content", None):
            return sum(
                logprob_info["logprob"] for logprob_info in logprob_obj["content"]
            )

    elif isinstance(choice, list):
        return sum(choice)
    return None


def extract_top_logprobs(
    completion: litellm.utils.Choices,
) -> list[list[tuple[str, float]]] | None:
    """Extract the top logprobs from an litellm completion."""
    logprobs_obj = getattr(completion, "logprobs", None)
    if logprobs_obj is None:
        return None

    content = getattr(logprobs_obj, "content", None)
    if not content or not isinstance(content, list):
        return None

    return [
        [(t.token, float(t.logprob)) for t in (getattr(pos, "top_logprobs", []) or [])]
        for pos in content
    ]


def validate_json_completion(
    completion: litellm.ModelResponse,
    output_type: type[BaseModel] | TypeAdapter | JSONSchema,
) -> None:
    """Validate a completion against a JSON schema.

    Args:
        completion: The completion to validate.
        output_type: A Pydantic model, Pydantic type adapter, or a JSON schema to
            validate the completion.
    """
    try:
        for choice in completion.choices:
            if not hasattr(choice, "message") or not choice.message.content:
                continue
            # make sure it is a JSON completion, even if None
            # We do want to modify the underlying message
            # so that users of it can just parse it as expected
            choice.message.content = choice.message.content.split("```json")[-1].split(
                "```"
            )[0]
            if isinstance(output_type, Mapping):  # JSON schema
                litellm.litellm_core_utils.json_validation_rule.validate_schema(
                    schema=dict(output_type), response=choice.message.content
                )
            elif isinstance(output_type, TypeAdapter):
                output_type.validate_json(choice.message.content)
            else:
                output_type.model_validate_json(choice.message.content)
    except ValidationError as err:
        raise JSONSchemaValidationError(
            "The completion does not match the specified schema."
        ) from err


def prepare_args(
    func: Callable[..., Any] | Callable[..., Awaitable],
    completion: str,
    name: str | None,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    with contextlib.suppress(TypeError):
        if "name" in signature(func).parameters:
            return (completion,), {"name": name}
    return (completion,), {}


async def do_callbacks(
    async_callbacks: Iterable[Callable[..., Awaitable]],
    sync_callbacks: Iterable[Callable[..., Any]],
    completion: str,
    name: str | None,
) -> None:
    await asyncio.gather(
        *(
            f(*args, **kwargs)
            for f in async_callbacks
            for args, kwargs in (prepare_args(f, completion, name),)
        )
    )
    for f in sync_callbacks:
        args, kwargs = prepare_args(f, completion, name)
        f(*args, **kwargs)


class LLMModel(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    llm_type: str | None = None
    name: str
    llm_result_callback: Callable[[LLMResult], Any | Awaitable[Any]] | None = Field(
        default=None,
        description=(
            "An async callback that will be executed on each"
            " LLMResult (different than callbacks that execute on each completion)"
        ),
        exclude=True,
    )
    config: dict = Field(default_factory=dict)

    async def acompletion(
        self, messages: list[Message], *, spec: ModelSpec | None = None, **kwargs
    ) -> list[LLMResult]:
        """Issue one completion request against the model given by `spec`.

        `spec` supplies the model name and per-request kwargs (api_base,
        api_key, timeout, extra_params). When None, subclasses default to the
        primary entry in `llm_config`.
        """
        raise NotImplementedError

    async def acompletion_iter(
        self, messages: list[Message], *, spec: ModelSpec | None = None, **kwargs
    ) -> AsyncIterable[LLMResult]:
        """Stream one completion from the model given by `spec`.

        See `acompletion` for the `spec` contract.
        """
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # gross approximation

    def __str__(self) -> str:
        return f"{type(self).__name__} {self.name}"

    # SEE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    # > `none` means the model will not call any tool and instead generates a message.
    # > `auto` means the model can pick between generating a message or calling one or more tools.
    # > `required` means the model must call one or more tools.
    NO_TOOL_CHOICE: ClassVar[str] = "none"
    MODEL_CHOOSES_TOOL: ClassVar[str] = "auto"
    TOOL_CHOICE_REQUIRED: ClassVar[str] = "required"
    # None means we won't provide a tool_choice to the LLM API
    UNSPECIFIED_TOOL_CHOICE: ClassVar[None] = None

    def _maybe_patch_gemini3_tool_response_messages(
        self, messages: list[Message]
    ) -> list[Message]:
        """As of 2025-11-18, Gemini 3 doesn't accept role="tool" in ToolResponseMessage.

        This function patches it to role="user".
        """
        if "gemini-3" not in self.name:
            return messages

        return [
            m
            if not isinstance(m, ToolResponseMessage)
            else m.model_copy(update={"role": "user"})
            for m in messages
        ]

    async def call(  # noqa: C901, PLR0915
        self,
        messages: list[Message],
        callbacks: (
            Sequence[Callable[..., Any] | Callable[..., Awaitable]] | None
        ) = None,
        name: str | None = None,
        output_type: type[BaseModel] | TypeAdapter | JSONSchema | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Tool | str | None = TOOL_CHOICE_REQUIRED,
        **kwargs,
    ) -> list[LLMResult]:
        """Call the LLM model with the given messages and configuration.

        Args:
            messages: A list of messages to send to the language model.
            callbacks: A list of callback functions to execute.
            name: Optional name for the result.
            output_type: The type of the output.
            tools: A list of tools to use.
            tool_choice: The tool choice to use.
            kwargs: Additional keyword arguments for the chat completion.

        Returns:
            A list of LLMResult objects containing the result of the call.

        Raises:
            ValueError: If the LLM type is unknown.
        """
        messages = self._maybe_patch_gemini3_tool_response_messages(messages)
        # Shallow copy because downstream code only mutates top-level keys.
        # Trying to avoid a deepcopy if not needed. If a future edit adds
        # a nested in-place mutation here, this needs revisiting.
        chat_kwargs = dict(kwargs)
        # if using the config for an LLMModel,
        # there may be a nested 'config' key
        # that can't be used by chat
        chat_kwargs.pop("config", None)
        n = chat_kwargs.get("n", self.config.get("n", 1))
        if n < 1:
            raise ValueError("Number of completions (n) must be >= 1.")

        # deal with tools
        if tools:
            chat_kwargs["tools"] = ToolsAdapter.dump_python(
                tools, exclude_none=True, by_alias=True
            )
            if tool_choice is not None:
                chat_kwargs["tool_choice"] = (
                    {
                        "type": "function",
                        "function": {"name": tool_choice.info.name},
                    }
                    if isinstance(tool_choice, Tool)
                    else tool_choice
                )
        else:
            chat_kwargs["tools"] = tools  # Allows for empty tools list

        # deal with specifying output type
        if isinstance(output_type, Mapping):  # Use structured outputs
            model_name: str = chat_kwargs.get("model", self.name)
            if not litellm.supports_response_schema(model_name, None):
                raise ValueError(f"Model {model_name} does not support JSON schema.")

            chat_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "strict": True,
                    # SEE: https://platform.openai.com/docs/guides/structured-outputs#additionalproperties-false-must-always-be-set-in-objects
                    "schema": dict(output_type) | {"additionalProperties": False},
                    "name": output_type["title"],  # Required by OpenAI as of 12/3/2024
                },
            }
        elif output_type is not None:  # Use JSON mode
            if isinstance(output_type, TypeAdapter):
                schema: str = json.dumps(output_type.json_schema())
            else:
                schema = json.dumps(output_type.model_json_schema())
            schema_msg = f"Respond following this JSON schema:\n\n{schema}"
            # Get the system prompt and its index, or the index to add it
            i, system_prompt = next(
                ((i, m) for i, m in enumerate(messages) if m.role == "system"),
                (0, None),
            )
            messages = [
                *messages[:i],
                (
                    system_prompt.append_text(schema_msg, inplace=False)
                    if system_prompt
                    else Message(role="system", content=schema_msg)
                ),
                *messages[i + 1 if system_prompt else i :],
            ]
            chat_kwargs["response_format"] = {"type": "json_object"}

        messages = [
            (
                m
                if not isinstance(m, ToolRequestMessage) or m.tool_calls
                # OpenAI doesn't allow for empty tool_calls lists, so downcast empty
                # ToolRequestMessage to Message here
                else Message(role=m.role, content=m.content)
            )
            for m in messages
        ]

        start_clock = asyncio.get_running_loop().time()
        streaming = callbacks is not None
        if streaming:
            if tools:
                raise NotImplementedError("Using tools with callbacks is not supported")
            if n > 1:
                raise NotImplementedError(
                    "Multiple completions with callbacks is not supported"
                )

        dispatch_result = await self._run_with_fallbacks(  # type: ignore[attr-defined]
            self._dispatch,  # type: ignore[attr-defined]
            messages=messages,
            streaming=streaming,
            **chat_kwargs,
        )

        if streaming:
            assert callbacks is not None
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]
            results: list[LLMResult] = []
            async for result in dispatch_result:
                if result.text:
                    if result.seconds_to_first_token == 0:
                        result.seconds_to_first_token = (
                            asyncio.get_running_loop().time() - start_clock
                        )
                    await do_callbacks(
                        async_callbacks, sync_callbacks, result.text, name
                    )
                results.append(result)
        else:
            results = dispatch_result

        for result in results:
            if not result.completion_count and result.text is not None:
                result.completion_count = self.count_tokens(result.text)
            result.seconds_to_last_token = (
                asyncio.get_running_loop().time() - start_clock
            )
            result.name = name
            if self.llm_result_callback:
                possibly_awaitable_result = self.llm_result_callback(result)
                if isawaitable(possibly_awaitable_result):
                    await possibly_awaitable_result
        return results

    async def call_single(
        self,
        messages: list[Message] | str,
        callbacks: (
            Sequence[Callable[..., Any] | Callable[..., Awaitable]] | None
        ) = None,
        name: str | None = None,
        output_type: type[BaseModel] | TypeAdapter | JSONSchema | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Tool | str | None = TOOL_CHOICE_REQUIRED,
        **kwargs,
    ) -> LLMResult:
        if isinstance(messages, str):
            # convenience for single message
            messages = [Message(content=messages)]
        results = await self.call(
            messages,
            callbacks,
            name,
            output_type,
            tools,
            tool_choice,
            n=1,
            **kwargs,
        )
        if not results:
            raise ValueError(
                f"Got 0 results from model {kwargs.get('model') or self.name!r},"
                f" given {len(messages)} message(s),"
                f" {len(tools) if tools is not None else None} tools,"
                f" and {tool_choice!r} tool choice."
            )
        if len(results) > 1:
            # Can be caused by issues like https://github.com/BerriAI/litellm/issues/12298
            logger.warning(
                "Got %d results when expecting just one from model %r"
                " (n=1). Using first result. All results:\n%s",
                len(results),
                kwargs.get("model") or self.name,
                "\n---\n".join(str(r) for r in results),
            )
        return results[0]


P = ParamSpec("P")


@overload
def rate_limited(
    func: Callable[P, Coroutine[Any, Any, list[LLMResult]]],
) -> Callable[P, Coroutine[Any, Any, list[LLMResult]]]: ...


@overload
def rate_limited(
    func: Callable[P, AsyncIterable[LLMResult]],
) -> Callable[P, Coroutine[Any, Any, AsyncIterable[LLMResult]]]: ...


def rate_limited(func):
    """Decorator to rate limit relevant methods of an LLMModel."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, "check_rate_limit"):
            raise NotImplementedError(
                f"Model {self.name} must have a `check_rate_limit` method."
            )

        # Estimate token count based on input
        if func.__name__ in {"acompletion", "acompletion_iter"}:
            messages = args[0] if args else kwargs.get("messages", [])
            token_count = len(str(messages)) / CHARACTERS_PER_TOKEN_ASSUMPTION
        else:
            token_count = 0  # Default if method is unknown

        await self.check_rate_limit(token_count)

        # If wrapping a generator, count the tokens for each
        # portion before yielding
        if isasyncgenfunction(func):

            async def rate_limited_generator() -> AsyncIterable[LLMResult]:
                async for item in func(self, *args, **kwargs):
                    token_count = 0
                    if isinstance(item, LLMResult):
                        token_count = int(
                            len(item.text or "") / CHARACTERS_PER_TOKEN_ASSUMPTION
                        )
                    await self.check_rate_limit(token_count)
                    yield item

            return rate_limited_generator()

        # We checked isasyncgenfunction above, so this must be an Awaitable
        result = await func(self, *args, **kwargs)
        if func.__name__ == "acompletion" and isinstance(result, list):
            await self.check_rate_limit(sum(r.completion_count for r in result))
        return result

    return wrapper


@overload
def request_limited(
    func: Callable[P, Coroutine[Any, Any, list[LLMResult]]],
) -> Callable[P, Coroutine[Any, Any, list[LLMResult]]]: ...


@overload
def request_limited(
    func: Callable[P, Coroutine[Any, Any, AsyncIterable[LLMResult]]],
) -> Callable[P, Coroutine[Any, Any, AsyncIterable[LLMResult]]]: ...


def request_limited(func):
    """Decorator to limit requests per minute for LLMModel methods."""

    @functools.wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not hasattr(self, "check_request_limit"):
            raise NotImplementedError(
                f"Model {self.name} must have a `check_request_limit` method."
            )

        await self.check_request_limit()

        if isasyncgenfunction(func):

            async def request_limited_generator() -> AsyncIterable[LLMResult]:
                first_item = True
                async for item in func(self, *args, **kwargs):
                    # Skip rate limit check for first item since we already checked at generator start
                    if not first_item:
                        await self.check_request_limit()
                    else:
                        first_item = False
                    yield item

            return request_limited_generator()
        return await func(self, *args, **kwargs)

    return wrapper


class PassThroughRouter(litellm.Router):  # TODO: add rate_limited
    """Router that is just a wrapper on LiteLLM's normal free functions."""

    def __init__(self, **kwargs):
        self._default_kwargs = kwargs

    async def atext_completion(self, *args, **kwargs):
        return await litellm.atext_completion(*args, **(self._default_kwargs | kwargs))

    async def acompletion(self, *args, **kwargs):
        return await litellm.acompletion(*args, **(self._default_kwargs | kwargs))

    async def aembedding(self, *args, **kwargs):
        return await litellm.aembedding(*args, **(self._default_kwargs | kwargs))

    async def aresponses(self, *args, **kwargs):
        return await litellm.aresponses(*args, **(self._default_kwargs | kwargs))


def default_tool_parser(
    choice: litellm.utils.Choices, tools: list[dict] | None
) -> Message | ToolRequestMessage:
    msg_type = (
        ToolRequestMessage
        if choice.finish_reason == "tool_calls"
        or getattr(choice.message, "tool_calls", None) is not None
        else Message
    )
    serialized_message = choice.message.model_dump()
    if (
        # Confirm we explicitly received an empty tool list, so we don't unnecessarily
        # make a tool request message over a normal message
        tools is not None
        and not tools  # Confirm it's the empty tools special case
        and not serialized_message.get("tool_calls")  # Don't clobber anything
    ):
        # This is a design decision made to simplify
        # downstream language agent logic, where:
        # 1. We wanted the presence of tools, even if the list is empty,
        #    to lead to a ToolRequestMessage
        # 2. However, OpenAI gpt-4o returns null tool_calls if tools is empty,
        #    not empty tool_calls, which leads to a plain Message
        # 3. So, we add this special case to make a ToolRequestMessage
        serialized_message["tool_calls"] = []
        msg_type = ToolRequestMessage
    return msg_type(**serialized_message)


class LiteLLMModel(LLMModel):
    """A wrapper around the litellm library."""

    model_config = ConfigDict(extra="forbid")

    name: str = CommonLLMNames.GPT_4O.value

    tool_parser: (
        Callable[
            [litellm.utils.Choices, list[dict] | None],
            Message | ToolRequestMessage,
        ]
        | Callable[
            [str, list[dict] | None],
            Message | ToolRequestMessage,
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Custom parser for converting LLM completions to tool requests. "
            "Accepts either `(completion: str, tools: list[dict] | None) -> ToolRequestMessage | Message` "
            "or `(choice: litellm.utils.Choices, tools: list[dict] | None) -> ToolRequestMessage | Message`. "
            "Returns `aviary.core.ToolRequestMessage` on successful parsing, `aviary.core.Message` otherwise."
        ),
    )
    config: dict = Field(
        default_factory=dict,
        description=(
            "Legacy dict-shaped configuration for backward compatibility. Accepts"
            " a `model_list` entry (litellm Router layout) plus optional"
            " `rate_limit` / `request_limit` dicts keyed by model group name for"
            " tokens-per-minute and requests-per-minute throttling. New code"
            " should use `llm_config` instead."
        ),
    )
    llm_config: LLMConfig | None = Field(
        default=None,
        description=(
            "Typed model chain. When unset, this is synthesized from `config`"
            " (via `LLMConfig.from_legacy_dict`) during validation, so callers"
            " passing the legacy `config` dict still work."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def maybe_set_config_attribute(cls, input_data: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
        """
        Set the config attribute if it is not provided.

        If name is not provided, uses the default name.
        If a user only gives a name, make a sensible config dict for them.
        """
        # Shallow copy: we only mutate top-level keys (`update` / `pop` on `data["config"]`); everything
        # nested below is read or replaced wholesale. If a future edit adds
        # a nested in-place mutation here, this needs revisiting.
        data = dict(input_data)
        if "config" in data:
            data["config"] = dict(data["config"])

        # unnest the config key if it's nested
        if "config" in data and "config" in data["config"]:
            data["config"].update(data["config"]["config"])
            data["config"].pop("config")

        if "config" not in data:
            data["config"] = {}
        if "name" not in data:
            data["name"] = data["config"].get("name", cls.model_fields["name"].default)
        if "model_list" not in data["config"]:
            try:
                is_openai_model = "openai" in litellm.get_llm_provider(data["name"])
            except litellm.BadRequestError:  # LiteLLM doesn't have provider registered
                is_openai_model = False
            max_tokens = data["config"].get("max_tokens")
            if (
                "logprobs" in data["config"] or "top_logprobs" in data["config"]
            ) and not is_openai_model:
                logger.warning(
                    "Ignoring token logprobs for non-OpenAI model %s, as they are not supported.",
                    data["name"],
                )
            data["config"] = {
                "model_list": [
                    {
                        "model_name": data["name"],
                        "litellm_params": (
                            {
                                "model": data["name"],
                                "n": data["config"].get("n", 1),
                                "temperature": data["config"].get("temperature", 1.0),
                                "max_tokens": data["config"].get("max_tokens", 4096),
                            }
                            | (
                                {}
                                if "gemini" not in data["name"]
                                else {"safety_settings": DEFAULT_VERTEX_SAFETY_SETTINGS}
                            )
                            | ({} if max_tokens else {"max_tokens": max_tokens})
                            | (
                                {}
                                if "logprobs" not in data["config"]
                                or not is_openai_model
                                else {"logprobs": data["config"]["logprobs"]}
                            )
                            | (
                                {}
                                if "top_logprobs" not in data["config"]
                                or not is_openai_model
                                else {"top_logprobs": data["config"]["top_logprobs"]}
                            )
                        ),
                    }
                ],
            } | data["config"]

        if "tool_parser" in data["config"]:
            data["tool_parser"] = data["config"].pop("tool_parser")

        # we only support one "model name" for now, here we validate
        model_list = data["config"]["model_list"]
        if IS_PYTHON_BELOW_312:
            if not isinstance(model_list, list):
                # Work around https://github.com/BerriAI/litellm/issues/5664
                raise TypeError(f"model_list must be a list, not a {type(model_list)}.")
        else:
            # pylint: disable-next=possibly-used-before-assignment
            _DeploymentTypedDictValidator.validate_python(model_list)

        return data

    @model_validator(mode="after")
    def _populate_llm_config(self) -> "LiteLLMModel":
        if self.llm_config is None:
            self.llm_config = LLMConfig.from_legacy_dict(self.config)
        return self

    # SEE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    # > `none` means the model will not call any tool and instead generates a message.
    # > `auto` means the model can pick between generating a message or calling one or more tools.
    # > `required` means the model must call one or more tools.
    NO_TOOL_CHOICE: ClassVar[str] = "none"
    MODEL_CHOOSES_TOOL: ClassVar[str] = "auto"
    TOOL_CHOICE_REQUIRED: ClassVar[str] = "required"
    # None means we won't provide a tool_choice to the LLM API
    UNSPECIFIED_TOOL_CHOICE: ClassVar[None] = None

    async def check_request_limit(self, **kwargs) -> None:
        """Check if the request is within the request rate limit."""
        if "request_limit" in self.config:
            await GLOBAL_LIMITER.try_acquire(
                ("client|request", self.name),
                self.config["request_limit"].get(self.name, None),
                weight=1,
                **kwargs,
            )

    async def check_rate_limit(self, token_count: float, **kwargs) -> None:
        if "rate_limit" in self.config:
            await GLOBAL_LIMITER.try_acquire(
                ("client", self.name),
                self.config["rate_limit"].get(self.name, None),
                weight=max(int(token_count), 1),
                **kwargs,
            )

    async def _dispatch(
        self,
        spec: ModelSpec,
        *,
        messages: list[Message],
        streaming: bool = False,
        **chat_kwargs,
    ) -> list[LLMResult] | AsyncIterable[LLMResult]:
        """Dispatch one request to `spec`, choosing Chat vs Responses per `spec.responses_api`.

        Non-streaming paths return a list of `LLMResult`s. Streaming paths
        return an async iterator that has already produced its first chunk;
        errors before the first chunk (stream-open failure, an immediate
        refusal) surface as exceptions from this coroutine, while mid-stream
        errors propagate unmodified when the caller iterates the result.
        """
        if spec.responses_api:
            previous_response_id, messages = _extract_previous_response_id(messages)
            tools = chat_kwargs.pop("tools", None)
            if streaming:
                gen = await self._aresponses_iter(
                    messages, tools, previous_response_id, spec=spec, **chat_kwargs
                )
                return await _commit_stream(gen)
            return await self._aresponses(
                messages, tools, previous_response_id, spec=spec, **chat_kwargs
            )

        # Chat Completions path: `tools` stays in chat_kwargs.
        if streaming:
            gen = await self.acompletion_iter(messages, spec=spec, **chat_kwargs)
            return await _commit_stream(gen)
        return await self.acompletion(messages, spec=spec, **chat_kwargs)

    async def _run_with_fallbacks(
        self,
        attempt: Callable[..., Coroutine[Any, Any, Any]],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Drive a single-attempt coroutine across the `LLMConfig.models` chain.

        `attempt(spec, *args, **kwargs)` must run one try at the model described
        by `spec` and raise on failure. Retries within a single model use
        exponential jitter backoff (see `retry.backoff_seconds`). Exceptions
        classified by `should_fallback` (or retry exhaustion on a retryable
        exception) advance to the next spec; anything else propagates.
        """
        llm_config = cast("LLMConfig", self.llm_config)  # populated by after-validator
        last_exc: BaseException | None = None
        for spec in llm_config.models:
            for i in range(spec.max_retries + 1):
                try:
                    return await attempt(spec, *args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if should_retry(exc) and i < spec.max_retries:
                        await asyncio.sleep(backoff_seconds(i))
                        continue
                    if should_retry(exc) or should_fallback(exc):
                        break
                    raise
        raise AllModelsExhaustedError(last_exc) from last_exc

    # the order should be first request and then rate(token)
    @request_limited
    @rate_limited
    async def acompletion(  # noqa: C901
        self, messages: list[Message], *, spec: ModelSpec | None = None, **kwargs
    ) -> list[LLMResult]:
        if spec is None:
            spec = cast("LLMConfig", self.llm_config).models[0]
        tools = kwargs.get("tools")
        if not tools:
            # OpenAI, Anthropic and potentially other LLM providers
            # don't allow empty tool_calls lists, so remove empty
            kwargs.pop("tools", None)

        # cast is necessary for LiteLLM typing bug: https://github.com/BerriAI/litellm/issues/7641
        prompts = cast(
            "list[litellm.types.llms.openai.AllMessageValues]",
            [m.model_dump(by_alias=True) for m in messages],
        )
        tool_choice = kwargs.get("tool_choice")
        if self.tool_parser is not None and tool_choice not in {
            None,
            self.NO_TOOL_CHOICE,
        }:
            logger.warning(
                f"Custom tool parser was provided."
                f"Setting tool_choice parameter to {self.NO_TOOL_CHOICE}."
            )
            kwargs["tool_choice"] = self.NO_TOOL_CHOICE

        call_kwargs = {**spec.to_litellm_kwargs(), **kwargs, "messages": prompts}
        try:
            completions = await track_costs(litellm.acompletion)(**call_kwargs)
        except Exception:
            logger.exception("acompletion attempt failed on %s.", spec.name)
            raise

        finish_reason = (
            getattr(completions.choices[0], "finish_reason", None)
            if completions.choices
            else None
        )
        if completions.choices and finish_reason in REFUSAL_REASON:
            refusal = ModelRefusalError(
                f"Model {spec.name} refused with finish_reason={finish_reason!r}.",
                model=spec.name,
                finish_reason=finish_reason,
                response=completions,
            )
            logger.error("Model %s refused.", spec.name, exc_info=refusal)
            raise refusal

        used_model = completions.model
        results: list[LLMResult] = []

        # Use getattr because ModelResponse.usage not in LiteLLM's type hints
        # In practice, usage always exists in non-streaming responses
        usage = getattr(completions, "usage", None)
        prompt_count = usage.prompt_tokens if usage else None
        completion_count = usage.completion_tokens if usage else None
        cache_read, cache_creation = parse_cached_usage(usage)

        try:
            cost = completion_cost(completion_response=completions, model=used_model)
        except Exception as e:
            cost = 0.0
            logger.warning(f"Failed to calculate cost for {used_model}: {e}")

        # We are not streaming here, so we can cast to list[litellm.utils.Choices]
        for choice in completions.choices:
            try:
                if self.tool_parser is None:
                    output_messages: (
                        Message
                        | ToolRequestMessage
                        | list[Message]
                        | list[ToolRequestMessage]
                    ) = default_tool_parser(choice, tools)
                else:
                    sig = signature(self.tool_parser)
                    first_param = next(iter(sig.parameters.values()))
                    arg: str | litellm.utils.Choices = (
                        choice.message.content or ""
                        if first_param.annotation is str
                        else choice
                    )
                    output_messages = self.tool_parser(arg, tools)  # type: ignore[arg-type]
            except ValidationError as exc:
                raise MalformedMessageError(
                    f"Failed to convert model response's message {choice.message}"
                    f" Got finish reason {choice.finish_reason!r},"
                    f" full response was {completions},"
                    f" and tool choice was {kwargs.get('tool_choice')!r}."
                ) from exc

            if not isinstance(output_messages, list):
                output_messages = [output_messages]

            reasoning_content = None
            if hasattr(choice.message, "reasoning_content"):
                reasoning_content = choice.message.reasoning_content

            results.append(
                LLMResult(
                    model=used_model,
                    text=choice.message.content,
                    prompt=messages,
                    messages=output_messages,
                    logprob=sum_logprobs(choice),
                    top_logprobs=extract_top_logprobs(choice),
                    prompt_count=prompt_count,
                    completion_count=completion_count,
                    cache_read_tokens=cache_read,
                    cache_creation_tokens=cache_creation,
                    cost=cost,
                    system_fingerprint=completions.system_fingerprint,
                    reasoning_content=reasoning_content,
                    finish_reason=choice.finish_reason,
                )
            )
        await self._maybe_validate(results)
        return results

    async def _maybe_validate(self, results: list[LLMResult]) -> None:
        """Run `llm_config.response_validator` against each result, if attached.

        Any exception from the validator is wrapped as `ResponseValidationError`
        so the retry/fallback loop can re-attempt the same model (up to
        `ModelSpec.max_retries`) and then advance to the next.
        """
        validator = cast("LLMConfig", self.llm_config).response_validator
        if not validator:
            return
        for result in results:
            try:
                outcome = validator(result)
                if isawaitable(outcome):
                    await outcome
            except Exception as exc:
                raise ResponseValidationError(
                    f"response_validator {validator!r} rejected {result!r}"
                ) from exc

    # the order should be first request and then rate(token)
    @request_limited
    @rate_limited
    async def acompletion_iter(  # noqa: C901
        self, messages: list[Message], *, spec: ModelSpec | None = None, **kwargs
    ) -> AsyncIterable[LLMResult]:
        if spec is None:
            spec = cast("LLMConfig", self.llm_config).models[0]
        # cast is necessary for LiteLLM typing bug: https://github.com/BerriAI/litellm/issues/7641
        prompts = cast(
            "list[litellm.types.llms.openai.AllMessageValues]",
            [m.model_dump(by_alias=True) for m in messages if m.content],
        )
        stream_options = {
            "include_usage": True,
        }
        # NOTE: Specifically requesting reasoning for deepseek-r1 models
        if kwargs.get("include_reasoning"):
            stream_options["include_reasoning"] = True

        call_kwargs = {
            **spec.to_litellm_kwargs(),
            **kwargs,
            "messages": prompts,
            "stream": True,
            "stream_options": stream_options,
        }
        try:
            stream_completions = await track_costs_iter(litellm.acompletion)(
                **call_kwargs
            )
        except Exception:
            logger.exception("acompletion_iter failed to open stream on %s.", spec.name)
            raise
        start_clock = asyncio.get_running_loop().time()
        outputs = []
        logprobs = []
        role = None
        finish_reason: str | None = None
        reasoning_content = []
        used_model = None
        async for completion in stream_completions:
            if not used_model:
                used_model = completion.model
            choice = completion.choices[0]
            delta = choice.delta
            # logprobs can be None, or missing a content attribute,
            # or a ChoiceLogprobs object with a NoneType/empty content attribute
            if logprob_content := getattr(choice.logprobs, "content", None):
                logprobs.append(logprob_content[0].logprob or 0)
            outputs.append(delta.content or "")
            role = delta.role or role
            # The usage-only chunk (when include_usage=True) has finish_reason=None,
            # so retain the last non-None finish_reason value
            finish_reason = choice.finish_reason or finish_reason
            if hasattr(delta, "reasoning_content"):
                reasoning_content.append(delta.reasoning_content or "")
        text = "".join(outputs)

        # Calculate usage info first so we can pass it during construction
        cache_read, cache_creation, cost = None, None, 0.0
        prompt_count, completion_count = None, None
        if hasattr(completion, "usage"):
            prompt_count = completion.usage.prompt_tokens
            completion_count = completion.usage.completion_tokens
            cache_read, cache_creation = parse_cached_usage(completion.usage)
            try:
                cost = completion_cost(completion_response=completion, model=used_model)
            except Exception as e:
                logger.warning(f"Failed to calculate cost for {used_model}: {e}")

        result = LLMResult(
            model=used_model,
            text=text,
            prompt=messages,
            messages=[Message(role=role, content=text)],
            logprob=sum_logprobs(logprobs),
            top_logprobs=extract_top_logprobs(completion),
            reasoning_content="".join(reasoning_content),
            prompt_count=prompt_count,
            completion_count=completion_count,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_creation,
            cost=cost,
            finish_reason=finish_reason,
        )

        if text:
            result.seconds_to_first_token = (
                asyncio.get_running_loop().time() - start_clock
            )

        if finish_reason in REFUSAL_REASON:
            refusal = ModelRefusalError(
                f"Model {spec.name} refused with finish_reason={finish_reason!r}.",
                model=spec.name,
                finish_reason=finish_reason,
                response=result,
            )
            logger.error("Model %s refused.", spec.name, exc_info=refusal)
            raise refusal

        yield result

    @request_limited
    @rate_limited
    async def _aresponses(
        self,
        messages: list[Message],
        tools: list[dict] | None,
        previous_response_id: str | None = None,
        *,
        spec: ModelSpec | None = None,
        **kwargs,
    ) -> list[LLMResult]:
        """Call the Responses API (non-streaming)."""
        if spec is None:
            spec = cast("LLMConfig", self.llm_config).models[0]
        responses_input = _convert_to_responses_input(messages)
        responses_tools = _convert_tools_for_responses(tools)

        call_kwargs: dict[str, Any] = {
            **spec.to_litellm_kwargs(),
            "input": responses_input,
            "tools": responses_tools,
            "store": True,
            **kwargs,
        }
        if previous_response_id is not None:
            call_kwargs["previous_response_id"] = previous_response_id

        try:
            response = await track_costs(litellm.aresponses)(**call_kwargs)
        except Exception:
            logger.exception("aresponses attempt failed on %s.", spec.name)
            raise

        used_model = response.model
        usage = getattr(response, "usage", None)
        prompt_count = usage.input_tokens if usage else None
        completion_count = usage.output_tokens if usage else None

        try:
            cost = completion_cost(completion_response=response, model=used_model)
        except Exception as e:
            cost = 0.0
            logger.warning(f"Failed to calculate cost for {used_model}: {e}")

        text_content, output_messages = _parse_responses_output(response.output)

        for msg in output_messages:
            msg.info = {**(msg.info or {}), "response_id": response.id}

        return [
            LLMResult(
                model=used_model,
                text=text_content,
                prompt=messages,
                messages=output_messages
                or [Message(role="assistant", content=text_content)],
                prompt_count=prompt_count,
                completion_count=completion_count,
                cost=cost,
                response_id=response.id,
            )
        ]

    def _build_result_from_response(
        self,
        response: ResponsesAPIResponse,
        messages: list[Message],
    ) -> LLMResult:
        """Build an LLMResult from a completed Responses API response."""
        text, output_messages = _parse_responses_output(response.output)  # type: ignore[arg-type]
        usage = response.usage

        for msg in output_messages:
            msg.info = {**(msg.info or {}), "response_id": response.id}

        return LLMResult(
            model=response.model,
            text=text,
            messages=output_messages or [Message(role="assistant", content=text)],
            prompt=messages,
            prompt_count=usage.input_tokens if usage else None,
            completion_count=usage.output_tokens if usage else None,
            response_id=response.id,
        )

    @request_limited
    @rate_limited
    async def _aresponses_iter(  # noqa: C901
        self,
        messages: list[Message],
        tools: list[dict] | None,
        previous_response_id: str | None = None,
        *,
        spec: ModelSpec | None = None,
        **kwargs,
    ) -> AsyncIterable[LLMResult]:
        """Stream results from the Responses API."""
        if spec is None:
            spec = cast("LLMConfig", self.llm_config).models[0]
        responses_input = _convert_to_responses_input(messages)
        responses_tools = _convert_tools_for_responses(tools)

        call_kwargs: dict[str, Any] = {
            **spec.to_litellm_kwargs(),
            "input": responses_input,
            "tools": responses_tools,
            "store": True,
            "stream": True,
            **kwargs,
        }
        if previous_response_id is not None:
            call_kwargs["previous_response_id"] = previous_response_id

        try:
            stream = await track_costs_iter(litellm.aresponses)(**call_kwargs)
        except Exception:
            logger.exception("aresponses_iter failed to open stream on %s.", spec.name)
            raise

        completed_response: ResponsesAPIResponse | None = None
        incomplete_response: ResponsesAPIResponse | None = None
        async for event in stream:
            if isinstance(event, OutputTextDeltaEvent):
                yield LLMResult(model=spec.name, text=event.delta, prompt=messages)
            elif isinstance(event, ResponseCompletedEvent):
                completed_response = event.response
            elif isinstance(event, ResponseIncompleteEvent):
                incomplete_response = event.response
            elif isinstance(event, ResponseFailedEvent):
                error_dict = event.response.error
                error_msg = (
                    error_dict.get("message", "Unknown error")
                    if error_dict
                    else "Response failed"
                )
                raise RuntimeError(f"Responses API request failed: {error_msg}")  # noqa: TRY004
            elif isinstance(event, ErrorEvent):
                raise RuntimeError(  # noqa: TRY004
                    f"Responses API streaming error: {event.error.message}"
                )

        if completed_response:
            yield self._build_result_from_response(completed_response, messages)
        elif incomplete_response:
            logger.warning(
                f"Responses API returned incomplete response for model {spec.name}."
            )
            yield self._build_result_from_response(incomplete_response, messages)
        else:
            raise RuntimeError(
                "Responses API stream ended unexpectedly without a terminal event"
            )

    def count_tokens(self, text: str) -> int:
        # NOTE: by design text is just str here, as None leads to a ValueError
        return litellm.token_counter(model=self.name, text=text)

    @property
    def provider(self) -> str:
        return litellm.get_llm_provider(
            self.name, api_base=self.config.get("api_base")
        )[1]

    async def select_tool(
        self, *selection_args, **selection_kwargs
    ) -> ToolRequestMessage:
        """Shim to aviary.core.ToolSelector that supports tool schemae."""
        primary = cast("LLMConfig", self.llm_config).models[0]

        async def _acompletion(**kw: Any) -> Any:
            return await litellm.acompletion(**primary.to_litellm_kwargs(), **kw)

        tool_selector = ToolSelector(
            model_name=self.name, acompletion=track_costs(_acompletion)
        )
        return await tool_selector(*selection_args, **selection_kwargs)
