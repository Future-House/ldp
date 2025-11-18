__all__ = [
    "CommonLLMNames",
    "LLMModel",
    "LiteLLMModel",
    "PassThroughRouter",
    "extract_top_logprobs",
    "parse_cached_usage",
    "rate_limited",
    "request_limited",
    "sum_logprobs",
    "validate_json_completion",
]

import asyncio
import contextlib
import copy
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
    Tool,
    ToolRequestMessage,
    ToolResponseMessage,
    ToolsAdapter,
    ToolSelector,
    is_coroutine_callable,
)
from aviary.message import MalformedMessageError
from litellm import completion_cost
from litellm.types.utils import Usage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from lmi.constants import (
    CHARACTERS_PER_TOKEN_ASSUMPTION,
    DEFAULT_VERTEX_SAFETY_SETTINGS,
    IS_PYTHON_BELOW_312,
)
from lmi.cost_tracker import track_costs, track_costs_iter
from lmi.exceptions import JSONSchemaValidationError
from lmi.rate_limiter import GLOBAL_LIMITER
from lmi.types import LLMResult
from lmi.utils import get_litellm_retrying_config

logger = logging.getLogger(__name__)

# List of possible refusal flags in finish_reason
REFUSAL_REASON = (
    "refusal",  # Anthropic safety filter
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
    CLAUDE_37_SONNET = "claude-3-7-sonnet-20250219"
    CLAUDE_45_SONNET = "claude-sonnet-4-5-20250929"

    # Use these when trying to think of a somewhat opinionated default
    OPENAI_BASELINE = "gpt-4o-2024-11-20"  # Fast and decent

    # Use these in unit testing
    OPENAI_TEST = "gpt-4o-mini-2024-07-18"  # Cheap, fast, and not OpenAI's cutting edge
    ANTHROPIC_TEST = (  # Cheap, fast, and not Anthropic's cutting edge
        "claude-3-5-haiku-20241022"
    )


class OverrideRouterConfig(BaseModel):
    model_list: list[dict[str, Any]]
    router_kwargs: dict[str, Any]
    fallbacks: list[dict[str, Any]]


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
            choice.message.content = (
                choice.message.content.split("```json")[-1].split("```")[0] or ""
            )
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

    async def acompletion(self, messages: list[Message], **kwargs) -> list[LLMResult]:
        """Return the completion as string and the number of tokens in the prompt and completion."""
        raise NotImplementedError

    async def acompletion_iter(
        self, messages: list[Message], **kwargs
    ) -> AsyncIterable[LLMResult]:
        """Return an async generator that yields completions.

        Only the last tuple will be non-zero.
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
        chat_kwargs = copy.deepcopy(kwargs)
        # if using the config for an LLMModel,
        # there may be a nested 'config' key
        # that can't be used by chat
        chat_kwargs.pop("config", None)
        n = chat_kwargs.get("n") or self.config.get("n", 1)
        if n < 1:
            raise ValueError("Number of completions (n) must be >= 1.")
        if "fallbacks" not in chat_kwargs and "fallbacks" in self.config:
            chat_kwargs["fallbacks"] = self.config.get("fallbacks", [])

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
            model_name: str = chat_kwargs.get("model") or self.name
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
        results: list[LLMResult] = []

        start_clock = asyncio.get_running_loop().time()
        if callbacks is None:
            results = await self.acompletion(messages, **chat_kwargs)
        else:
            if tools:
                raise NotImplementedError("Using tools with callbacks is not supported")
            n = chat_kwargs.get("n") or self.config.get("n", 1)
            if n > 1:
                raise NotImplementedError(
                    "Multiple completions with callbacks is not supported"
                )
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]
            stream_results = await self.acompletion_iter(messages, **chat_kwargs)
            text_result = []
            async for result in stream_results:
                if result.text:
                    if result.seconds_to_first_token == 0:
                        result.seconds_to_first_token = (
                            asyncio.get_running_loop().time() - start_clock
                        )
                    text_result.append(result.text)
                    await do_callbacks(
                        async_callbacks, sync_callbacks, result.text, name
                    )
                results.append(result)

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
        if len(results) != 1:
            # Can be caused by issues like https://github.com/BerriAI/litellm/issues/12298
            raise ValueError(f"Got {len(results)} results when expecting just one.")
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


class LiteLLMModel(LLMModel):
    """A wrapper around the litellm library."""

    model_config = ConfigDict(extra="forbid")

    name: str = CommonLLMNames.GPT_4O.value
    config: dict = Field(
        default_factory=dict,
        description=(
            "Configuration of this model containing several important keys. The"
            " optional `model_list` key stores a list of all model configurations"
            " (SEE: https://docs.litellm.ai/docs/routing). The optional"
            " `router_kwargs` key is keyword arguments to pass to the Router class."
            " Inclusion of a key `pass_through_router` with a truthy value will lead"
            " to using not using LiteLLM's Router, instead just LiteLLM's free"
            f" functions (see {PassThroughRouter.__name__}). Rate limiting applies"
            " regardless of `pass_through_router` being present. The optional"
            " `rate_limit` key is a dictionary keyed by model group name with values"
            " of type limits.RateLimitItem (in tokens / minute) or valid"
            " limits.RateLimitItem string for parsing. The optional `request_limit`"
            " key is a dictionary keyed by model group name with values representing"
            " the maximum number of requests per minute."
        ),
    )
    _router: litellm.Router | None = None

    @model_validator(mode="before")
    @classmethod
    def maybe_set_config_attribute(cls, input_data: dict[str, Any]) -> dict[str, Any]:  # noqa: C901
        """
        Set the config attribute if it is not provided.

        If name is not provided, uses the default name.
        If a user only gives a name, make a sensible config dict for them.
        """
        data = copy.deepcopy(input_data)

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

        if "router_kwargs" not in data["config"]:
            data["config"]["router_kwargs"] = {}
        data["config"]["router_kwargs"] = (
            get_litellm_retrying_config() | data["config"]["router_kwargs"]
        )
        if not data["config"].get("pass_through_router"):
            data["config"]["router_kwargs"] = {"retry_after": 5} | data["config"][
                "router_kwargs"
            ]

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

    # SEE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    # > `none` means the model will not call any tool and instead generates a message.
    # > `auto` means the model can pick between generating a message or calling one or more tools.
    # > `required` means the model must call one or more tools.
    NO_TOOL_CHOICE: ClassVar[str] = "none"
    MODEL_CHOOSES_TOOL: ClassVar[str] = "auto"
    TOOL_CHOICE_REQUIRED: ClassVar[str] = "required"
    # None means we won't provide a tool_choice to the LLM API
    UNSPECIFIED_TOOL_CHOICE: ClassVar[None] = None

    def __getstate__(self):
        # Prevent _router from being pickled, SEE: https://stackoverflow.com/a/2345953
        state = super().__getstate__()
        state["__dict__"] = state["__dict__"].copy()
        state["__dict__"].pop("_router", None)
        return state

    def get_router(
        self, override_config: OverrideRouterConfig | None = None
    ) -> litellm.Router:
        """Get the router, optionally with an override configuration.

        Args:
            override_config: Optional configuration to override the default router settings.
                Should contain 'model_list' and 'router_kwargs' keys.

        Returns:
            A litellm.Router instance.
        """
        if override_config:
            override_model_list = override_config.model_list
            override_router_kwargs = override_config.router_kwargs

            can_override = override_model_list and override_router_kwargs
            if not can_override:
                logger.warning(
                    "Cannot override router with provided config. Will use default config."
                )
            else:
                return litellm.Router(
                    model_list=override_model_list, **override_router_kwargs
                )

        if self._router is None:
            router_kwargs: dict = self.config.get("router_kwargs", {})
            if self.config.get("pass_through_router"):
                self._router = PassThroughRouter(**router_kwargs)
            else:
                self._router = litellm.Router(
                    model_list=self.config["model_list"], **router_kwargs
                )
        return self._router

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

    async def _handle_refusal_via_fallback(
        self, messages: list[Message], kwargs: dict[str, Any]
    ) -> list[LLMResult]:
        # Let's remove the current model from the configuration
        current_model = self.name
        override_config = OverrideRouterConfig(
            model_list=kwargs.pop("model_list", self.config["model_list"]),
            router_kwargs=kwargs.pop("router_kwargs", self.config["router_kwargs"]),
            fallbacks=kwargs.pop("fallbacks", self.config["fallbacks"]),
        )

        new_model_list = [
            model
            for model in override_config.model_list
            if model.get("model_name") != current_model
        ]

        new_fallbacks = []
        for fallback in override_config.fallbacks:
            if current_model not in fallback:
                new_fallbacks.append(fallback)
            else:
                target_model = fallback[current_model][0]
                new_fallbacks.append({target_model: fallback[current_model][1:]})

        # Now we update the configuration
        # Here I am resetting all the configuration I think the user needs to set
        # But it is possible there is a user misconfiguration that I'd be fixing here
        # We use model name to make the request. Will reset it after the request
        self.name = new_model_list[0].get("model_name") or ""
        if "fallbacks" in kwargs:
            kwargs["fallbacks"] = new_fallbacks
        kwargs["override_config"] = {
            "model_list": new_model_list,
            "router_kwargs": override_config.router_kwargs
            | {
                "fallbacks": new_fallbacks,
            },
            "fallbacks": new_fallbacks,
        }

        results = await self.acompletion(messages, **kwargs)
        # Put the object back to the original state
        self.name = current_model
        return results

    # the order should be first request and then rate(token)
    @request_limited
    @rate_limited
    async def acompletion(  # noqa: C901
        self, messages: list[Message], **kwargs
    ) -> list[LLMResult]:
        override_config = kwargs.pop("override_config", None)
        if override_config:
            override_config = OverrideRouterConfig(**override_config)
        router = self.get_router(override_config)
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
        completions = await track_costs(router.acompletion)(
            self.name, prompts, **kwargs
        )

        finish_reason = (
            getattr(completions.choices[0], "finish_reason", None)
            if completions.choices
            else None
        )
        if completions.choices and finish_reason in REFUSAL_REASON:
            logger.warning(
                f"The LLM request was refused with finish reason '{finish_reason}' "
                f"for model {self.name}. "
                "Attempting to fallback to next model in the list."
            )
            if override_config is not None:
                # Case we had a previous refusal
                kwargs["model_list"] = override_config.model_list
                kwargs["router_kwargs"] = override_config.router_kwargs
                kwargs["fallbacks"] = override_config.fallbacks

            model_list = kwargs.get("model_list") or self.config.get("model_list", None)
            if model_list and len(model_list) > 1:
                return await self._handle_refusal_via_fallback(messages, kwargs)
            logger.warning(
                f"No fallback models available after refusal for model {self.name}. "
                "Will return return a LLMResult with the refusal completion."
            )

        used_model = completions.model or self.name
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
        choices = cast("list[litellm.utils.Choices]", completions.choices)
        for choice in choices:
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
            try:
                output_messages = [msg_type(**serialized_message)]
            except ValidationError as exc:
                raise MalformedMessageError(
                    f"Failed to convert model response's message {choice.message}"
                    f" into a {msg_type.__name__}."
                    f" Got finish reason {choice.finish_reason!r},"
                    f" full response was {completions},"
                    f" and tool choice was {kwargs.get('tool_choice')!r}."
                ) from exc

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
                )
            )
        return results

    # the order should be first request and then rate(token)
    @request_limited
    @rate_limited
    async def acompletion_iter(
        self, messages: list[Message], **kwargs
    ) -> AsyncIterable[LLMResult]:
        override_config = kwargs.pop("override_config", None)
        if override_config:
            override_config = OverrideRouterConfig(**override_config)
        router = self.get_router(override_config)
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

        stream_completions = await track_costs_iter(router.acompletion)(
            self.name,
            prompts,
            stream=True,
            stream_options=stream_options,
            **kwargs,
        )
        start_clock = asyncio.get_running_loop().time()
        outputs = []
        logprobs = []
        role = None
        reasoning_content = []
        used_model = None
        async for completion in stream_completions:
            if not used_model:
                used_model = completion.model or self.name
            choice = completion.choices[0]
            delta = choice.delta
            # logprobs can be None, or missing a content attribute,
            # or a ChoiceLogprobs object with a NoneType/empty content attribute
            if logprob_content := getattr(choice.logprobs, "content", None):
                logprobs.append(logprob_content[0].logprob or 0)
            outputs.append(delta.content or "")
            role = delta.role or role
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
        )

        if text:
            result.seconds_to_first_token = (
                asyncio.get_running_loop().time() - start_clock
            )

        yield result

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
        tool_selector = ToolSelector(
            model_name=self.name, acompletion=track_costs(self.get_router().acompletion)
        )
        return await tool_selector(*selection_args, **selection_kwargs)
