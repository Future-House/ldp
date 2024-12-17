import asyncio
import contextlib
import functools
import json
import logging
from abc import ABC
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
)
from inspect import isasyncgenfunction, signature
from typing import (
    Any,
    ClassVar,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

import litellm
from aviary.core import (
    Message,
    Tool,
    ToolRequestMessage,
    ToolsAdapter,
    ToolSelector,
    is_coroutine_callable,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from lcaw.constants import (
    CHARACTERS_PER_TOKEN_ASSUMPTION,
    DEFAULT_VERTEX_SAFETY_SETTINGS,
    EXTRA_TOKENS_FROM_USER_ROLE,
    IS_PYTHON_BELOW_312,
)
from lcaw.exceptions import JSONSchemaValidationError
from lcaw.prompts import default_system_prompt
from lcaw.rate_limiter import GLOBAL_LIMITER
from lcaw.types import Chunk, LLMResult
from lcaw.utils import get_litellm_retrying_config

logger = logging.getLogger(__name__)

if not IS_PYTHON_BELOW_312:
    _DeploymentTypedDictValidator = TypeAdapter(
        list[litellm.DeploymentTypedDict],
        config=ConfigDict(arbitrary_types_allowed=True),
    )

# Yes, this is a hack, it mostly matches
# https://github.com/python-jsonschema/referencing/blob/v0.35.1/referencing/jsonschema.py#L20-L21
JSONSchema: TypeAlias = Mapping[str, Any]


def sum_logprobs(choice: litellm.utils.Choices) -> float | None:
    """Calculate the sum of the log probabilities of an LLM completion (a Choices object).

    Args:
        choice: A sequence of choices from the completion.

    Returns:
        The sum of the log probabilities of the choice.
    """
    try:
        logprob_obj = choice.logprobs
    except AttributeError:
        return None
    if isinstance(logprob_obj, dict):
        if logprob_obj.get("content"):
            return sum(logprob_info["logprob"] for logprob_info in logprob_obj["content"])
    elif choice.logprobs.content:
        return sum(logprob_info.logprob for logprob_info in choice.logprobs.content)
    return None


def validate_json_completion(completion: litellm.ModelResponse, output_type: type[BaseModel] | JSONSchema) -> None:
    """Validate a completion against a JSON schema.

    Args:
        completion: The completion to validate.
        output_type: A JSON schema or a Pydantic model to validate the completion.
    """
    try:
        for choice in completion.choices:
            if not hasattr(choice, "message") or not choice.message.content:
                continue
            # make sure it is a JSON completion, even if None
            # We do want to modify the underlying message
            # so that users of it can just parse it as expected
            choice.message.content = choice.message.content.split("```json")[-1].split("```")[0] or ""
            if isinstance(output_type, Mapping):  # JSON schema
                litellm.litellm_core_utils.json_validation_rule.validate_schema(
                    schema=dict(output_type), response=choice.message.content
                )
            else:
                output_type.model_validate_json(choice.message.content)
    except ValidationError as err:
        raise JSONSchemaValidationError("The completion does not match the specified schema.") from err


def prepare_args(func: Callable, chunk: str, name: str | None) -> tuple[tuple, dict]:
    with contextlib.suppress(TypeError):
        if "name" in signature(func).parameters:
            return (chunk,), {"name": name}
    return (chunk,), {}


async def do_callbacks(
    async_callbacks: Iterable[Callable[..., Awaitable]],
    sync_callbacks: Iterable[Callable[..., Any]],
    chunk: str,
    name: str | None,
) -> None:
    for f in async_callbacks:
        args, kwargs = prepare_args(f, chunk, name)
        await f(*args, **kwargs)
    for f in sync_callbacks:
        args, kwargs = prepare_args(f, chunk, name)
        f(*args, **kwargs)


class LLMModel(ABC, BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    llm_type: str | None = None
    name: str
    llm_result_callback: Callable[[LLMResult], None] | Callable[[LLMResult], Awaitable[None]] | None = Field(
        default=None,
        description=(
            "An async callback that will be executed on each"
            " LLMResult (different than callbacks that execute on each chunk)"
        ),
        exclude=True,
    )
    config: dict = Field(default_factory=dict)

    async def acomplete(self, prompt: str) -> Chunk:
        """Return the completion as string and the number of tokens in the prompt and completion."""
        raise NotImplementedError

    async def acomplete_iter(self, prompt: str) -> AsyncIterable[Chunk]:
        """Return an async generator that yields chunks of the completion.

        Only the last tuple will be non-zero.
        """
        raise NotImplementedError
        if False:  # type: ignore[unreachable]  # pylint: disable=using-constant-test
            yield  # Trick mypy: https://github.com/python/mypy/issues/5070#issuecomment-1050834495

    async def achat(self, messages: list[Message]) -> Chunk:
        """Return the completion as string and the number of tokens in the prompt and completion."""
        raise NotImplementedError

    async def achat_iter(self, messages: list[Message]) -> AsyncIterable[Chunk]:
        """Return an async generator that yields chunks of the completion.

        Only the last tuple will be non-zero.
        """
        raise NotImplementedError
        if False:  # type: ignore[unreachable]  # pylint: disable=using-constant-test
            yield  # Trick mypy: https://github.com/python/mypy/issues/5070#issuecomment-1050834495

    def infer_llm_type(self) -> str:
        return "completion"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # gross approximation

    async def call(
        self,
        messages: list[Message],
        callbacks: list[Callable] | None = None,
        name: str | None = None,
        system_prompt: str | None = None,
    ) -> LLMResult:
        """Call the LLM model with the given messages and configuration.

        Args:
            messages: A list of messages to send to the language model.
            callbacks: A list of callback functions to execute
                after receiving the response.
            name: Optional name for the result.
            system_prompt: System prompt to use, or None/empty string to not use one.
                This should be passed in the messages if using a chat model
            **chat_kwargs: Additional keyword arguments to pass to the chat function.

        Returns:
            LLMResult object containing the result of the call.
        """
        if self.llm_type is None:
            self.llm_type = self.infer_llm_type()

        if self.llm_type == "chat":
            return await self._run_chat(
                messages=messages,
                callbacks=callbacks,
                name=name,
            )

        if self.llm_type == "completion":
            # Build a static prompt from the messages ignoring roles
            prompt = "\n".join(m.content for m in messages if m.content)
            return await self._run_completion(
                prompt=prompt,
                data={},
                callbacks=callbacks,
                name=name,
                system_prompt=system_prompt,
            )

        raise ValueError(f"Unknown llm_type {self.llm_type!r}.")

    async def run_prompt(
        self,
        prompt: str,
        data: dict,
        callbacks: list[Callable] | None = None,
        name: str | None = None,
        system_prompt: str | None = default_system_prompt,
    ) -> LLMResult:
        messages = None  # using prompt, not messages
        if self.llm_type is None:
            self.llm_type = self.infer_llm_type()

        if self.llm_type == "chat":
            human_message_prompt = {"role": "user", "content": prompt}
            messages = [
                Message(role=m["role"], content=m["content"].format(**data))
                for m in (
                    [{"role": "system", "content": system_prompt}, human_message_prompt]
                    if system_prompt
                    else [human_message_prompt]
                )
            ]
            return await self._run_chat(messages, callbacks, name, system_prompt)
        if self.llm_type == "completion":
            return await self._run_completion(prompt, data, callbacks, name, system_prompt)
        raise ValueError(f"Unknown llm_type {self.llm_type!r}.")

    async def _run_chat(
        self,
        messages: list[Message],
        callbacks: list[Callable] | None = None,
        name: str | None = None,
        system_prompt: str | None = default_system_prompt,
    ) -> LLMResult:
        """Run a chat prompt.

        Args:
            messages: List of messages to use.
            callbacks: Optional functions to call with each chunk of the completion.
            name: Optional name for the result.
            system_prompt: System prompt to use, or None/empty string to not use one.

        Returns:
            Result of the chat.
        """
        result = LLMResult(
            model=self.name,
            name=name,
            prompt=messages,
            prompt_count=(
                sum(self.count_tokens(m.content) for m in messages if m.content is not None)
                + sum(self.count_tokens(m.role) for m in messages)
            ),
        )

        start_clock = asyncio.get_running_loop().time()
        if callbacks is None:
            chunk = await self.achat(messages)
            output = chunk.text
        else:
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]
            completion = await self.achat_iter(messages)  # type: ignore[misc]
            text_result = []
            async for chunk in completion:
                if chunk.text:
                    if result.seconds_to_first_token == 0:
                        result.seconds_to_first_token = asyncio.get_running_loop().time() - start_clock
                    text_result.append(chunk.text)
                    await do_callbacks(async_callbacks, sync_callbacks, chunk.text, name)
            output = "".join(text_result)
        usage = chunk.prompt_tokens, chunk.completion_tokens
        if sum(usage) > 0:
            result.prompt_count, result.completion_count = usage
        elif output:
            result.completion_count = self.count_tokens(output)
        result.text = output or ""
        result.seconds_to_last_token = asyncio.get_running_loop().time() - start_clock
        if self.llm_result_callback:
            if is_coroutine_callable(self.llm_result_callback):
                await self.llm_result_callback(result)  # type: ignore[misc]
            else:
                self.llm_result_callback(result)
        return result

    async def _run_completion(
        self,
        prompt: str,
        data: dict,
        callbacks: Iterable[Callable] | None = None,
        name: str | None = None,
        system_prompt: str | None = default_system_prompt,
    ) -> LLMResult:
        """Run a completion prompt.

        Args:
            prompt: Prompt to use.
            data: Keys for the input variables that will be formatted into prompt.
            callbacks: Optional functions to call with each chunk of the completion.
            name: Optional name for the result.
            system_prompt: System prompt to use, or None/empty string to not use one.

        Returns:
            Result of the completion.
        """
        formatted_prompt: str = (system_prompt + "\n\n" + prompt if system_prompt else prompt).format(**data)
        result = LLMResult(
            model=self.name,
            name=name,
            prompt=formatted_prompt,
            prompt_count=self.count_tokens(formatted_prompt),
        )

        start_clock = asyncio.get_running_loop().time()
        if callbacks is None:
            chunk = await self.acomplete(formatted_prompt)
            output = chunk.text
        else:
            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]

            completion = self.acomplete_iter(formatted_prompt)
            text_result = []
            async for chunk in completion:
                if chunk.text:
                    if result.seconds_to_first_token == 0:
                        result.seconds_to_first_token = asyncio.get_running_loop().time() - start_clock
                    text_result.append(chunk.text)
                    await do_callbacks(async_callbacks, sync_callbacks, chunk.text, name)
            output = "".join(text_result)
        usage = chunk.prompt_tokens, chunk.completion_tokens
        if sum(usage) > 0:
            result.prompt_count, result.completion_count = usage
        elif output:
            result.completion_count = self.count_tokens(output)
        result.text = output or ""
        result.seconds_to_last_token = asyncio.get_running_loop().time() - start_clock
        if self.llm_result_callback:
            if is_coroutine_callable(self.llm_result_callback):
                await self.llm_result_callback(result)  # type: ignore[misc]
            else:
                self.llm_result_callback(result)
        return result


LLMModelOrChild = TypeVar("LLMModelOrChild", bound=LLMModel)


def rate_limited(
    func: Callable[[LLMModelOrChild, Any], Awaitable[Chunk] | AsyncIterable[Chunk]],
) -> Callable[
    [LLMModelOrChild, Any, Any],
    Awaitable[Chunk | AsyncIterator[Chunk] | AsyncIterator[LLMModelOrChild]],
]:
    """Decorator to rate limit relevant methods of an LLMModel."""

    @functools.wraps(func)
    async def wrapper(
        self: LLMModelOrChild, *args: Any, **kwargs: Any
    ) -> Chunk | AsyncIterator[Chunk] | AsyncIterator[LLMModelOrChild]:
        if not hasattr(self, "check_rate_limit"):
            raise NotImplementedError(f"Model {self.name} must have a `check_rate_limit` method.")

        # Estimate token count based on input
        if func.__name__ in {"acomplete", "acomplete_iter"}:
            prompt = args[0] if args else kwargs.get("prompt", "")
            token_count = len(prompt) / CHARACTERS_PER_TOKEN_ASSUMPTION + EXTRA_TOKENS_FROM_USER_ROLE
        elif func.__name__ in {"achat", "achat_iter"}:
            messages = args[0] if args else kwargs.get("messages", [])
            token_count = len(str(messages)) / CHARACTERS_PER_TOKEN_ASSUMPTION
        else:
            token_count = 0  # Default if method is unknown

        await self.check_rate_limit(token_count)

        # If wrapping a generator, count the tokens for each
        # portion before yielding
        if isasyncgenfunction(func):

            async def rate_limited_generator() -> AsyncGenerator[LLMModelOrChild, None]:
                async for item in func(self, *args, **kwargs):
                    token_count = 0
                    if isinstance(item, Chunk):
                        token_count = int(len(item.text or "") / CHARACTERS_PER_TOKEN_ASSUMPTION)
                    await self.check_rate_limit(token_count)
                    yield item

            return rate_limited_generator()

        result = await func(self, *args, **kwargs)  # type: ignore[misc]

        if func.__name__ in {"acomplete", "achat"} and isinstance(result, Chunk):
            await self.check_rate_limit(result.completion_tokens)
        return result

    return wrapper


class PassThroughRouter(litellm.Router):  # TODO: add rate_limited
    """Router that is just a wrapper on LiteLLM's normal free functions."""

    def __init__(self, **kwargs):
        self._default_kwargs = kwargs

    async def atext_completion(self, *args, **kwargs):
        return await litellm.atext_completion(*args, **(self._default_kwargs | kwargs))

    async def acompletion(self, *args, **kwargs):
        return await litellm.acompletion(*args, **(self._default_kwargs | kwargs))


class LiteLLMModel(LLMModel):
    """A wrapper around the litellm library."""

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
            " limits.RateLimitItem string for parsing."
        ),
    )
    name: str = "gpt-4o-mini"
    _router: litellm.Router | None = None

    @model_validator(mode="before")
    @classmethod
    def maybe_set_config_attribute(cls, data: dict[str, Any]) -> dict[str, Any]:
        """If a user only gives a name, make a sensible config dict for them."""
        if "config" not in data:
            data["config"] = {}
        if "name" in data and "model_list" not in data["config"]:
            data["config"] = {
                "model_list": [
                    {
                        "model_name": data["name"],
                        "litellm_params": {"model": data["name"]}
                        | ({} if "gemini" not in data["name"] else {"safety_settings": DEFAULT_VERTEX_SAFETY_SETTINGS}),
                    }
                ],
            } | data["config"]

        if "router_kwargs" not in data["config"]:
            data["config"]["router_kwargs"] = {}
        data["config"]["router_kwargs"] = get_litellm_retrying_config() | data["config"]["router_kwargs"]
        if not data["config"].get("pass_through_router"):
            data["config"]["router_kwargs"] = {"retry_after": 5} | data["config"]["router_kwargs"]

        # we only support one "model name" for now, here we validate
        model_list = data["config"]["model_list"]
        if IS_PYTHON_BELOW_312:
            if not isinstance(model_list, list):
                # Work around https://github.com/BerriAI/litellm/issues/5664
                raise TypeError(f"model_list must be a list, not a {type(model_list)}.")
        else:
            # pylint: disable-next=possibly-used-before-assignment
            _DeploymentTypedDictValidator.validate_python(model_list)
        if len({m["model_name"] for m in model_list}) > 1:
            raise ValueError("Only one model name per model list is supported for now.")
        return data

    def __getstate__(self):
        # Prevent _router from being pickled, SEE: https://stackoverflow.com/a/2345953
        state = super().__getstate__()
        state["__dict__"] = state["__dict__"].copy()
        state["__dict__"].pop("_router", None)
        return state

    @property
    def router(self) -> litellm.Router:
        if self._router is None:
            router_kwargs: dict = self.config.get("router_kwargs", {})
            if self.config.get("pass_through_router"):
                self._router = PassThroughRouter(**router_kwargs)
            else:
                self._router = litellm.Router(model_list=self.config["model_list"], **router_kwargs)
        return self._router

    async def check_rate_limit(self, token_count: float, **kwargs) -> None:
        if "rate_limit" in self.config:
            await GLOBAL_LIMITER.try_acquire(
                ("client", self.name),
                self.config["rate_limit"].get(self.name, None),
                weight=max(int(token_count), 1),
                **kwargs,
            )

    @rate_limited
    async def acomplete(self, prompt: str) -> Chunk:  # type: ignore[override]
        response = await self.router.atext_completion(model=self.name, prompt=prompt)
        return Chunk(
            text=response.choices[0].text,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    @rate_limited
    async def acomplete_iter(  # type: ignore[override]
        self, prompt: str
    ) -> AsyncIterable[Chunk]:
        completion = await self.router.atext_completion(
            model=self.name,
            prompt=prompt,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in completion:
            yield Chunk(text=chunk.choices[0].text, prompt_tokens=0, completion_tokens=0)
        if hasattr(chunk, "usage") and hasattr(chunk.usage, "prompt_tokens"):
            yield Chunk(text=chunk.choices[0].text, prompt_tokens=0, completion_tokens=0)

    @rate_limited
    async def achat(self, messages: list[Message]) -> Chunk:  # type: ignore[override]
        prompts = [m.model_dump(by_alias=True) for m in messages if m.content]
        response = await self.router.acompletion(self.name, prompts)
        return Chunk(
            text=cast(litellm.Choices, response.choices[0]).message.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    @rate_limited
    async def achat_iter(  # type: ignore[override]
        self, messages: list[Message]
    ) -> AsyncIterable[Chunk]:
        prompts = [m.model_dump(by_alias=True) for m in messages if m.content]
        completion = await self.router.acompletion(
            self.name,
            prompts,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in completion:
            yield Chunk(
                text=chunk.choices[0].delta.content,
                prompt_tokens=0,
                completion_tokens=0,
            )
        if hasattr(chunk, "usage") and hasattr(chunk.usage, "prompt_tokens"):
            yield Chunk(
                text=None,
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens,
            )

    def infer_llm_type(self) -> str:
        if all("text-completion" in m.get("litellm_params", {}).get("model", "") for m in self.config["model_list"]):
            return "completion"
        return "chat"

    def count_tokens(self, text: str) -> int:
        return litellm.token_counter(model=self.name, text=text)

    async def select_tool(self, *selection_args, **selection_kwargs) -> ToolRequestMessage:
        """Shim to aviary.core.ToolSelector that supports tool schemae."""
        tool_selector = ToolSelector(model_name=self.name, acompletion=self.router.acompletion)
        return await tool_selector(*selection_args, **selection_kwargs)


class MultipleCompletionLLMModel(BaseModel):
    """Run n completions at once, all starting from the same messages."""

    model_config = ConfigDict(extra="forbid")

    # this should keep the original model
    # if fine-tuned, this should still refer to the base model
    name: str = "unknown"
    config: dict = Field(
        default_factory=lambda: {
            "model": "gpt-4o-mini",  # TODO: create a field validator
            "temperature": 0.1,
        },
        description=(
            "Configuration of the model:"
            "model is the name of the llm model to use,"
            "temperature is the sampling temperature, and"
            "n is the number of completions to generate by default."
        ),
    )
    encoding: Any | None = None

    def __str__(self) -> str:
        return f"{type(self).__name__} {self.name}"

    @model_validator(mode="after")
    def set_model_name(self) -> Self:
        if (self.config.get("model") in {"gpt-3.5-turbo", None} and self.name != "unknown") or (
            self.name != "unknown" and "model" not in self.config
        ):
            self.config["model"] = self.name
        elif "model" in self.config and self.name == "unknown":
            self.name = self.config["model"]
        # note we do not consider case where both are set
        # because that could be true if the model is fine-tuned
        return self

    async def achat(self, messages: Iterable[Message], **kwargs) -> litellm.ModelResponse:
        return await litellm.acompletion(
            messages=[m.model_dump(by_alias=True) for m in messages],
            **(self.config | kwargs),
        )

    async def achat_iter(self, messages: Iterable[Message], **kwargs) -> AsyncGenerator:
        return cast(
            AsyncGenerator,
            await litellm.acompletion(
                messages=[m.model_dump(by_alias=True) for m in messages],
                stream=True,
                stream_options={
                    "include_usage": True,  # Included to get prompt token counts
                },
                **(self.config | kwargs),
            ),
        )

    # SEE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    # > `none` means the model will not call any tool and instead generates a message.
    # > `auto` means the model can pick between generating a message or calling one or more tools.
    # > `required` means the model must call one or more tools.
    NO_TOOL_CHOICE: ClassVar[str] = "none"
    MODEL_CHOOSES_TOOL: ClassVar[str] = "auto"
    TOOL_CHOICE_REQUIRED: ClassVar[str] = "required"
    # None means we won't provide a tool_choice to the LLM API
    UNSPECIFIED_TOOL_CHOICE: ClassVar[None] = None

    async def call(  # noqa: C901, PLR0915
        self,
        messages: list[Message],
        callbacks: list[Callable] | None = None,
        output_type: type[BaseModel] | JSONSchema | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Tool | str | None = TOOL_CHOICE_REQUIRED,
        **chat_kwargs,
    ) -> list[LLMResult]:
        """
        Call the LLM model with the given messages and configuration.

        Args:
            messages: A list of messages to send to the language model.
            callbacks: A list of callback functions to execute after receiving the response.
            output_type: The type of the output model.
            tools: A list of tools to use during the call.
            tool_choice: The tool or tool identifier to use.
            **chat_kwargs: Additional keyword arguments to pass to the chat function.

        Returns:
            A list of LLMResult objects containing the results of the call.

        Raises:
            ValueError: If the number of completions (n) is invalid.
        """
        # add static configuration to kQwargs
        chat_kwargs = self.config | chat_kwargs

        start_clock = asyncio.get_running_loop().time()

        # Deal with tools. Note OpenAI throws a 400 response if tools is empty:
        # > Invalid 'tools': empty array. Expected an array with minimum length 1,
        # > but got an empty array instead.
        # So, circumvent this behavior if tools in (None, [])
        if tools:
            chat_kwargs["tools"] = ToolsAdapter.dump_python(tools, exclude_none=True, by_alias=True)
            if tool_choice is not None:
                chat_kwargs["tool_choice"] = (
                    {
                        "type": "function",
                        "function": {"name": tool_choice.info.name},
                    }
                    if isinstance(tool_choice, Tool)
                    else tool_choice
                )

        # deal with specifying output type
        if isinstance(output_type, Mapping):  # Use structured outputs
            model_name: str = chat_kwargs.get("model", "")
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
            schema = json.dumps(output_type.model_json_schema(mode="serialization"))
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

        n = chat_kwargs.get("n", 1)  # number of completions
        if n < 1:
            raise ValueError("Number of completions (n) must be >= 1.")

        prompt = [
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

        if callbacks is None:
            completion: litellm.ModelResponse = await self.achat(prompt, **chat_kwargs)
            if output_type is not None:
                validate_json_completion(completion, output_type)

            for choice in completion.choices:
                if isinstance(choice, litellm.utils.StreamingChoices):
                    raise NotImplementedError("Streaming is not yet supported.")

                if (
                    tools is not None  # Allows for empty tools list
                    or choice.finish_reason == "tool_calls"
                    or (getattr(choice.message, "tool_calls", None) is not None)
                ):
                    serialized_choice_message = choice.message.model_dump()
                    serialized_choice_message["tool_calls"] = serialized_choice_message.get("tool_calls") or []
                    output_messages: list[Message | ToolRequestMessage] = [
                        ToolRequestMessage(**serialized_choice_message)
                    ]
                else:
                    output_messages = [Message(**choice.message.model_dump())]

                results.append(
                    LLMResult(
                        model=self.name,
                        config=chat_kwargs,
                        prompt=prompt,
                        messages=output_messages,
                        logprob=sum_logprobs(choice),
                        system_fingerprint=completion.system_fingerprint,
                        # Note that these counts are aggregated over all choices
                        completion_count=completion.usage.completion_tokens,  # type: ignore[attr-defined,unused-ignore]
                        prompt_count=completion.usage.prompt_tokens,  # type: ignore[attr-defined,unused-ignore]
                    )
                )
        else:
            if tools:
                raise NotImplementedError("Using tools with callbacks is not supported")
            if n > 1:
                raise NotImplementedError("Multiple completions with callbacks is not supported")
            result = LLMResult(model=self.name, config=chat_kwargs, prompt=prompt)

            sync_callbacks = [f for f in callbacks if not is_coroutine_callable(f)]
            async_callbacks = [f for f in callbacks if is_coroutine_callable(f)]
            stream_completion = await self.achat_iter(messages, **chat_kwargs)
            text_result = []
            role = "assistant"

            async for chunk in stream_completion:
                delta = chunk.choices[0].delta
                role = delta.role or role
                if delta.content:
                    s = delta.content
                    if result.seconds_to_first_token == 0:
                        result.seconds_to_first_token = asyncio.get_running_loop().time() - start_clock
                    text_result.append(s)
                    [await f(s) for f in async_callbacks]
                    [f(s) for f in sync_callbacks]
                if hasattr(chunk, "usage"):
                    result.prompt_count = chunk.usage.prompt_tokens

            output = "".join(text_result)
            result.completion_count = litellm.token_counter(
                model=self.name,
                text=output,
            )
            # TODO: figure out how tools stream, and log probs
            result.messages = [Message(role=role, content=output)]
            results.append(result)

        if not results:
            # This happens in unit tests. We should probably not keep this block around
            # long-term. Previously, we would emit an empty ToolRequestMessage if
            # completion.choices were empty, so  I am replicating that here.
            results.append(
                LLMResult(
                    model=self.name,
                    config=chat_kwargs,
                    prompt=prompt,
                    messages=[ToolRequestMessage(tool_calls=[])],
                )
            )

        end_clock = asyncio.get_running_loop().time()

        for result in results:
            # Manually update prompt count if not set, which can
            # happen if the target model doesn't support 'include_usage'
            if not result.prompt_count and result.messages:
                result.prompt_count = litellm.token_counter(
                    model=self.name,
                    messages=[m.model_dump() for m in result.messages],
                )

            # update with server-side counts
            result.seconds_to_last_token = end_clock - start_clock

        return results

    async def call_single(
        self,
        messages: list[Message],
        callbacks: list[Callable] | None = None,
        output_type: type[BaseModel] | None = None,
        tools: list[Tool] | None = None,
        tool_choice: Tool | str | None = TOOL_CHOICE_REQUIRED,
        **chat_kwargs,
    ) -> LLMResult:
        return (await self.call(messages, callbacks, output_type, tools, tool_choice, n=1, **chat_kwargs))[0]
