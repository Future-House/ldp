"""This module contains commonly-used Op implementations."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Generic, TypeVar, cast, overload

import numpy as np
import tree
from aviary.core import Message, Tool, ToolRequestMessage, is_coroutine_callable
from pydantic import BaseModel

from ldp.llms import (
    EmbeddingModel,
    HybridEmbeddingModel,
    LiteEmbeddingModel,
    LLMModel,
    LLMResult,
    SparseEmbeddingModel,
)

from .gradient_estimators import assign_constant_grads
from .memory import Memory, MemoryModel, UIndexMemoryModel
from .op_utils import CallID, get_call_id, get_training_mode
from .ops import GradInType, Op, OpCtx, ResultOrValue, TOutput

logger = logging.getLogger(__name__)


def logsumexp(a: np.ndarray | list[float]) -> float:
    a_max = np.max(a)
    return a_max + np.log(np.sum(np.exp(a - a_max)))


class IdentityOp(Op[TOutput]):
    """
    An operation that simply returns the input value.

    NOTE: this op is equivalent to FxnOp(lambda x: x).
    """

    async def forward(self, value: TOutput) -> TOutput:
        # We assume value already has the correct run_id from its producer
        return value

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args: list[ResultOrValue],
        input_kwargs: dict[str, ResultOrValue],
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        return [], {"value": grad_output}


class StopGradOp(IdentityOp[TOutput]):
    """Pass through Op that terminates gradients in the backward pass."""

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args: list[ResultOrValue],
        input_kwargs: dict[str, ResultOrValue],
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        return assign_constant_grads(input_args, input_kwargs, None)


TConfig = TypeVar("TConfig", bound=BaseModel | dict)


class ConfigOp(Op[TConfig], Generic[TConfig]):
    """An operation that contains a configuration object."""

    def __init__(self, config: TConfig):
        self.config = config

    async def forward(self) -> TConfig:
        return self.config

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args: list[ResultOrValue],
        input_kwargs: dict[str, ResultOrValue],
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        # Check that the grad_output structure is consistent with our config
        tree.assert_same_structure(
            grad_output, ctx.get(call_id, "output").value, check_types=False
        )

        # Terminate here - we're a leaf since a ConfigOp takes no inputs
        return [], {}


TResult = TypeVar("TResult")


class Cacheable(Generic[TResult]):
    def __init__(self, co: Awaitable[TResult]) -> None:
        self.co = co
        self.done = False
        self.result: TResult | None = None
        self.lock = asyncio.Lock()

    async def get_result(self) -> TResult | None:
        async with self.lock:
            if not self.done:
                self.result = await self.co
                self.done = True
            return self.result

    def __await__(self):
        return self.get_result().__await__()


def async_cache(func):
    @functools.lru_cache(maxsize=1024)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return Cacheable(co=func(*args, **kwargs))

    return wrapper


class FxnOp(Op[TOutput]):
    """
    Wrap a function for a straight through gradient approximation for all args/kwargs.

    Basically, consider the fxn as a transform upon the inputs during the forward pass,
    and propagating the same gradient for all inputs during the backward pass.
    """

    def __init__(
        self,
        fxn: Callable[..., TOutput] | Callable[..., Awaitable[TOutput]],
        cache: bool = False,
        fxn_name: str | None = None,  # useful for lambdas
    ):
        if cache:
            self.fxn = (
                async_cache(fxn) if is_coroutine_callable(fxn) else lru_cache()(fxn)
            )
        else:
            self.fxn = fxn

        try:
            self.fxn_name = fxn_name or fxn.__name__
        except AttributeError:  # unittest.mock.Mock or lambda
            self.fxn_name = str(fxn)

        # override forward args with the signature of the function
        fwd_sig = inspect.signature(self.fxn)
        self._fwd_args = list(fwd_sig.parameters.values())

    def __repr__(self) -> str:
        return f"{type(self).__name__} {self.fxn_name} ({id(self)})"

    async def forward(self, *args, **kwargs) -> TOutput:
        if is_coroutine_callable(self.fxn):
            return await self.fxn(*args, **kwargs)
        return self.fxn(*args, **kwargs)

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args: list[ResultOrValue],
        input_kwargs: dict[str, ResultOrValue],
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        return assign_constant_grads(input_args, input_kwargs, 0.0)


class PromptOp(FxnOp[str]):
    """An operation that formats kwargs into a prompt string."""

    async def _fxn(
        self, prompt_kwargs: dict[str, str] | None = None, **kwargs: str
    ) -> str:
        return self.prompt.format(**{**(prompt_kwargs or {}), **kwargs})

    def __init__(self, prompt: str):
        self.prompt = prompt
        super().__init__(fxn=self._fxn, cache=False)

    def __repr__(self) -> str:
        # we want to use Op.__repr__, not FxnOp.__repr__
        return super(FxnOp, self).__repr__()


class LLMCallOp(Op[Message]):
    """An operation for LLM calls interaction."""

    def __init__(self, num_samples_logprob_estimate: int = 0) -> None:
        super().__init__()
        self.num_samples_partition_estimate = num_samples_logprob_estimate

    @overload
    async def forward(
        self,
        config: dict,
        msgs: list[Message],
        tools: list[Tool] = ...,
        tool_choice: Tool | str | None = LLMModel.TOOL_CHOICE_REQUIRED,
    ) -> ToolRequestMessage: ...

    @overload
    async def forward(
        self,
        config: dict,
        msgs: list[Message],
        tools: None = None,
        tool_choice: str | None = LLMModel.TOOL_CHOICE_REQUIRED,
    ) -> Message: ...

    async def forward(
        self,
        config: dict,
        msgs: list[Message],
        tools: list[Tool] | None = None,
        tool_choice: Tool | str | None = LLMModel.TOOL_CHOICE_REQUIRED,
    ) -> Message:
        """Calls the LLM.

        Args:
            config: Configuration passed to LLMModel.
            msgs: Input messages to prompt model with.
            tools: A list of Tools that the model may call, if supported.
            tool_choice: Configures how the model should choose a tool.
                Can be a Tool or a string; see here for string options:
                https://platform.openai.com/docs/guides/function-calling#configuring-function-calling-behavior-using-the-tool_choice-parameter
                NOTE: if `tools` is None or empty, this parameter is ignored.

        Returns:
            Output message from the model.
        """
        model = LLMModel(config=config)

        if not tools:
            # if no tools are provided, tool_choice must be 'none'
            tool_choice = "none"

        result = await model.call(messages=msgs, tools=tools, tool_choice=tool_choice)
        if result.messages is None:
            raise ValueError("No messages returned")

        # if not set, assume temp = 1. TODO: when would it not be set?
        temperature: float = (result.config or {}).get("temperature", 1.0)

        # Compute a Monte Carlo estimate of the logprob of this sequence at the given temperature.
        logprob = await self.compute_logprob(
            raw_log_p=result.logprob,
            temperature=temperature,
            model=model,
            messages=msgs,
            tools=tools,
            tool_choice=tool_choice,
        )

        call_id = get_call_id()
        self.ctx.update(call_id, "result", result)
        # This is the logprob of this sequence according to the raw model, without
        # any temperature/top-p distribution shaping.
        self.ctx.update(call_id, "raw_logprob", result.logprob)

        self.ctx.update(call_id, "temperature", temperature)
        self.ctx.update(call_id, "logprob", logprob)

        return result.messages[0]

    async def compute_logprob(
        self,
        raw_log_p: float | None,
        temperature: float,
        model: LLMModel,
        **model_kwargs,
    ) -> float | None:
        """This method computes a Monte Carlo estimate of logprob for a given temperature.

        It takes as input the logprob at T=1. The derivation is in Section 5.1 of the Aviary notes.
        """
        if temperature == 1:
            return raw_log_p

        if temperature == 0:
            return 1.0

        if raw_log_p is None or self.num_samples_partition_estimate == 0:
            return None

        # TODO: possibly move to MultipleCompletionLLMModel here, though we need to check that the estimates
        # are consistent - not sure we'd be sampling from the same distribution as N independent samples.
        # TODO: think about whether sampling params besides temperature need to be accounted for, like top_p
        results = await asyncio.gather(*[
            model.call(temperature=1, **model_kwargs)
            for _ in range(self.num_samples_partition_estimate)
        ])
        temp_factor = 1.0 / temperature - 1.0

        # Partition function estimate:
        # Z_T = E_P[ e^(lnP/T - lnP) ]
        log_Z_T = logsumexp([
            temp_factor * cast(float, result.logprob) for result in results
        ]) - np.log(self.num_samples_partition_estimate)

        return (raw_log_p / temperature) - log_Z_T

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args: list[ResultOrValue],
        input_kwargs: dict[str, ResultOrValue],
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        # By default, we want to descend into config, but not msgs/tools/tool_choice
        # Essentially: we can think of each config field as an independent parameter,
        # but not necessarily each message or tool.

        # tree.map_structure allows us to assign a gradient of 0 to all fields of config
        grad_config = tree.map_structure(lambda _: 0.0, input_kwargs["config"])
        grad_kwargs = {"config": grad_config}
        for arg in ("msgs", "tools", "tool_choice"):
            if arg in input_kwargs:
                grad_kwargs[arg] = 0.0

        return [], grad_kwargs

    def get_examples(self) -> list[tuple[LLMResult, float]]:
        examples = [
            (
                self.ctx.get(c, "result", None),
                # get 'model' kwarg from grad_input
                # use default of None if not found
                self.ctx.get(c, "grad_input", default=([], {}))[1].get("model", None),
            )
            for c in self.get_call_ids()
        ]
        # filter out the None values
        return [(e, w) for e, w in examples if e is not None]


class MemoryOp(Op[list[Memory]]):
    """An operation for managing memory retrieval and storage."""

    def __init__(self, memory_model: MemoryModel | None = None):
        super().__init__()
        self.memory_model = memory_model or UIndexMemoryModel(
            embedding_model=EmbeddingModel.from_name("sparse")
        )

    async def forward(
        self,
        query: str,
        input: str | None = None,  # noqa: A002
        matches: int = 3,
    ) -> list[Memory]:
        """Retrieve relevant memories based on a query."""
        if get_training_mode():
            call_id = get_call_id()
            self.ctx.update(call_id, "query", query)
            self.ctx.update(call_id, "memory_input", input)
        return await self.memory_model.get_memory(query, matches)

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args: list[ResultOrValue],
        input_kwargs: dict[str, ResultOrValue],
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        """Backward pass for memory retrieval - goes back to item."""
        return assign_constant_grads(input_args, input_kwargs, 0.0)


class EmbeddingOp(Op):
    """A general operation for embedding text using LiteLLM."""

    def __init__(
        self,
        *,
        dense_embedding: str = "text-embedding-3-small",
        dense_embedding_dim: int = 512,
        sparse_embedding_dim: int = 0,
        **embedding_model_kwargs,
    ):
        if "timeout" not in embedding_model_kwargs:
            embedding_model_kwargs.setdefault("timeout", 60)
        emb_models: list[EmbeddingModel] = []
        if dense_embedding_dim > 0:
            emb_models.append(
                LiteEmbeddingModel(
                    name=dense_embedding,
                    dimensions=dense_embedding_dim,
                    embed_kwargs=embedding_model_kwargs,
                )
            )
        if sparse_embedding_dim > 0:
            emb_models.append(SparseEmbeddingModel(dimensions=sparse_embedding_dim))
        self.embedding = HybridEmbeddingModel(models=emb_models)

    async def forward(self, string_input: str) -> np.ndarray:
        return await self.embedding.embed_text(string_input)

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args,
        input_kwargs,
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        return [], {"string_input": None}
