import contextvars
import logging
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

import litellm

from lmi.types import LLMResponse

logger = logging.getLogger(__name__)


class CostTracker:
    def __init__(self):
        self.lifetime_cost_usd = 0.0
        self.last_report = 0.0
        # A contextvar so that different coroutines don't affect each other's cost tracking
        self.enabled = contextvars.ContextVar[bool]("track_costs", default=False)
        # Not a contextvar because I can't imagine a scenario where you'd want more fine-grained control
        self.report_every_usd = 1.0
        self._callbacks: list[Callable[[LLMResponse], Awaitable]] = []

    def add_callback(self, callback: Callable[[LLMResponse], Awaitable]) -> None:
        self._callbacks.append(callback)

    async def record(self, response: LLMResponse) -> None:
        # Only record on responses with usage information (final chunk in streaming)
        # We check for usage presence rather than cost > 0 because:
        # - Free models, unknown models, or custom pricing can have cost = 0
        # - We still want to fire callbacks for these to maintain visibility
        if not getattr(response, "usage", None):
            return

        self.lifetime_cost_usd += litellm.cost_calculator.completion_cost(
            completion_response=response
        )

        if self.lifetime_cost_usd - self.last_report > self.report_every_usd:
            logger.info(f"Cumulative lmi API call cost: ${self.lifetime_cost_usd:.8f}")
            self.last_report = self.lifetime_cost_usd

        for callback in self._callbacks:
            try:
                await callback(response)
            except Exception as e:
                logger.warning(
                    f"Callback failed during cost tracking: {e}", exc_info=True
                )


GLOBAL_COST_TRACKER = CostTracker()


def set_reporting_threshold(threshold_usd: float) -> None:
    GLOBAL_COST_TRACKER.report_every_usd = threshold_usd


def enable_cost_tracking(enabled: bool = True) -> None:
    GLOBAL_COST_TRACKER.enabled.set(enabled)


@contextmanager
def cost_tracking_ctx(enabled: bool = True):
    prev = GLOBAL_COST_TRACKER.enabled.get()
    GLOBAL_COST_TRACKER.enabled.set(enabled)
    try:
        yield
    finally:
        GLOBAL_COST_TRACKER.enabled.set(prev)


TReturn = TypeVar(
    "TReturn",
    bound=Awaitable[litellm.ModelResponse]
    | Awaitable[litellm.types.utils.EmbeddingResponse],
)
TParams = ParamSpec("TParams")


def track_costs(
    func: Callable[TParams, TReturn],
) -> Callable[TParams, TReturn]:
    """Automatically track API costs of a coroutine call.

    Note that the costs will only be recorded if `enable_cost_tracking()` is called,
    or if in a `cost_tracking_ctx()` context.

    Usage:
    ```
    @track_costs
    async def api_call(...) -> litellm.ModelResponse:
        ...
    ```

    Args:
        func: A coroutine that returns a ModelResponse or EmbeddingResponse

    Returns:
        A wrapped coroutine with the same signature.
    """

    async def wrapped_func(*args, **kwargs):
        response = await func(*args, **kwargs)
        if GLOBAL_COST_TRACKER.enabled.get():
            await GLOBAL_COST_TRACKER.record(response)
        return response

    return wrapped_func


class TrackedStreamWrapper:
    """Class that tracks costs as one iterates through the stream.

    Note that the following is not possible:
    ```
    async def wrap(func):
        resp: CustomStreamWrapper = await func()
        async for response in resp:
            yield response


    # This is ok
    async for resp in await litellm.acompletion(stream=True):
        print(resp)


    # This is not, because we cannot await an AsyncGenerator
    async for resp in await wrap(litellm.acompletion)(stream=True):
        print(resp)
    ```

    In order for `track_costs_iter` to not change how users call functions,
    we introduce this class to wrap the stream.
    """

    def __init__(self, stream: litellm.CustomStreamWrapper):
        self.stream = stream

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        response = await self.stream.__anext__()
        if GLOBAL_COST_TRACKER.enabled.get():
            await GLOBAL_COST_TRACKER.record(response)
        return response


def track_costs_iter(
    func: Callable[TParams, Awaitable[litellm.CustomStreamWrapper]],
) -> Callable[TParams, Awaitable[TrackedStreamWrapper]]:
    """Automatically track API costs of a streaming coroutine.

    The return type is changed to `TrackedStreamWrapper`, which can be iterated
    through in the same way. The underlying litellm object is available at
    `TrackedStreamWrapper.stream`.

    Note that the costs will only be recorded if `enable_cost_tracking()` is called,
    or if in a `cost_tracking_ctx()` context.

    Usage:
    ```
    @track_costs_iter
    async def streaming_api_call(...) -> litellm.CustomStreamWrapper:
        ...
    ```

    Args:
        func: A coroutine that returns CustomStreamWrapper.

    Returns:
        A wrapped coroutine with the same arguments but with a
        return type of TrackedStreamWrapper.
    """

    @wraps(func)
    async def wrapped_func(*args, **kwargs):
        return TrackedStreamWrapper(await func(*args, **kwargs))

    return wrapped_func
