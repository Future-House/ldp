import asyncio
import contextvars
import datetime
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar
from uuid import UUID

import litellm

from lmi.external import FUTUREHOUSE_API_KEY, SHOULD_TRACK_COST

from .external import (
    CostComponent,
    ExecutionType,
    JobEventCreateRequest,
    RestClient,
)

logger = logging.getLogger(__name__)


def get_execution_id_from_env() -> UUID | None:
    """Get execution ID from environment variable.

    Returns:
        UUID if FUTUREHOUSE_EXECUTION_ID is set, None otherwise
    """
    execution_id_str = os.environ.get("FUTUREHOUSE_EXECUTION_ID")
    if execution_id_str:
        try:
            return UUID(execution_id_str)
        except ValueError:
            logger.warning(
                f"Invalid FUTUREHOUSE_EXECUTION_ID format: {execution_id_str}"
            )
    return None


def get_execution_type_from_env() -> ExecutionType | None:
    """Get execution type from environment variable.

    Returns:
        ExecutionType if FUTUREHOUSE_EXECUTION_TYPE is set, None otherwise
    """
    execution_type_str = os.environ.get("FUTUREHOUSE_EXECUTION_TYPE")
    if execution_type_str:
        try:
            return ExecutionType(execution_type_str.lower())
        except ValueError:
            logger.warning(f"Invalid FUTUREHOUSE_EXECUTION_TYPE: {execution_type_str}")
    return None


class CostTracker:
    def __init__(self):
        self.lifetime_cost_usd = 0.0
        self.last_report = 0.0
        # Not a contextvar because I can't imagine a scenario where you'd want more fine-grained control
        self.report_every_usd = 1.0
        self._job_event_client: RestClient | None = None
        self._enabled = contextvars.ContextVar[bool]("track_costs", default=False)

    @property
    def enabled(self) -> bool:
        """Check if cost tracking should be enabled.

        Priority:
        1. Explicit enable context variable (set by either cost_tracking_ctx or enable_cost_tracking)
        2. If context variable is not set or False, check Environment variable SHOULD_TRACK_COST
        """
        explicit_value = self._enabled.get()
        if explicit_value:
            return explicit_value
        return os.environ.get(SHOULD_TRACK_COST, "false").lower() == "true"

    @property
    def future_house_api_platform_key(self) -> str | None:
        return os.environ.get(FUTUREHOUSE_API_KEY, None)

    @property
    def rest_client(self) -> RestClient | None:
        """Get or create the REST client for job event reporting."""
        if self._job_event_client is None and self.future_house_api_platform_key:
            try:
                self._job_event_client = RestClient(
                    api_key=self.future_house_api_platform_key
                )
                logger.debug("REST client configured for job event reporting")
            except ValueError as e:
                logger.debug(f"Failed to configure REST client: {e}")
        return self._job_event_client

    def record(
        self,
        response: (
            litellm.ModelResponse
            | litellm.types.utils.EmbeddingResponse
            | litellm.types.utils.ModelResponseStream
        ),
    ) -> None:
        self.lifetime_cost_usd += litellm.cost_calculator.completion_cost(
            completion_response=response
        )

        if self.lifetime_cost_usd - self.last_report > self.report_every_usd:
            logger.info(f"Cumulative lmi API call cost: ${self.lifetime_cost_usd:.8f}")
            self.last_report = self.lifetime_cost_usd

        # Record for remote tracking if enabled
        if self.future_house_api_platform_key:
            self._record_remote(response)

    def _record_remote(
        self,
        response: (
            litellm.ModelResponse
            | litellm.types.utils.EmbeddingResponse
            | litellm.types.utils.ModelResponseStream
        ),
    ) -> None:
        """Record response for remote tracking (job event reporting).

        Args:
            response: The litellm response to record
        """
        # Get execution context from environment variables
        execution_id = get_execution_id_from_env()
        execution_type = get_execution_type_from_env()

        if execution_id and execution_type and self.rest_client:
            # need to store a reference to the task to avoid it being garbage collected
            task = asyncio.create_task(
                self._report_job_event(response, execution_id, execution_type)
            )
            task.add_done_callback(
                lambda _: logger.debug("Job event reported: %s", response)
            )

    async def _report_job_event(
        self,
        response: (
            litellm.ModelResponse
            | litellm.types.utils.EmbeddingResponse
            | litellm.types.utils.ModelResponseStream
        ),
        execution_id: UUID,
        execution_type: ExecutionType,
    ) -> None:
        """Asynchronously report job event to external tracking system."""
        try:
            if not self.rest_client:
                return

            # TODO: can avoid this call by taking the cost as a parameter
            cost = litellm.cost_calculator.completion_cost(completion_response=response)

            # Extract token counts
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = getattr(response.usage, "prompt_tokens", None) or 0
                completion_tokens = (
                    getattr(response.usage, "completion_tokens", None) or 0
                )

            ended_at = None
            model_name = "unknown"

            if hasattr(response, "created"):
                ended_at = response.created

            if (
                hasattr(response, "model")
                and response.model
                and isinstance(response.model, str)
            ):
                model_name = response.model

            request = JobEventCreateRequest(
                execution_id=execution_id,
                execution_type=execution_type,
                cost_component=CostComponent.LLM_USAGE,
                # TODO: update to use the actual start time from the response
                started_at=datetime.datetime.now(),
                ended_at=ended_at,
                crow=None,
                amount_usd=cost if cost >= 0 else None,
                amount_acu=cost if cost >= 0 else None,
                rate=None,
                input_token_count=prompt_tokens if prompt_tokens > 0 else None,
                completion_token_count=completion_tokens
                if completion_tokens > 0
                else None,
                metadata={
                    "model": model_name,
                    "response_type": type(response).__name__,
                    "model_config": getattr(response, "model_config", None),
                },
            )

            # Report the job event
            await self.rest_client.create_job_event(request)
            logger.debug(f"Job event reported for execution {execution_id}")

        except Exception as e:
            logger.debug(f"Failed to report job event: {e}")


GLOBAL_COST_TRACKER = CostTracker()


def set_reporting_threshold(threshold_usd: float) -> None:
    GLOBAL_COST_TRACKER.report_every_usd = threshold_usd


def enable_cost_tracking(enabled: bool = True) -> None:
    GLOBAL_COST_TRACKER._enabled.set(enabled)


@contextmanager
def cost_tracking_ctx(enabled: bool = True):
    prev = GLOBAL_COST_TRACKER._enabled.get()
    GLOBAL_COST_TRACKER._enabled.set(enabled)
    try:
        yield
    finally:
        GLOBAL_COST_TRACKER._enabled.set(prev)


def get_execution_context() -> tuple[UUID | None, ExecutionType | None]:
    """Get the current execution context from environment variables.

    Returns:
        Tuple of (execution_id, execution_type)
    """
    execution_id = get_execution_id_from_env()
    execution_type = get_execution_type_from_env()
    return execution_id, execution_type


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
        if GLOBAL_COST_TRACKER.enabled:
            GLOBAL_COST_TRACKER.record(response)
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

    def __next__(self):
        response = next(self.stream)
        if GLOBAL_COST_TRACKER.enabled:
            GLOBAL_COST_TRACKER.record(response)
        return response

    async def __anext__(self):
        response = await self.stream.__anext__()
        if GLOBAL_COST_TRACKER.enabled:
            GLOBAL_COST_TRACKER.record(response)
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
