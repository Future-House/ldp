"""Independent job event callback system for LLM calls."""

__all__ = [
    "CostComponent",
    "ExecutionType", 
    "JobEventCallback",
    "create_job_event_callback",
    "set_execution_context",
    "execution_context",
    "get_execution_context",
    "configure_rest_client",
    "get_rest_client",
    "is_job_event_tracking_enabled",
]

import asyncio
import contextvars
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from .external import (
    CostComponent,
    ExecutionType,
    JobEventCreateRequest,
    RestClient,
)
from .types import LLMResult

logger = logging.getLogger(__name__)


def is_job_event_tracking_enabled() -> bool:
    """Check if job event tracking is enabled via environment variable.
    
    Returns:
        True if TRACK_JOB_EVENTS environment variable is set to 'true' (case-insensitive)
    """
    return os.environ.get("TRACK_JOB_EVENTS", "FALSE").lower() == "true"


# Context variables for execution context
_execution_id = contextvars.ContextVar[UUID | None]("execution_id", default=None)
_execution_type = contextvars.ContextVar[ExecutionType | None]("execution_type", default=None)
_crow = contextvars.ContextVar[str | None]("crow", default=None)

_GLOBAL_REST_CLIENT: RestClient | None = None


def configure_rest_client(**kwargs) -> None:
    """Configure the global REST client for job event reporting.
    
    Args:
        **kwargs: Arguments to pass to RestClient constructor
    """
    global _GLOBAL_REST_CLIENT
    
    if not is_job_event_tracking_enabled():
        logger.debug("Job event tracking is disabled (TRACK_JOB_EVENTS not set to 'true')")
        _GLOBAL_REST_CLIENT = None
        return
    
    try:
        _GLOBAL_REST_CLIENT = RestClient(**kwargs)
        logger.info("Job event tracking configured and enabled")
    except ValueError as e:
        # If API key is not available, log a debug message and don't configure the client
        logger.debug(f"REST client not configured: {e}")
        _GLOBAL_REST_CLIENT = None


def get_rest_client() -> RestClient | None:
    """Get the global REST client.
    
    Returns:
        The global REST client, or None if not configured
    """
    return _GLOBAL_REST_CLIENT


def _auto_configure_rest_client() -> None:
    """Auto-configure the REST client if job event tracking is enabled and FUTUREHOUSE_API_KEY is available."""
    global _GLOBAL_REST_CLIENT
    if _GLOBAL_REST_CLIENT is None and is_job_event_tracking_enabled():
        configure_rest_client()


class JobEventCallback:
    """Independent callback for job event reporting."""
    
    def __init__(
        self,
        execution_id: Optional[UUID] = None,
        execution_type: Optional[ExecutionType] = None,
        cost_component: CostComponent = CostComponent.LLM_USAGE,
        crow: Optional[str] = None,
    ):
        """Initialize the job event callback.
        
        Args:
            execution_id: ID of the current execution (trajectory or session)
            execution_type: Type of execution (TRAJECTORY or SESSION)
            cost_component: Type of cost component (LLM_USAGE, EXTERNAL_SERVICE, or STEP)
            crow: Foreign key to the crows db key name
        """
        self.execution_id = execution_id or _execution_id.get()
        self.execution_type = execution_type or _execution_type.get()
        self.cost_component = cost_component
        self.crow = crow or _crow.get()
    
    def __call__(self, result: LLMResult) -> None:
        """Handle LLM result callback for job event reporting.
        
        Args:
            result: The LLM result containing cost and usage information
        """
        if not is_job_event_tracking_enabled():
            return
            
        if not self.execution_id or not self.execution_type:
            return
            
        asyncio.create_task(self._async_report_job_event(result))
    
    async def _async_report_job_event(self, result: LLMResult) -> None:
        """Asynchronously report job event to external tracking system."""
        try:
            if not is_job_event_tracking_enabled():
                return
                
            client = get_rest_client()
            if not client:
                return  # No client configured
            
            # Parse the date from result to get start/end times
            # For streaming, we use the timing information from the result
            now = datetime.now()
            if hasattr(result, 'date') and result.date:
                try:
                    result_time = datetime.fromisoformat(result.date.replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    result_time = now
            else:
                result_time = now
                
            # Calculate start time based on timing information
            total_time = result.seconds_to_last_token or 0.0
            start_time = result_time
            if total_time > 0:
                start_time = result_time - timedelta(seconds=total_time)
            
            # Create job event request with all relevant fields
            request = JobEventCreateRequest(
                execution_id=self.execution_id,
                execution_type=self.execution_type,
                cost_component=self.cost_component,
                started_at=start_time,
                ended_at=result_time,
                crow=self.crow,
                amount_usd=result.cost if result.cost >= 0 else None,
                amount_acu=result.cost if result.cost >= 0 else None,
                rate=None,
                input_token_count=result.prompt_count if result.prompt_count > 0 else None,
                completion_token_count=result.completion_count if result.completion_count > 0 else None,
                metadata={
                    "result_id": str(result.id),
                    "session_id": str(result.session_id) if result.session_id else None,
                    "model": result.model,
                    "name": result.name,
                    "timestamp": result.date,
                    "seconds_to_first_token": result.seconds_to_first_token,
                    "seconds_to_last_token": result.seconds_to_last_token,
                    "logprob": result.logprob,
                    "top_logprobs": result.top_logprobs,
                    "reasoning_content": result.reasoning_content,
                    "text_length": len(result.text) if result.text else 0,
                    "config": result.config,
                },
            )
            
            # Report the job event
            await client.acreate_job_event(request)
            
        except Exception as e:
            logger.debug(f"Failed to report job event: {e}")


def create_job_event_callback(
    execution_id: Optional[UUID] = None,
    execution_type: Optional[ExecutionType] = None,
    cost_component: CostComponent = CostComponent.LLM_USAGE,
    crow: Optional[str] = None,
) -> JobEventCallback:
    """Create a job event callback.
    
    Args:
        execution_id: ID of the current execution
        execution_type: Type of execution
        cost_component: Type of cost component
        crow: Foreign key to the crows db key name
        
    Returns:
        Job event callback instance
    """
    return JobEventCallback(
        execution_id=execution_id,
        execution_type=execution_type,
        cost_component=cost_component,
        crow=crow,
    )


def set_execution_context(
    execution_id: UUID | None, 
    execution_type: ExecutionType | None,
    crow: str | None = None,
) -> None:
    """Set the execution context for job event reporting.
    
    Args:
        execution_id: ID of the current execution (trajectory or session)
        execution_type: Type of execution (TRAJECTORY or SESSION)
        crow: Foreign key to the crows db key name
    """
    _execution_id.set(execution_id)
    _execution_type.set(execution_type)
    _crow.set(crow)


@contextmanager
def execution_context(
    execution_id: UUID | None, 
    execution_type: ExecutionType | None,
    crow: str | None = None,
):
    """Context manager for setting execution context for job event reporting.
    
    Args:
        execution_id: ID of the current execution (trajectory or session)
        execution_type: Type of execution (TRAJECTORY or SESSION)
        crow: Foreign key to the crows db key name
    """
    prev_execution_id = _execution_id.get()
    prev_execution_type = _execution_type.get()
    prev_crow = _crow.get()
    
    set_execution_context(execution_id, execution_type, crow)
    try:
        yield
    finally:
        set_execution_context(prev_execution_id, prev_execution_type, prev_crow)


def get_execution_context() -> tuple[UUID | None, ExecutionType | None]:
    """Get the current execution context.
    
    Returns:
        Tuple of (execution_id, execution_type)
    """
    return _execution_id.get(), _execution_type.get()
