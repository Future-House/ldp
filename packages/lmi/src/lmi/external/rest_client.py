"""REST client for FutureHouse platform API operations."""

import copy
import logging
import os
from enum import StrEnum
from typing import Any, ClassVar, cast
from uuid import UUID

from httpx import AsyncClient, Client, HTTPStatusError, codes
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from .job_event_models import (
    JobEventCreateRequest,
    JobEventCreateResponse,
    JobEventUpdateRequest,
)

logger = logging.getLogger(__name__)


class Stage(StrEnum):
    """Available deployment stages."""
    DEV = "https://dev-api.platform.futurehouse.org"
    PROD = "https://api.platform.futurehouse.org"


class RestClientError(Exception):
    """Base exception for REST client errors."""


class JobEventClientError(RestClientError):
    """Raised when there's an error with job event operations."""


class JobEventCreationError(JobEventClientError):
    """Raised when there's an error creating a job event."""


class JobEventUpdateError(JobEventClientError):
    """Raised when there's an error updating a job event."""


class RestClient:
    """REST client for FutureHouse platform API operations."""
    
    REQUEST_TIMEOUT: ClassVar[float] = 30.0  # sec - for general API calls
    MAX_RETRY_ATTEMPTS: ClassVar[int] = 3
    RETRY_MULTIPLIER: ClassVar[int] = 1
    MAX_RETRY_WAIT: ClassVar[int] = 10

    def __init__(
        self,
        stage: Stage = Stage.PROD,
        service_uri: str | None = None,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        verbose_logging: bool = False,
    ):
        """Initialize the REST client.
        
        Args:
            stage: Deployment stage (DEV or PROD)
            service_uri: Custom service URI (overrides stage)
            api_key: API key for authentication. If not provided, will look for FUTUREHOUSE_API_KEY environment variable
            headers: Additional headers to include in requests
            verbose_logging: Enable verbose logging
        """
        if verbose_logging:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        self.base_url = service_uri or stage.value
        self.stage = stage
        self.api_key = api_key or os.environ.get("FUTUREHOUSE_API_KEY")
        
        if self.api_key is None:
            raise ValueError(
                "API key must be provided either as parameter or via FUTUREHOUSE_API_KEY environment variable"
            )
        
        self._clients: dict[str, Client | AsyncClient] = {}
        self.headers = headers or {}

    @property
    def client(self) -> Client:
        """Authenticated HTTP client for regular API calls."""
        return cast(Client, self.get_client("application/json", authenticated=True))

    @property
    def async_client(self) -> AsyncClient:
        """Authenticated async HTTP client for regular API calls."""
        return cast(
            AsyncClient,
            self.get_client("application/json", authenticated=True, async_client=True),
        )

    def get_client(
        self,
        content_type: str | None = "application/json",
        authenticated: bool = True,
        async_client: bool = False,
        timeout: float | None = None,
    ) -> Client | AsyncClient:
        """Return a cached HTTP client or create one if needed.

        Args:
            content_type: The desired content type header.
            authenticated: Whether the client should include authentication.
            async_client: Whether to use an async client.
            timeout: Custom timeout in seconds. Uses REQUEST_TIMEOUT if not provided.

        Returns:
            An HTTP client configured with the appropriate headers.
        """
        client_timeout = timeout or self.REQUEST_TIMEOUT
        key = f"{content_type or 'none'}_{authenticated}_{async_client}_{client_timeout}"

        if key not in self._clients:
            headers = copy.deepcopy(self.headers)

            if authenticated and self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            if content_type:
                headers["Content-Type"] = content_type

            headers["x-client"] = "lmi-sdk"

            self._clients[key] = (
                AsyncClient(
                    base_url=self.base_url,
                    headers=headers,
                    timeout=client_timeout,
                )
                if async_client
                else Client(
                    base_url=self.base_url,
                    headers=headers,
                    timeout=client_timeout,
                )
            )

        return self._clients[key]

    @retry(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=MAX_RETRY_WAIT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def create_job_event(self, request: JobEventCreateRequest) -> JobEventCreateResponse:
        """Create a new job event.
        
        Args:
            request: Job event creation request
            
        Returns:
            Job event creation response
            
        Raises:
            JobEventCreationError: If the API call fails
        """
        try:
            response = self.client.post(
                "/v0.1/job-events",
                json=request.model_dump(exclude_none=True, mode="json"),
            )
            response.raise_for_status()
            return JobEventCreateResponse(**response.json())
        except HTTPStatusError as e:
            if e.response.status_code == codes.BAD_REQUEST:
                raise JobEventCreationError(
                    f"Invalid job event creation request: {e.response.text}."
                ) from e
            if e.response.status_code == codes.NOT_FOUND:
                raise JobEventCreationError(
                    f"Execution not found for job event creation: {e.response.text}."
                ) from e
            raise JobEventCreationError(
                f"Error creating job event: {e.response.status_code} - {e.response.text}."
            ) from e
        except Exception as e:
            raise JobEventCreationError(
                f"An unexpected error occurred during job event creation: {e!r}."
            ) from e

    @retry(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=MAX_RETRY_WAIT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def acreate_job_event(self, request: JobEventCreateRequest) -> JobEventCreateResponse:
        """Asynchronously create a new job event.
        
        Args:
            request: Job event creation request
            
        Returns:
            Job event creation response
            
        Raises:
            JobEventCreationError: If the API call fails
        """
        try:
            response = await self.async_client.post(
                "/v0.1/job-events",
                json=request.model_dump(exclude_none=True, mode="json"),
            )
            response.raise_for_status()
            return JobEventCreateResponse(**response.json())
        except HTTPStatusError as e:
            if e.response.status_code == codes.BAD_REQUEST:
                raise JobEventCreationError(
                    f"Invalid job event creation request: {e.response.text}."
                ) from e
            if e.response.status_code == codes.NOT_FOUND:
                raise JobEventCreationError(
                    f"Execution not found for job event creation: {e.response.text}."
                ) from e
            raise JobEventCreationError(
                f"Error creating job event: {e.response.status_code} - {e.response.text}."
            ) from e
        except Exception as e:
            raise JobEventCreationError(
                f"An unexpected error occurred during job event creation: {e!r}."
            ) from e

    @retry(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=MAX_RETRY_WAIT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def update_job_event(
        self, job_event_id: UUID, request: JobEventUpdateRequest
    ) -> None:
        """Update an existing job event.
        
        Args:
            job_event_id: ID of the job event to update
            request: Job event update request
            
        Returns:
            Job event update response
            
        Raises:
            JobEventUpdateError: If the API call fails
        """
        try:
            response = self.client.patch(
                f"/v0.1/job-events/{job_event_id}",
                json=request.model_dump(exclude_none=True, mode="json"),
            )
            response.raise_for_status()
        except HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                raise JobEventUpdateError(
                    f"Job event with ID {job_event_id} not found."
                ) from e
            if e.response.status_code == codes.BAD_REQUEST:
                raise JobEventUpdateError(
                    f"Invalid job event update request: {e.response.text}."
                ) from e
            raise JobEventUpdateError(
                f"Error updating job event: {e.response.status_code} - {e.response.text}."
            ) from e
        except Exception as e:
            raise JobEventUpdateError(
                f"An unexpected error occurred during job event update: {e!r}."
            ) from e

    @retry(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_MULTIPLIER, max=MAX_RETRY_WAIT),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def aupdate_job_event(
        self, job_event_id: UUID, request: JobEventUpdateRequest
    ) -> None:
        """Asynchronously update an existing job event.
        
        Args:
            job_event_id: ID of the job event to update
            request: Job event update request
            
        Returns:
            Job event update response
            
        Raises:
            JobEventUpdateError: If the API call fails
        """
        try:
            response = await self.async_client.patch(
                f"/v0.1/job-events/{job_event_id}",
                json=request.model_dump(exclude_none=True, mode="json"),
            )
            response.raise_for_status()
        except HTTPStatusError as e:
            if e.response.status_code == codes.NOT_FOUND:
                raise JobEventUpdateError(
                    f"Job event with ID {job_event_id} not found."
                ) from e
            if e.response.status_code == codes.BAD_REQUEST:
                raise JobEventUpdateError(
                    f"Invalid job event update request: {e.response.text}."
                ) from e
            raise JobEventUpdateError(
                f"Error updating job event: {e.response.status_code} - {e.response.text}."
            ) from e
        except Exception as e:
            raise JobEventUpdateError(
                f"An unexpected error occurred during job event update: {e!r}."
            ) from e
