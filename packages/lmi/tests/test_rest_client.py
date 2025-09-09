"""Tests for REST client and job event functionality."""

import os
from datetime import datetime
from uuid import uuid4

import pytest

from lmi.external import (
    CostComponent,
    ExecutionType,
    JobEventCreateRequest,
    RestClient,
    Stage,
)
from lmi.job_event_callback import is_job_event_tracking_enabled

class TestJobEventModels:
    """Test job event model validation."""

    def test_job_event_create_request_validation(self):
        """Test that JobEventCreateRequest validates correctly."""
        request = JobEventCreateRequest(
            execution_id=uuid4(),
            execution_type=ExecutionType.TRAJECTORY,
            cost_component=CostComponent.LLM_USAGE,
            started_at=datetime.now(),
            ended_at=datetime.now(),
        )
        assert request.execution_type == ExecutionType.TRAJECTORY
        assert request.cost_component == CostComponent.LLM_USAGE

    def test_job_event_create_request_with_optional_fields(self):
        """Test JobEventCreateRequest with optional fields."""
        request = JobEventCreateRequest(
            execution_id=uuid4(),
            execution_type=ExecutionType.SESSION,
            cost_component=CostComponent.EXTERNAL_SERVICE,
            started_at=datetime.now(),
            ended_at=datetime.now(),
            crow="test-crow",
            amount_usd=0.05,
            rate=0.001,
            input_token_count=50,
            completion_token_count=25,
            metadata={"model": "gpt-4", "temperature": 0.7},
        )
        assert request.crow == "test-crow"
        assert request.amount_usd == 0.05
        assert request.metadata["model"] == "gpt-4"

    def test_execution_type_enum(self):
        """Test ExecutionType enum values."""
        assert ExecutionType.TRAJECTORY == "trajectory"
        assert ExecutionType.SESSION == "session"

    def test_cost_component_enum(self):
        """Test CostComponent enum values."""
        assert CostComponent.LLM_USAGE == "llm_usage"
        assert CostComponent.EXTERNAL_SERVICE == "external_service"
        assert CostComponent.STEP == "step"


class TestRestClient:
    """Test REST client functionality."""

    def test_client_initialization(self):
        """Test that RestClient initializes correctly."""
        client = RestClient(
            api_key="test-key",
            service_uri="https://api.example.com",
        )
        assert client.base_url == "https://api.example.com"
        assert client.api_key == "test-key"

    def test_client_initialization_without_api_key(self):
        """Test client initialization without API key raises ValueError."""
        # Ensure no API key in environment
        original_key = os.environ.get("FUTUREHOUSE_API_KEY")
        if "FUTUREHOUSE_API_KEY" in os.environ:
            del os.environ["FUTUREHOUSE_API_KEY"]
        
        try:
            with pytest.raises(ValueError, match="API key must be provided"):
                RestClient(service_uri="https://api.example.com")
        finally:
            # Restore original key if it existed
            if original_key is not None:
                os.environ["FUTUREHOUSE_API_KEY"] = original_key

    def test_client_initialization_with_custom_headers(self):
        """Test client initialization with custom headers."""
        custom_headers = {"X-Custom-Header": "custom-value"}
        client = RestClient(
            api_key="test-key",
            service_uri="https://api.example.com",
            headers=custom_headers,
        )
        assert client.headers["X-Custom-Header"] == "custom-value"

    def test_stage_usage(self):
        """Test that Stage enum works correctly."""
        client = RestClient(
            api_key="test-key",
            stage=Stage.DEV,
        )
        assert client.base_url == Stage.DEV.value
        assert client.stage == Stage.DEV

    def test_client_initialization_with_env_var(self):
        """Test client initialization using environment variable."""
        original_key = os.environ.get("FUTUREHOUSE_API_KEY")
        os.environ["FUTUREHOUSE_API_KEY"] = "env-test-key"
        
        try:
            client = RestClient(service_uri="https://api.example.com")
            assert client.api_key == "env-test-key"
        finally:
            # Restore original key or remove if it didn't exist
            if original_key is not None:
                os.environ["FUTUREHOUSE_API_KEY"] = original_key
            else:
                del os.environ["FUTUREHOUSE_API_KEY"]

    def test_client_default_stage(self):
        """Test that client uses PROD stage by default."""
        client = RestClient(api_key="test-key")
        assert client.stage == Stage.PROD
        assert client.base_url == Stage.PROD.value