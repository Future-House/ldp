import os
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import numpy as np
import pytest
from aviary.core import Message

from lmi.cost_tracker import GLOBAL_COST_TRACKER, cost_tracking_ctx
from lmi.embeddings import LiteLLMEmbeddingModel
from lmi.external.job_event_models import (
    CostComponent,
    ExecutionType,
    JobEventCreateRequest,
)
from lmi.llms import CommonLLMNames, LiteLLMModel
from lmi.utils import VCR_DEFAULT_MATCH_ON


@contextmanager
def assert_costs_increased():
    """All tests in this file should increase accumulated costs."""
    initial_cost = GLOBAL_COST_TRACKER.lifetime_cost_usd
    yield
    assert GLOBAL_COST_TRACKER.lifetime_cost_usd > initial_cost


@contextmanager
def remote_tracking_ctx(api_key: str = "test-api-key", execution_id: str | None = None):
    """Enable remote cost tracking for tests."""
    os.environ["FUTUREHOUSE_API_KEY"] = api_key
    if execution_id:
        os.environ["FUTUREHOUSE_EXECUTION_ID"] = execution_id
        os.environ["FUTUREHOUSE_EXECUTION_TYPE"] = "TRAJECTORY"
    try:
        yield
    finally:
        os.environ.pop("FUTUREHOUSE_API_KEY", None)
        os.environ.pop("FUTUREHOUSE_EXECUTION_ID", None)
        os.environ.pop("FUTUREHOUSE_EXECUTION_TYPE", None)


class TestLiteLLMEmbeddingCosts:
    @pytest.mark.asyncio
    async def test_embed_documents(self):
        stub_texts = ["test1", "test2"]
        with assert_costs_increased(), cost_tracking_ctx():
            model = LiteLLMEmbeddingModel(name="text-embedding-3-small", ndim=8)
            await model.embed_documents(stub_texts)


class TestLiteLLMModel:
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "model_name": CommonLLMNames.OPENAI_TEST.value,
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.OPENAI_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.OPENAI_TEST.value,
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ],
                },
                id="OpenAI-model",
            ),
            pytest.param(
                {
                    "model_name": CommonLLMNames.ANTHROPIC_TEST.value,
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.ANTHROPIC_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.ANTHROPIC_TEST.value,
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ],
                },
                id="Anthropic-model",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_cost_call(self, config: dict[str, Any]) -> None:
        with assert_costs_increased(), cost_tracking_ctx():
            llm = LiteLLMModel(name=config["model_name"], config=config)
            messages = [
                Message(role="system", content="Respond with single words."),
                Message(role="user", content="What is the meaning of the universe?"),
            ]
            await llm.call(messages)

    @pytest.mark.asyncio
    async def test_cost_call_w_figure(self) -> None:
        async def ac(x) -> None:
            pass

        with cost_tracking_ctx():
            with assert_costs_increased():
                llm = LiteLLMModel(name=CommonLLMNames.GPT_4O.value)
                image = np.zeros((32, 32, 3), dtype=np.uint8)
                image[:] = [255, 0, 0]
                messages = [
                    Message(
                        role="system",
                        content="You are a detective who investigate colors",
                    ),
                    Message.create_message(
                        role="user",
                        text=(
                            "What color is this square? Show me your chain of"
                            " reasoning."
                        ),
                        images=image,
                    ),
                ]  # TODO: It's not decoding the image. It's trying to guess the color from the encoded image string.
                await llm.call(messages)

            with assert_costs_increased():
                await llm.call(messages, [ac])

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(
                {
                    "model_list": [
                        {
                            "model_name": CommonLLMNames.OPENAI_TEST.value,
                            "litellm_params": {
                                "model": CommonLLMNames.OPENAI_TEST.value,
                                "temperature": 0,
                                "max_tokens": 56,
                            },
                        }
                    ]
                },
                id="with-router",
            ),
            pytest.param(
                {
                    "pass_through_router": True,
                    "router_kwargs": {"temperature": 0, "max_tokens": 56},
                },
                id="without-router",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_cost_call_single(self, config: dict[str, Any]) -> None:
        with cost_tracking_ctx(), assert_costs_increased():
            llm = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value, config=config)

            outputs = []

            def accum(x) -> None:
                outputs.append(x)

            prompt = "The {animal} says"
            data = {"animal": "duck"}
            system_prompt = "You are a helpful assistant."
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=prompt.format(**data)),
            ]

            await llm.call_single(
                messages=messages,
                callbacks=[accum],
            )


class TestRemoteCostTracking:
    @pytest.mark.asyncio
    async def test_remote_cost_tracking_unit_test(self) -> None:
        """Unit test for remote cost tracking without making real API calls."""
        execution_id = str(uuid4())

        # Mock the rest client
        mock_rest_client = AsyncMock()
        mock_rest_client.create_job_event = AsyncMock()

        # Mock the litellm response
        mock_response = MagicMock()
        mock_response.created = 1234567890
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        with (
            cost_tracking_ctx(),
            remote_tracking_ctx(execution_id=execution_id),
            patch.dict(
                os.environ,
                {"FUTUREHOUSE_API_KEY": "test-key"},  # pragma: allowlist secret
            ),
        ):
            # Set the rest client directly
            GLOBAL_COST_TRACKER._job_event_client = mock_rest_client
            try:
                # Mock the cost calculation
                with patch(
                    "litellm.cost_calculator.completion_cost", return_value=0.001
                ):
                    # Call the _report_job_event method directly
                    await GLOBAL_COST_TRACKER._report_job_event(
                        mock_response, UUID(execution_id), ExecutionType.TRAJECTORY
                    )
            finally:
                # Clean up
                GLOBAL_COST_TRACKER._job_event_client = None

        assert mock_rest_client.create_job_event.called

        call_args = mock_rest_client.create_job_event.call_args
        job_event_request = call_args[0][0]

        assert isinstance(job_event_request, JobEventCreateRequest)
        assert job_event_request.execution_id == UUID(execution_id)
        assert job_event_request.execution_type == ExecutionType.TRAJECTORY
        assert job_event_request.cost_component == CostComponent.LLM_USAGE
        assert job_event_request.amount_usd == 0.001
        assert job_event_request.input_token_count == 10
        assert job_event_request.completion_token_count == 5
        assert job_event_request.metadata is not None
        assert job_event_request.metadata["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_remote_cost_tracking_no_execution_id(self) -> None:
        """Test that remote tracking is not called when execution ID is missing."""
        mock_rest_client = AsyncMock()
        mock_rest_client.create_job_event = AsyncMock()

        with (
            cost_tracking_ctx(),
            remote_tracking_ctx(),
            patch.object(GLOBAL_COST_TRACKER, "_job_event_client", mock_rest_client),
            patch("litellm.cost_calculator.completion_cost", return_value=0.001),
        ):
            # Call _record_remote directly with no execution ID
            GLOBAL_COST_TRACKER._record_remote(MagicMock())

            # Verify the endpoint was NOT called (no execution ID)
            assert not mock_rest_client.create_job_event.called

    @pytest.mark.asyncio
    async def test_remote_cost_tracking_no_api_key(self) -> None:
        """Test that remote tracking is not enabled when API key is missing."""
        with cost_tracking_ctx(enabled=True):
            # Should not have rest client without API key
            assert GLOBAL_COST_TRACKER.rest_client is None

    @pytest.mark.asyncio
    async def test_remote_cost_tracking_error_handling(self) -> None:
        """Test that errors in remote tracking don't affect the main operation."""
        execution_id = str(uuid4())

        # Mock the rest client to raise an exception
        mock_rest_client = AsyncMock()
        mock_rest_client.create_job_event = AsyncMock(
            side_effect=Exception("API Error")
        )

        with (
            cost_tracking_ctx(enabled=True),
            remote_tracking_ctx(execution_id=execution_id),
            patch.dict(
                os.environ,
                {"FUTUREHOUSE_API_KEY": "test-key"},  # pragma: allowlist secret
            ),
        ):
            GLOBAL_COST_TRACKER._job_event_client = mock_rest_client
            try:
                # Mock the cost calculation
                with patch(
                    "litellm.cost_calculator.completion_cost", return_value=0.001
                ):
                    # Create a proper mock response
                    mock_response = MagicMock()
                    mock_response.created = 1234567890
                    mock_response.model = "gpt-4o-mini"
                    mock_response.usage = MagicMock()
                    mock_response.usage.prompt_tokens = 10
                    mock_response.usage.completion_tokens = 5

                    # Call the _report_job_event method directly
                    await GLOBAL_COST_TRACKER._report_job_event(
                        mock_response, UUID(execution_id), ExecutionType.TRAJECTORY
                    )
            finally:
                GLOBAL_COST_TRACKER._job_event_client = None

            # Verify the endpoint was called (and failed)
            assert mock_rest_client.create_job_event.called
