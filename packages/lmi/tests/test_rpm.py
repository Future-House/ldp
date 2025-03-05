"""Tests for request rate limit (RPM) functionality in LiteLLMModel.

This module contains tests to verify the rate limiting functionality,
including both sequential and concurrent request scenarios.
"""

import asyncio
import time

import pytest

from lmi.llms import CommonLLMNames, LiteLLMModel
from lmi.types import Message


def _setup_model_config(rpm_limit: int | None = None) -> tuple[dict, str]:
    """Set up the model configuration based on RPM limit.

    Args:
        rpm_limit: The RPM limit to test with. None means no limit.

    Returns:
        Tuple containing the model configuration and test name.
    """
    model_config = {
        "model_list": [
            {
                "model_name": CommonLLMNames.OPENAI_TEST.value,
                "litellm_params": {
                    "model": CommonLLMNames.OPENAI_TEST.value,
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
            }
        ]
    }

    if rpm_limit is not None:
        model_config["request_limit"] = {  # type: ignore[assignment]
            "deepseek/deepseek-chat": f"{rpm_limit}/minute"
        }

    return model_config, f"RPM={rpm_limit}" if rpm_limit else "No Limit"


async def _run_request_test(
    is_concurrent: bool, req_count: int, req_limit: int
) -> None:
    """Run request test with and without RPM limit.

    Args:
        is_concurrent: Whether to run requests concurrently.
        req_count: Number of requests to send.
        req_limit: RPM limit to test with.
    """
    messages = [Message(role="user", content="Hi")]
    results = {}

    # Test without RPM limit
    model_config, test_name = _setup_model_config()
    model = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value, config=model_config)

    start_time = time.perf_counter()
    if is_concurrent:
        concurrent_tasks = [model.call_single(messages) for _ in range(req_count)]
        await asyncio.gather(*concurrent_tasks)
    else:
        for _ in range(req_count):
            await model.call_single(messages)
    results[test_name] = time.perf_counter() - start_time

    # Test with RPM limit
    model_config, test_name = _setup_model_config(req_limit)
    model = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value, config=model_config)

    # Wait for rate limit to reset and ensure we start at the beginning of a minute
    await asyncio.sleep(60)

    start_time = time.perf_counter()
    if is_concurrent:
        concurrent_tasks = [model.call_single(messages) for _ in range(req_count)]
        await asyncio.gather(*concurrent_tasks)
    else:
        for _ in range(req_count):
            await model.call_single(messages)
    results[test_name] = time.perf_counter() - start_time

    # Validate RPM test results
    # Get time with RPM limit (any key starting with "RPM=")
    time_with_limit = next(
        (v for k, v in results.items() if k.startswith("RPM=")), None
    )
    time_no_limit = results.get("No Limit")

    if time_with_limit is None or time_no_limit is None:
        pytest.fail("Missing test results for either rate-limited or unlimited case")

    # With RPM limit, completing requests should take appropriate time
    min_expected_time = 60.0
    error_msg = (
        f"With RPM limit, requests should take at least {min_expected_time} seconds, "
        f"but only took {time_with_limit:.2f} seconds"
    )
    assert time_with_limit >= min_expected_time, error_msg

    # The time with limit should be significantly longer than without limit
    error_msg = (
        "Expected time with RPM limit to be significantly longer than without limit"
    )
    assert time_with_limit > time_no_limit * 1.5, error_msg


@pytest.mark.asyncio
async def test_sequential_requests():
    """Test sequential requests with and without RPM limit.

    This test sends requests one after another and verifies that rate limiting
    properly throttles the requests when enabled.
    """
    await _run_request_test(is_concurrent=False, req_count=2, req_limit=1)


@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test concurrent requests with and without RPM limit.

    This test sends multiple requests simultaneously and verifies that rate limiting
    properly throttles the requests when enabled.
    """
    await _run_request_test(is_concurrent=True, req_count=5, req_limit=4)


if __name__ == "__main__":
    asyncio.run(test_sequential_requests())
