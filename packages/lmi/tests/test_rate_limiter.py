import asyncio
import time
from itertools import product
from typing import Any
from unittest.mock import patch

import httpx_aiohttp
import pytest
from aviary.core import Message
from limits import RateLimitItemPerSecond

from lmi.constants import CHARACTERS_PER_TOKEN_ASSUMPTION
from lmi.embeddings import LiteLLMEmbeddingModel
from lmi.llms import CommonLLMNames, LiteLLMModel
from lmi.rate_limiter import CROSSREF_BASE_URL, FALLBACK_RATE_LIMIT, GlobalRateLimiter
from lmi.types import LLMResult

LLM_CONFIG_W_RATE_LIMITS = [
    # following ensures that "short-form" rate limits are also supported
    # where the user doesn't specify the model_list
    {
        "name": CommonLLMNames.OPENAI_TEST.value,
        "config": {
            "rate_limit": {
                CommonLLMNames.OPENAI_TEST.value: RateLimitItemPerSecond(20, 3)
            },
        },
    },
    {
        "name": CommonLLMNames.OPENAI_TEST.value,
        "config": {
            "model_list": [
                {
                    "model_name": CommonLLMNames.OPENAI_TEST.value,
                    "litellm_params": {
                        "model": CommonLLMNames.OPENAI_TEST.value,
                        "temperature": 0,
                    },
                }
            ],
            "rate_limit": {
                CommonLLMNames.OPENAI_TEST.value: RateLimitItemPerSecond(20, 1)
            },
        },
    },
    {
        "name": CommonLLMNames.OPENAI_TEST.value,
        "config": {
            "model_list": [
                {
                    "model_name": CommonLLMNames.OPENAI_TEST.value,
                    "litellm_params": {
                        "model": CommonLLMNames.OPENAI_TEST.value,
                        "temperature": 0,
                    },
                }
            ],
            "rate_limit": {
                CommonLLMNames.OPENAI_TEST.value: RateLimitItemPerSecond(1_000_000, 1)
            },
        },
    },
    {
        "name": CommonLLMNames.OPENAI_TEST.value,
        "config": {
            "model_list": [
                {
                    "model_name": CommonLLMNames.OPENAI_TEST.value,
                    "litellm_params": {
                        "model": CommonLLMNames.OPENAI_TEST.value,
                        "temperature": 0,
                    },
                }
            ]
        },
    },
]

RATE_LIMITER_PROMPT = "Animals make many noises. The duck says"

LLM_METHOD_AND_INPUTS = [
    {
        "method": "acompletion",
        "kwargs": {
            "messages": [Message.create_message(role="user", text=RATE_LIMITER_PROMPT)]
        },
    },
    {
        "method": "acompletion_iter",
        "kwargs": {
            "messages": [Message.create_message(role="user", text=RATE_LIMITER_PROMPT)]
        },
    },
]

rate_limit_configurations = list(
    product(LLM_CONFIG_W_RATE_LIMITS, LLM_METHOD_AND_INPUTS)
)

EMBEDDING_CONFIG_W_RATE_LIMITS = [
    {"config": {"rate_limit": RateLimitItemPerSecond(20, 5)}},
    {"config": {"rate_limit": RateLimitItemPerSecond(20, 3)}},
    {"config": {"rate_limit": RateLimitItemPerSecond(1_000_000, 1)}},
    {},
]

# Test configurations with different request counts and scenarios
SEQUENTIAL_REQUEST_LIMITS = {
    "name": CommonLLMNames.OPENAI_TEST.value,
    "config": {
        "model_list": [
            {
                "model_name": CommonLLMNames.OPENAI_TEST.value,
                "litellm_params": {"model": CommonLLMNames.OPENAI_TEST.value},
            }
        ],
        "request_limit": {
            CommonLLMNames.OPENAI_TEST.value: RateLimitItemPerSecond(1, 20)
        },
    },
}

CONCURRENT_REQUEST_LIMITS = {
    "name": CommonLLMNames.OPENAI_TEST.value,
    "config": {
        "model_list": [
            {
                "model_name": CommonLLMNames.OPENAI_TEST.value,
                "litellm_params": {"model": CommonLLMNames.OPENAI_TEST.value},
            }
        ],
        "request_limit": {
            CommonLLMNames.OPENAI_TEST.value: RateLimitItemPerSecond(4, 20)
        },
    },
}


ACCEPTABLE_RATE_LIMIT_ERROR: float = 0.10  # 10% error margin for token estimate error


async def time_n_llm_methods(
    llm: LiteLLMModel, method: str, n: int, use_gather: bool = False, *args, **kwargs
) -> float:
    """Give the token per second rate of a method call."""
    start_time = time.time()
    outputs = []

    if not use_gather:
        for _ in range(n):
            if "iter" in method:
                outputs.extend([
                    output
                    async for output in await getattr(llm, method)(*args, **kwargs)
                ])
            else:
                outputs.append(await getattr(llm, method)(*args, **kwargs))

    else:
        outputs = await asyncio.gather(*[
            getattr(llm, method)(*args, **kwargs) for _ in range(n)
        ])

    character_count = 0
    token_count = 0

    if isinstance(outputs[0], LLMResult):
        character_count = sum(len(o.text or "") for o in outputs)
    else:
        character_count = sum(len(o) for o in outputs)

    if hasattr(outputs[0], "prompt_count"):
        token_count = sum(o.prompt_count + o.completion_count for o in outputs)

    return (
        (character_count / CHARACTERS_PER_TOKEN_ASSUMPTION)
        if token_count == 0
        else token_count
    ) / (time.time() - start_time)


@pytest.mark.parametrize("llm_config_w_rate_limits", LLM_CONFIG_W_RATE_LIMITS)
@pytest.mark.asyncio
async def test_rate_limit_on_call_single(
    llm_config_w_rate_limits: dict[str, Any],
) -> None:
    llm = LiteLLMModel(**llm_config_w_rate_limits)

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

    estimated_tokens_per_second = await time_n_llm_methods(
        llm,
        "call_single",
        3,
        messages=messages,
        callbacks=[accum],
    )

    if "rate_limit" in llm.config:
        max_tokens_per_second = (
            llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].amount
            / llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].multiples
        )
        assert estimated_tokens_per_second / max_tokens_per_second < (
            1.0 + ACCEPTABLE_RATE_LIMIT_ERROR
        )
    else:
        assert estimated_tokens_per_second > 0

    outputs = []

    def accum2(x) -> None:
        outputs.append(x)

    estimated_tokens_per_second = await time_n_llm_methods(
        llm,
        "call_single",
        3,
        use_gather=True,
        messages=messages,
        callbacks=[accum2],
    )

    if "rate_limit" in llm.config:
        max_tokens_per_second = (
            llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].amount
            / llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].multiples
        )
        assert estimated_tokens_per_second / max_tokens_per_second < (
            1.0 + ACCEPTABLE_RATE_LIMIT_ERROR
        )
    else:
        assert estimated_tokens_per_second > 0


@pytest.mark.parametrize(
    ("llm_config_w_rate_limits", "llm_method_kwargs"), rate_limit_configurations
)
@pytest.mark.asyncio
async def test_rate_limit_on_sequential_completion_litellm_methods(
    llm_config_w_rate_limits: dict[str, Any],
    llm_method_kwargs: dict[str, Any],
) -> None:
    llm = LiteLLMModel(**llm_config_w_rate_limits)

    estimated_tokens_per_second = await time_n_llm_methods(
        llm,
        llm_method_kwargs["method"],
        3,
        use_gather=False,
        **llm_method_kwargs["kwargs"],
    )
    if "rate_limit" in llm.config:
        max_tokens_per_second = (
            llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].amount
            / llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].multiples
        )
        assert estimated_tokens_per_second / max_tokens_per_second < (
            1.0 + ACCEPTABLE_RATE_LIMIT_ERROR
        )
    else:
        assert estimated_tokens_per_second > 0


@pytest.mark.parametrize(
    ("llm_config_w_rate_limits", "llm_method_kwargs"), rate_limit_configurations
)
@pytest.mark.asyncio
async def test_rate_limit_on_parallel_completion_litellm_methods(
    llm_config_w_rate_limits: dict[str, Any],
    llm_method_kwargs: dict[str, Any],
) -> None:
    llm = LiteLLMModel(**llm_config_w_rate_limits)

    if "iter" not in llm_method_kwargs["method"]:
        estimated_tokens_per_second = await time_n_llm_methods(
            llm,
            llm_method_kwargs["method"],
            3,
            use_gather=True,
            **llm_method_kwargs["kwargs"],
        )
        if "rate_limit" in llm.config:
            max_tokens_per_second = (
                llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].amount
                / llm.config["rate_limit"][CommonLLMNames.OPENAI_TEST.value].multiples
            )
            assert estimated_tokens_per_second / max_tokens_per_second < (
                1.0 + ACCEPTABLE_RATE_LIMIT_ERROR
            )
        else:
            assert estimated_tokens_per_second > 0


@pytest.mark.parametrize(
    "embedding_config_w_rate_limits", EMBEDDING_CONFIG_W_RATE_LIMITS
)
@pytest.mark.asyncio
async def test_embedding_rate_limits(
    embedding_config_w_rate_limits: dict[str, Any],
) -> None:
    embedding_model = LiteLLMEmbeddingModel(**embedding_config_w_rate_limits)
    embedding_model.config["batch_size"] = 5
    texts_to_embed = ["the duck says"] * 10
    start = time.time()
    await embedding_model.embed_documents(texts=texts_to_embed)
    estimated_tokens_per_second = sum(
        len(t) / CHARACTERS_PER_TOKEN_ASSUMPTION for t in texts_to_embed
    ) / (time.time() - start)

    if "rate_limit" in embedding_config_w_rate_limits:
        max_tokens_per_second = (
            embedding_config_w_rate_limits["rate_limit"].amount
            / embedding_config_w_rate_limits["rate_limit"].multiples
        )
        assert estimated_tokens_per_second / max_tokens_per_second < (
            1.0 + ACCEPTABLE_RATE_LIMIT_ERROR
        )
    else:
        assert estimated_tokens_per_second > 0


async def _run_rpm_request_test(
    is_concurrent: bool, llm_config_w_request_limits: dict[str, Any]
) -> None:
    """Run request test with RPM limit.

    Args:
        is_concurrent: Whether to run requests concurrently.
        llm_config_w_request_limits: llm config with request configuration parameters.
    """
    rate_limit_item = next(
        iter(llm_config_w_request_limits["config"]["request_limit"].values())
    )
    req_request = rate_limit_item.amount
    req_count = req_request + 1

    messages = [Message(content=RATE_LIMITER_PROMPT)]
    model = LiteLLMModel(**llm_config_w_request_limits)

    start_time = time.perf_counter()
    if is_concurrent:
        concurrent_tasks = [model.call_single(messages) for _ in range(req_count)]
        await asyncio.gather(*concurrent_tasks)
    else:
        for _ in range(req_count):
            await model.call_single(messages)
    time_with_limit = time.perf_counter() - start_time

    min_expected_time = rate_limit_item.multiples
    error_msg = (
        f"With rate limit of {rate_limit_item}, {req_count} requests "
        f"take at least {min_expected_time} seconds, but only took {time_with_limit:.2f} seconds"
    )
    assert time_with_limit >= min_expected_time, error_msg


@pytest.mark.parametrize("llm_config_w_request_limits", [SEQUENTIAL_REQUEST_LIMITS])
@pytest.mark.asyncio
async def test_rpm_sequential_requests(llm_config_w_request_limits: dict[str, Any]):
    """Test sequential requests with RPM limit.

    This test sends requests one after another and verifies that rate limiting
    properly throttles the requests.
    """
    await _run_rpm_request_test(
        is_concurrent=False, llm_config_w_request_limits=llm_config_w_request_limits
    )


@pytest.mark.parametrize("llm_config_w_request_limits", [CONCURRENT_REQUEST_LIMITS])
@pytest.mark.asyncio
async def test_rpm_concurrent_requests(llm_config_w_request_limits: dict[str, Any]):
    """Test concurrent requests with RPM limit.

    This test sends multiple requests simultaneously and verifies that rate limiting
    properly throttles the requests.
    """
    await _run_rpm_request_test(
        is_concurrent=True, llm_config_w_request_limits=llm_config_w_request_limits
    )


class TestGlobalRateLimiter:
    @pytest.mark.asyncio
    async def test_parsing_namespace(self) -> None:
        limiter = GlobalRateLimiter()
        async with (  # Throwaway client to build a request
            httpx_aiohttp.HttpxAiohttpClient() as client
        ):
            req = client.build_request(method="GET", url=f"{CROSSREF_BASE_URL}/stub")
        with patch.object(
            limiter.rate_limiter,
            "hit",
            side_effect=limiter.rate_limiter.hit,
            autospec=True,
        ) as mock_hit:
            await limiter.try_acquire((req.method, str(req.url.host)))
            await limiter.try_acquire((req.method.lower(), str(req.url.host)))
        assert mock_hit.await_count == 2
        # Our rate limiter's namespace is case-sensitive,
        # so the first call won't match a rate limit (GET vs get),
        # but the second call will
        assert mock_hit.await_args_list[0][0][0] == FALLBACK_RATE_LIMIT
        assert mock_hit.await_args_list[1][0][0] > FALLBACK_RATE_LIMIT
