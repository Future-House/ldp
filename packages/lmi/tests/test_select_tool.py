"""Regression coverage for selection through the model chain."""

from unittest.mock import AsyncMock, Mock, patch

import litellm
import pytest
from aviary.core import Message, Tool, ToolRequestMessage
from aviary.message import MalformedMessageError
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse

from lmi.config import LLMConfig, ModelSpec
from lmi.cost_tracker import GLOBAL_COST_TRACKER, cost_tracking_ctx
from lmi.llms import LiteLLMModel
from lmi.types import LLMResult


def lookup(query: str) -> str:
    """Look up a query.

    Args:
        query: Text to look up.
    """
    return query


TOOL = Tool.from_function(lookup)
MESSAGES = [Message(content="Look up example.")]
FUNCTION = {"name": "lookup", "arguments": '{"query":"example"}'}
TOOL_CALL = {"id": "call_1", "type": "function", "function": FUNCTION.copy()}


def chat_response(finish_reason: str = "tool_calls") -> ModelResponse:
    return ModelResponse(
        model="gpt-4o",
        choices=[
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "tool_calls": [TOOL_CALL]},
            }
        ],
        usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("responses_api", [False, True])
@pytest.mark.parametrize("tool_choice", ["required", TOOL])
async def test_selection_fallback_and_accounting(
    responses_api: bool, tool_choice: Tool | str
) -> None:
    callback = Mock()
    model = LiteLLMModel(
        llm_config=LLMConfig(
            models=[
                ModelSpec(name="gpt-4o-mini", max_retries=0),
                ModelSpec(name="gpt-4o", responses_api=responses_api),
            ]
        ),
        llm_result_callback=callback,
        config={"n": 4},
    )
    response = (
        ResponsesAPIResponse(
            id="resp_selection",
            created_at=0,
            model="gpt-4o",
            status="completed",
            output=[
                {"type": "function_call", "call_id": "call_1", "id": "fc_1", **FUNCTION}
            ],
            usage={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        )
        if responses_api
        else chat_response()
    )
    failure = litellm.ContextWindowExceededError(
        message="too long", model="gpt-4o-mini", llm_provider="openai"
    )
    with (
        patch(
            "litellm.acompletion", AsyncMock(side_effect=[failure, response])
        ) as chat,
        patch("litellm.aresponses", AsyncMock(return_value=response)) as responses,
        patch.object(
            LiteLLMModel, "check_request_limit", new_callable=AsyncMock
        ) as request_limit,
        patch.object(
            LiteLLMModel, "check_rate_limit", new_callable=AsyncMock
        ) as rate_limit,
        patch.object(GLOBAL_COST_TRACKER, "record", new_callable=AsyncMock) as record,
        cost_tracking_ctx(),
    ):
        selection = await model.select_tool(MESSAGES, [TOOL], tool_choice)

    assert isinstance(selection, ToolRequestMessage)
    assert selection.tool_calls[0].function.arguments == {"query": "example"}
    assert selection.info == {
        "usage": (5, 3),
        "model": "gpt-4o",
        **({"response_id": "resp_selection"} if responses_api else {}),
    }
    attempts = chat.await_args_list + responses.await_args_list
    assert [attempt.kwargs["model"] for attempt in attempts] == [
        "gpt-4o-mini",
        "gpt-4o",
    ]
    kwargs = attempts[-1].kwargs
    assert kwargs["n"] == 1
    expected_choice = (
        (
            {"type": "function", "name": "lookup"}
            if responses_api
            else {"type": "function", "function": {"name": "lookup"}}
        )
        if isinstance(tool_choice, Tool)
        else tool_choice
    )
    assert kwargs["tool_choice"] == expected_choice
    assert request_limit.await_count == 2
    assert rate_limit.await_count == (2 if responses_api else 3)
    record.assert_awaited_once_with(response)
    callback.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "finish_reason", "count"),
    [
        ([ToolRequestMessage(tool_calls=[TOOL_CALL])], "tool_calls", 0),
        ([ToolRequestMessage(tool_calls=[TOOL_CALL])], "tool_calls", 2),
        ([], "stop", 1),
        ([Message(role="assistant", content="text")], "stop", 1),
        ([ToolRequestMessage()], "stop", 1),
        ([ToolRequestMessage(), ToolRequestMessage()], "tool_calls", 1),
        ([ToolRequestMessage(tool_calls=[TOOL_CALL])], "length", 1),
    ],
)
async def test_malformed_selection(
    messages: list[Message], finish_reason: str, count: int
) -> None:
    result = LLMResult(model="gpt-4o", messages=messages, finish_reason=finish_reason)
    with (
        patch.object(
            LiteLLMModel, "call", AsyncMock(return_value=[result] * count)
        ) as completion,
        pytest.raises(MalformedMessageError),
    ):
        await LiteLLMModel().select_tool(MESSAGES, [TOOL])
    completion.assert_awaited_once_with(
        MESSAGES, tools=[TOOL], tool_choice="required", n=1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_choice", "tools", "finish_reason", "accepted"),
    [
        ("required", [], "stop", True),
        ("auto", [TOOL], "tool_calls", True),
        ("none", [TOOL], "tool_calls", True),
        (None, [TOOL], "tool_calls", True),
        ("auto", [TOOL], "stop", False),
        ("none", [TOOL], "stop", False),
        (None, [TOOL], "stop", False),
    ],
)
async def test_optional_selection_and_metadata(
    tool_choice: str | None, tools: list[Tool], finish_reason: str, accepted: bool
) -> None:
    message = ToolRequestMessage(content="No tool needed.", info={"trace": "keep"})
    result = LLMResult(model="gpt-4o", messages=[message], finish_reason=finish_reason)
    with patch.object(LiteLLMModel, "call", AsyncMock(return_value=[result])):
        if not accepted:
            with pytest.raises(MalformedMessageError, match="finish reason"):
                await LiteLLMModel().select_tool(MESSAGES, tools, tool_choice)
            return
        selection = await LiteLLMModel().select_tool(MESSAGES, tools, tool_choice)
    assert not selection.tool_calls
    assert selection.info == {"trace": "keep", "usage": (0, 0), "model": "gpt-4o"}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_choice", ["required", TOOL])
async def test_custom_parser_metadata(tool_choice: Tool | str) -> None:
    selection = ToolRequestMessage(tool_calls=[TOOL_CALL], info={"trace": "keep"})
    model = LiteLLMModel(name="gpt-4o", tool_parser=lambda *_: selection)
    with patch("litellm.acompletion", AsyncMock(return_value=chat_response("stop"))):
        actual = await model.select_tool(MESSAGES, [TOOL], tool_choice)
    assert actual is selection
    assert actual.info == {"trace": "keep", "usage": (5, 3), "model": "gpt-4o"}
