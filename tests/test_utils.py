from dataclasses import dataclass
from typing import Any

from aviary.core import Message, ToolRequestMessage, ToolResponseMessage
from aviary.message import EnvStateMessage

from ldp.utils import format_error_details, split_message_transitions


@dataclass
class MockResponse:
    status_code: int
    _json: dict | None = None
    _text: str = ""

    def json(self) -> dict[str, Any]:
        if self._json is None:
            raise ValueError("No JSON")
        return self._json

    @property
    def text(self) -> str:
        return self._text


class MockHTTPError(Exception):
    def __init__(self, status_code: int, detail: str | None = None, text: str = ""):
        self.response = MockResponse(
            status_code=status_code,
            _json={"detail": detail} if detail else None,
            _text=text,
        )
        super().__init__(f"HTTP {status_code}")


def test_format_basic_error():
    error = ValueError("something went wrong")
    details = format_error_details(error)
    assert details == "something went wrong"


def test_format_http_error_with_json():
    error = MockHTTPError(
        status_code=500,
        detail="Traceback:\n  File 'app.py', line 123\n  raise ValueError('oops')",
    )
    details = format_error_details(error)
    assert "Status code: 500" in details
    assert "Server Traceback:" in details
    assert "File 'app.py'" in details


def test_format_http_error_with_text():
    error = MockHTTPError(status_code=404, text="Not found")
    details = format_error_details(error)
    assert "Status code: 404" in details
    assert "Response body: Not found" in details


def test_split_message_transitions():
    """Test message breakdown with EnvStateMessage→ToolRequestMessage transitions."""
    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Initial user message"),
        ToolRequestMessage(content="First tool request", tool_calls=[]),
        ToolResponseMessage(name="tool1", content="Tool response", tool_call_id="1"),
        EnvStateMessage(content="Environment state 1"),
        ToolRequestMessage(content="Second tool request", tool_calls=[]),
        ToolResponseMessage(name="tool2", content="Tool response 2", tool_call_id="2"),
        EnvStateMessage(content="Environment state 2"),
        ToolRequestMessage(content="Third tool request", tool_calls=[]),
        ToolResponseMessage(name="tool3", content="Tool response 3", tool_call_id="3"),
    ]

    result = split_message_transitions(messages)

    assert len(result) == 4

    # Block 1:
    assert len(result[0]) == 2
    assert result[0][0].role == "system"
    assert result[0][1].role == "user"

    # Block 2:
    assert len(result[1]) == 3
    assert isinstance(result[1][0], ToolRequestMessage)
    assert isinstance(result[1][1], ToolResponseMessage)
    assert isinstance(result[1][2], EnvStateMessage)

    # Block 3:
    assert len(result[2]) == 3
    assert isinstance(result[2][0], ToolRequestMessage)
    assert isinstance(result[2][1], ToolResponseMessage)
    assert isinstance(result[2][2], EnvStateMessage)

    # Block 4:
    assert len(result[3]) == 2
    assert isinstance(result[3][0], ToolRequestMessage)
    assert isinstance(result[3][1], ToolResponseMessage)
