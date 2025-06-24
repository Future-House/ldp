from dataclasses import dataclass
from typing import Any

from ldp.utils import format_error_details


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
