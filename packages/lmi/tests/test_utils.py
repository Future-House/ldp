import base64

import pytest

from lmi.utils import bytes_to_string, string_to_bytes


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(b"Hello, World!", id="simple-text"),
        pytest.param(b"", id="empty-bytes"),
        pytest.param(bytes([0, 1, 2, 255, 128, 64]), id="binary-data"),
        pytest.param(b"Test data for base64 encoding", id="base64-validation"),
        pytest.param("Hello 世界 🌍".encode(), id="utf8-text"),
    ],
)
def test_str_bytes_conversions(value: bytes) -> None:
    # Test round-trip conversion
    encoded_string = bytes_to_string(value)
    decoded_bytes = string_to_bytes(encoded_string)
    assert decoded_bytes == value

    # Validate that encoded string is valid base64
    assert base64.b64decode(encoded_string) == value
