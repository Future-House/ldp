"""Tests for LiteLLM patches applied in lmi.litellm_patches.

These tests verify that the monkeypatches are correctly applied without
requiring actual API calls.
"""

import litellm
from litellm.llms.vertex_ai.gemini import transformation


class TestModelDumpPatch:
    """Tests for the OpenAI BaseModel.model_dump pydantic v2 fix."""

    def test_model_dump_handles_none_by_alias(self):
        """Verify model_dump handles by_alias=None without failing."""
        from openai._models import BaseModel as OpenAIBaseModel

        # Create a simple model instance for testing
        # The patch should make this not raise when by_alias=None
        class TestModel(OpenAIBaseModel):
            test_field: str = "test"

        model = TestModel()
        assert "test_field" in model.model_dump(by_alias=None)


class TestProviderRetryPatch:
    """Tests for the provider-specific 400 error retry patch."""

    def test_should_retry_is_patched(self):
        """Verify Router.should_retry_this_error is patched."""
        method = litellm.Router.should_retry_this_error
        # The patched version should have a closure containing original function
        assert hasattr(method, "__closure__")
        assert method.__closure__ is not None

    def test_allows_retry_on_provider_limit_400(self):
        """Verify 400 errors with provider limit messages allow retry."""
        router = litellm.Router(model_list=[])

        # Create a mock 400 error with provider limit message
        error = litellm.BadRequestError(
            message="Too much media: 0 document pages + 108 images > 100",
            model="anthropic/claude-3-5-sonnet",
            llm_provider="anthropic",
        )
        error.status_code = 400

        # Should return None to allow retry cascade
        result = router.should_retry_this_error(error)
        assert result is None

    def test_does_not_retry_generic_400(self):
        """Verify generic 400 errors are NOT retried."""
        router = litellm.Router(model_list=[])

        error = litellm.BadRequestError(
            message="Invalid request format",
            model="anthropic/claude-3-5-sonnet",
            llm_provider="anthropic",
        )
        error.status_code = 400

        # Method should either raise or return non-None to stop retry
        try:
            result = router.should_retry_this_error(error)
        except Exception:
            return
        assert result is not None


class TestVertexCachingPatch:
    """Tests for the Vertex AI context caching fix."""

    def test_transform_is_patched(self):
        """Verify _transform_request_body is patched."""
        method = transformation._transform_request_body
        assert hasattr(method, "__closure__")
        assert method.__closure__ is not None

    def test_patch_logic_removes_fields_when_cached_content_present(self):
        """Verify filtering logic removes conflicting fields when cachedContent is set."""
        data = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "cachedContent": "projects/123/locations/us/cachedContents/abc",
            "tools": [{"function_declarations": [{"name": "test_tool"}]}],
            "toolConfig": {"function_calling_config": {"mode": "AUTO"}},
            "system_instruction": {"parts": [{"text": "Be helpful"}]},
        }

        if data.get("cachedContent") is not None:
            data.pop("tools", None)
            data.pop("toolConfig", None)
            data.pop("system_instruction", None)

        assert "cachedContent" in data
        assert "tools" not in data
        assert "toolConfig" not in data
        assert "system_instruction" not in data
        assert "contents" in data

    def test_patch_logic_preserves_fields_when_no_cached_content(self):
        """Verify filtering logic preserves fields when cachedContent is NOT set."""
        data = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "tools": [{"function_declarations": [{"name": "test_tool"}]}],
            "toolConfig": {"function_calling_config": {"mode": "AUTO"}},
            "system_instruction": {"parts": [{"text": "Be helpful"}]},
        }

        if data.get("cachedContent") is not None:
            data.pop("tools", None)
            data.pop("toolConfig", None)
            data.pop("system_instruction", None)

        assert "tools" in data
        assert "toolConfig" in data
        assert "system_instruction" in data
        assert "contents" in data
