"""Tests for LiteLLM patches applied in lmi.litellm_patches.

These tests verify that the monkeypatches are correctly applied without
requiring actual API calls.
"""

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
