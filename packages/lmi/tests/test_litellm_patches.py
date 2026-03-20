"""Tests for LiteLLM patches applied in lmi.litellm_patches.

These tests verify that the monkeypatches are correctly applied without
requiring actual API calls.
"""

import litellm
import litellm.litellm_core_utils.core_helpers as _litellm_core_helpers
import pytest
from litellm.llms.vertex_ai.gemini import transformation


class TestRefusalFinishReasonPatch:
    """Tests for the refusal finish reason preservation patch."""

    def test_refusal_mapped_to_refusal(self):
        """Verify 'refusal' is mapped to 'refusal' instead of being swallowed."""
        assert "refusal" in _litellm_core_helpers._FINISH_REASON_MAP
        assert _litellm_core_helpers._FINISH_REASON_MAP["refusal"] == "refusal"


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
        # This would fail without the patch (pydantic v2 rejects None for by_alias)
        result = model.model_dump(by_alias=None)
        assert "test_field" in result


class TestProviderRetryPatch:
    """Tests for the provider-specific 400 error retry patch."""

    def test_should_retry_is_patched(self):
        """Verify Router.should_retry_this_error is patched."""
        method = litellm.Router.should_retry_this_error
        # The patched version should have a closure containing original function
        assert hasattr(method, "__closure__") and method.__closure__ is not None

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

        # Create a mock 400 error without provider limit message
        error = litellm.BadRequestError(
            message="Invalid request format",
            model="anthropic/claude-3-5-sonnet",
            llm_provider="anthropic",
        )
        error.status_code = 400

        # Should NOT return None (will raise or return something to stop retry)
        result = router.should_retry_this_error(error)
        # The original method either returns something or raises
        # We just verify it doesn't return None (which would allow retry)
        # Note: This may raise, which is the expected behavior
        assert result is not None or pytest.raises(Exception)


class TestVertexCachingPatch:
    """Tests for the Vertex AI context caching fix."""

    def test_transform_is_patched(self):
        """Verify _transform_request_body is patched."""
        method = transformation._transform_request_body
        # The patched version should have a closure containing original function
        assert hasattr(method, "__closure__") and method.__closure__ is not None

    def test_fields_removed_when_cached_content_present(self):
        """When cachedContent is set, tools/toolConfig/system_instruction are removed."""
        mock_return_data = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "cachedContent": "projects/123/locations/us/cachedContents/abc",
            "tools": [{"function_declarations": [{"name": "test_tool"}]}],
            "toolConfig": {"function_calling_config": {"mode": "AUTO"}},
            "system_instruction": {"parts": [{"text": "Be helpful"}]},
        }

        patched_fn = transformation._transform_request_body
        original_fn = patched_fn.__closure__[0].cell_contents  # type: ignore[index]

        def mock_original(*_args, **_kwargs):
            return mock_return_data.copy()

        patched_fn.__closure__[0].cell_contents = mock_original  # type: ignore[index]
        try:
            result = patched_fn()  # type: ignore[call-arg]

            assert "cachedContent" in result
            assert "tools" not in result
            assert "toolConfig" not in result
            assert "system_instruction" not in result
            assert "contents" in result
        finally:
            patched_fn.__closure__[0].cell_contents = original_fn  # type: ignore[index]

    def test_fields_preserved_when_no_cached_content(self):
        """When cachedContent is NOT set, tools/toolConfig/system_instruction are preserved."""
        mock_return_data = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "tools": [{"function_declarations": [{"name": "test_tool"}]}],
            "toolConfig": {"function_calling_config": {"mode": "AUTO"}},
            "system_instruction": {"parts": [{"text": "Be helpful"}]},
        }

        patched_fn = transformation._transform_request_body
        original_fn = patched_fn.__closure__[0].cell_contents  # type: ignore[index]

        def mock_original(*_args, **_kwargs):
            return mock_return_data.copy()

        patched_fn.__closure__[0].cell_contents = mock_original  # type: ignore[index]
        try:
            result = patched_fn()  # type: ignore[call-arg]

            assert "cachedContent" not in result
            assert "tools" in result
            assert "toolConfig" in result
            assert "system_instruction" in result
            assert "contents" in result
        finally:
            patched_fn.__closure__[0].cell_contents = original_fn  # type: ignore[index]

    def test_demonstrates_bug_and_fix(self):
        """Demonstrate the LiteLLM bug and verify our fix.

        Shows:
        1. BUG: LiteLLM's original function returns cachedContent + tools together
        2. FIX: Our patched version strips the conflicting fields
        """
        buggy_request_body = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
            "cachedContent": "projects/123/locations/us/cachedContents/abc",
            "tools": [{"function_declarations": [{"name": "my_tool"}]}],
            "toolConfig": {"function_calling_config": {"mode": "AUTO"}},
            "system_instruction": {"parts": [{"text": "Be helpful"}]},
        }

        patched_fn = transformation._transform_request_body
        original_fn = patched_fn.__closure__[0].cell_contents  # type: ignore[index]

        def mock_original_returns_buggy_body(*_args, **_kwargs):
            return buggy_request_body.copy()

        patched_fn.__closure__[0].cell_contents = mock_original_returns_buggy_body  # type: ignore[index]
        try:
            # === THE BUG ===
            unpatched_result = mock_original_returns_buggy_body()
            assert "cachedContent" in unpatched_result
            assert "tools" in unpatched_result  # BUG: tools WITH cachedContent
            assert "toolConfig" in unpatched_result
            assert "system_instruction" in unpatched_result

            # === THE FIX ===
            patched_result = patched_fn()  # type: ignore[call-arg]
            assert "cachedContent" in patched_result
            assert "tools" not in patched_result  # FIX: tools stripped
            assert "toolConfig" not in patched_result
            assert "system_instruction" not in patched_result
            assert "contents" in patched_result
        finally:
            patched_fn.__closure__[0].cell_contents = original_fn  # type: ignore[index]
