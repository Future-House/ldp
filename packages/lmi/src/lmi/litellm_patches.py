"""LiteLLM monkeypatches for known bugs and compatibility issues.

This module applies patches at import time to fix issues in litellm that haven't
been fixed upstream or were closed as stale without resolution.

Patches applied:
1. OpenAI BaseModel.model_dump pydantic v2 fix (by_alias=None issue)
2. Vertex AI context caching fix (tools + cachedContent conflict)
"""


# Patch 1: OpenAI BaseModel.model_dump pydantic v2 fix
#
# Pydantic v2 + openai/litellm has an issue where model_dump is called with
# by_alias=None, which fails. The OpenAI SDK only fixes this for pydantic v1.
# LiteLLM inherits from OpenAI's BaseModel, so we patch it here.
# See: https://github.com/anthropics/anthropic-sdk-python/pull/1165
def _apply_model_dump_patch():
    try:
        from openai._models import (  # noqa: PLC2701
            BaseModel as OpenAIBaseModel,
        )
    except ImportError:
        return  # OpenAI SDK not installed, skip patch

    original_model_dump = OpenAIBaseModel.model_dump

    def _patched_model_dump(self, *args, by_alias=None, **kwargs):
        return original_model_dump(
            self, *args, by_alias=by_alias if by_alias is not None else False, **kwargs
        )

    OpenAIBaseModel.model_dump = _patched_model_dump  # type: ignore[method-assign]


_apply_model_dump_patch()


# Patch 2: Vertex AI context caching fix
#
# Bug: LiteLLM sends both cachedContent AND tools/system_instruction in Gemini
# generateContent requests. Gemini's API requires that when using cached content,
# these fields must NOT be in the request (they're already in the cache).
#
# Error: "Tool config, tools and system instruction should not be set in the
# request when using cached content." (400 INVALID_ARGUMENT)
#
# Root cause: _transform_request_body adds tools, toolConfig, and system_instruction
# to the request unconditionally, even when cachedContent is present.
# https://github.com/BerriAI/litellm/blob/v1.82.3/litellm/llms/vertex_ai/gemini/transformation.py
#
# This patch wraps _transform_request_body to strip those fields from the request
# when cachedContent is set, since they're already included in the cached content.
#
# Upstream issue (closed by stale bot, not fixed):
# https://github.com/BerriAI/litellm/issues/17304
def _apply_vertex_caching_fix():
    try:
        from litellm.llms.vertex_ai.gemini import (
            transformation as _vertex_transform,
        )
    except ImportError:
        return  # Vertex support not available, skip patch

    original_transform_request_body = _vertex_transform._transform_request_body

    def _patched_transform_request_body(*args, **kwargs):
        data = original_transform_request_body(*args, **kwargs)
        # If cachedContent is set, remove tools/system_instruction/toolConfig
        # (they're already in the cache - Gemini rejects duplicates)
        if data.get("cachedContent") is not None:
            data.pop("tools", None)
            data.pop("toolConfig", None)
            data.pop("system_instruction", None)
        return data

    _vertex_transform._transform_request_body = _patched_transform_request_body


_apply_vertex_caching_fix()
