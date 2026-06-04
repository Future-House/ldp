"""Configuration types for LMI.

An `LLMConfig` is an ordered chain of `ModelSpec` entries. `models[0]` is the
primary model; `models[1:]` are fallbacks tried in order when the primary
fails in ways that another model might handle.

`LLMConfig.from_legacy_dict` accepts the dict-shaped configuration
(`{model_list, fallbacks, router_kwargs}`) that mirrors litellm's Router layout.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from itertools import chain
from typing import Any, Self

import litellm
from aviary.core import Message, ToolRequestMessage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)

from lmi.constants import DEFAULT_VERTEX_SAFETY_SETTINGS
from lmi.types import LLMResult

ResponseValidator = Callable[[LLMResult], Awaitable[None] | None]
ToolParser = (
    Callable[[litellm.utils.Choices, list[dict] | None], Message | ToolRequestMessage]
    | Callable[[str, list[dict] | None], Message | ToolRequestMessage]
)

_DEFAULT_TEMPERATURE = 1.0
_PROVIDER_GATED_PARAMS = frozenset({"logprobs", "top_logprobs"})

# Legacy `litellm_params` keys that map onto dedicated `ModelSpec` fields rather
# than flowing through into `extra_params`.
_RESERVED_LEGACY_PARAMS = frozenset({
    "model",
    "api_base",
    "api_key",
    "timeout",
    "max_retries",
})

# Per-call retry kwargs that LiteLLM honors via its own internal retry loop. LMI
# owns retries through `_run_with_fallbacks` + `ModelSpec.max_retries`, so these
# must never reach `litellm.acompletion`/`litellm.aresponses` regardless of how
# they ended up in our `ModelSpec.extra_params`.
_LITELLM_RETRY_KWARGS = frozenset({"num_retries", "max_retries"})

# `tool_parser` is an LMI-level callable handled on `LiteLLMModel`; if it ever
# ends up in `extra_params` (e.g. via a name-shape dict that wasn't unpacked
# by the `LLMConfig` before-validator), filter it before dispatching to litellm.
_NON_LITELLM_EXTRA_PARAMS = _LITELLM_RETRY_KWARGS | frozenset({"tool_parser"})


class ModelSpec(BaseModel):
    """One model in an `LLMConfig` chain."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description=(
            "LiteLLM model string in 'provider/model' format, "
            "e.g. 'openai/gpt-4o-mini' or 'anthropic/claude-3-5-sonnet-20241022'."
        ),
        examples=["openai/gpt-4o-mini"],
    )
    api_base: str | None = None
    api_key: SecretStr | None = None
    timeout: float | None = Field(
        default=60.0,
        description="Per-request timeout in seconds, or None to not set one.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description=(
            "Retries against this model before falling over to the next entry"
            " in the chain. 0 means attempt the model once with no retry before"
            " falling over."
        ),
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Pass-through kwargs for litellm.acompletion / litellm.aresponses,"
            " e.g. temperature, max_tokens, safety_settings, vertex_project."
        ),
    )
    responses_api: bool = Field(
        default=False,
        description=(
            "If True, dispatch this model via OpenAI's stateful Responses API"
            " (`litellm.aresponses`) instead of the Chat Completions API"
            " (`litellm.acompletion`)."
        ),
    )

    @model_validator(mode="after")
    def _reject_unsupported_params(self) -> Self:
        """Gate `logprobs`/`top_logprobs` against the model's litellm support.

        Lives here rather than in `from_name` so every construction path (direct
        instantiation, `from_legacy_params`, deserialization) is covered.
        """
        if unsupported := _unsupported_params(self.name, self.extra_params):
            raise ValueError(
                f"Model {self.name!r} does not support parameter(s)"
                f" {sorted(unsupported)}; the provider rejects them."
            )
        return self

    @classmethod
    def from_name(cls, name: str, **overrides: Any) -> ModelSpec:
        """Build a `ModelSpec` with provider-aware defaults for `extra_params`.

        Applies: Gemini default safety settings and a `temperature` default.
        `logprobs` / `top_logprobs` are gated per-model against litellm's
        capability table, so passing them to a provider that doesn't support
        them (e.g. Anthropic) raises `ValueError`. Explicit values in
        `overrides` always win over the defaults.

        `overrides` may set any `ModelSpec` field, plus request-shape kwargs
        (`temperature`, `max_tokens`, `n`, `logprobs`, `top_logprobs`,
        `safety_settings`, ...) which are merged into `extra_params`.
        """
        extra: dict[str, Any] = {}
        if "gemini" in name:
            extra["safety_settings"] = DEFAULT_VERTEX_SAFETY_SETTINGS
        extra["temperature"] = _DEFAULT_TEMPERATURE

        spec_field_overrides: dict[str, Any] = {}
        extra_overrides: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in cls.model_fields:
                spec_field_overrides[key] = value
            else:
                extra_overrides[key] = value

        spec_field_overrides.setdefault("name", name)
        merged_extra = (
            extra | extra_overrides | spec_field_overrides.pop("extra_params", {})
        )
        return cls(extra_params=merged_extra, **spec_field_overrides)

    @classmethod
    def from_legacy_params(
        cls, params: dict[str, Any], router_kwargs: dict[str, Any]
    ) -> Self:
        """Build a spec from a legacy `litellm_params` entry plus router defaults.

        `params` is one `model_list[*].litellm_params` dict; `router_kwargs` is
        the config's `router_kwargs`. Only forwards `timeout`/`max_retries` when
        present so this class's own field defaults remain the single source of
        truth (no re-injected literals).
        """
        api_key = params.get("api_key")
        spec_kwargs: dict[str, Any] = {
            "name": params["model"],
            "api_base": params.get("api_base"),
            "api_key": SecretStr(api_key) if api_key is not None else None,
            "extra_params": {
                k: v for k, v in params.items() if k not in _RESERVED_LEGACY_PARAMS
            },
        }
        if (timeout := params.get("timeout", router_kwargs.get("timeout"))) is not None:
            spec_kwargs["timeout"] = timeout
        if (
            retries := params.get("max_retries", router_kwargs.get("num_retries"))
        ) is not None:
            spec_kwargs["max_retries"] = retries
        return cls(**spec_kwargs)

    def to_litellm_kwargs(self) -> dict[str, Any]:
        """Flatten into kwargs for `litellm.acompletion` / `litellm.aresponses`."""
        sanitized_extra = {
            k: v
            for k, v in self.extra_params.items()
            if k not in _NON_LITELLM_EXTRA_PARAMS
        }
        out: dict[str, Any] = {"model": self.name} | sanitized_extra
        if self.timeout is not None:
            out["timeout"] = self.timeout
        if self.api_base is not None:
            out["api_base"] = self.api_base
        if self.api_key is not None:
            out["api_key"] = self.api_key.get_secret_value()
        return out


class LLMConfig(BaseModel):
    """Ordered model chain: `models[0]` is primary, `models[1:]` are fallbacks."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    models: list[ModelSpec] = Field(
        min_length=1,
        description=(
            "Ordered list of models. The first entry is the primary; subsequent"
            " entries are tried in order when earlier models fail in ways that"
            " another model might handle (context overflow, content policy,"
            " model-unavailable, or exhausted retries)."
        ),
    )
    response_validator: ResponseValidator | None = Field(
        default=None,
        exclude=True,
        description=(
            "Optional callable invoked on each successful `LLMResult`. Raises"
            " any exception to reject the response; we convert that into"
            " `ResponseValidationError` and let the retry/fallback loop"
            " handle it."
        ),
    )
    tool_parser: ToolParser | None = Field(
        default=None,
        exclude=True,
        description=(
            "Custom parser for converting LLM completions to tool requests."
            " Lifted onto `LiteLLMModel.tool_parser` during model construction;"
            " see that field for the accepted signatures."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, v: Any) -> Any:
        """Accept an `LLMConfig` or any dict shape LMI knows about.

        Runs before field validation so a plain `LLMConfig`-typed field accepts,
        without any `Annotated` wrapper:

        - an `LLMConfig` instance (passes through)
        - a dict with `"models"` — the canonical form, validated as-is
        - a dict with `"model_list"` — the legacy litellm-Router shape; see
          `from_legacy_dict`
        - a dict with `"name"` — a bare model name plus flat request-shape
          kwargs (e.g. `temperature`, `max_tokens`); built via
          `ModelSpec.from_name`
        """
        if isinstance(v, cls):
            return v
        if not isinstance(v, dict):
            raise TypeError(f"Cannot build an LLMConfig from {type(v).__name__}")
        # A `mode="before"` validator must hand back a dict for core validation
        # to build from (returning a constructed `LLMConfig` is rejected), so the
        # non-canonical shapes are normalized to `{"models": [...]}`.
        if "models" in v:
            return v
        if "model_list" in v:
            return {"models": cls._models_from_legacy(v)}
        if "name" in v:
            kwargs = dict(v)
            name = kwargs.pop("name")
            tool_parser = kwargs.pop("tool_parser", None)
            return {
                "models": [ModelSpec.from_name(name, **kwargs)],
                "tool_parser": tool_parser,
            }
        raise ValueError(
            "Can't infer LLMConfig shape from dict; expected 'models',"
            " 'model_list', or 'name' key"
        )

    def with_extra_params(self, **params: Any) -> LLMConfig:
        """Return a copy where every `ModelSpec.extra_params` has `params` merged in.

        Useful for chain-wide request-shape additions like stop sequences: the
        caller doesn't have to rebuild each spec individually, and the original
        `LLMConfig` is left untouched.
        """
        return self.model_copy(
            update={
                "models": [
                    m.model_copy(update={"extra_params": m.extra_params | params})
                    for m in self.models
                ]
            }
        )

    @classmethod
    def from_legacy_dict(cls, legacy: dict[str, Any]) -> LLMConfig:
        """Build an `LLMConfig` from the legacy dict-shaped configuration.

        The legacy shape is `{model_list: [{model_name, litellm_params}, ...],
        fallbacks: [{primary_name: [fallback_names, ...]}, ...], router_kwargs: {...}}`.
        The `fallbacks` list is flattened into the ordering of `models`; any
        entries in `model_list` not reached by the primary's fallback chain are
        appended at the end.
        """
        return cls(models=cls._models_from_legacy(legacy))

    @classmethod
    def _models_from_legacy(cls, legacy: dict[str, Any]) -> list[ModelSpec]:
        """Flatten a legacy dict into the ordered `ModelSpec` chain."""
        model_list = legacy.get("model_list") or []
        if not model_list:
            raise ValueError("Legacy config has empty or missing model_list")

        fallback_map: dict[str, list[str]] = {}
        for entry in legacy.get("fallbacks") or []:
            fallback_map.update(entry)

        params_by_name: dict[str, dict[str, Any]] = {
            m["model_name"]: dict(m.get("litellm_params", {})) for m in model_list
        }

        referenced = set(chain.from_iterable(fallback_map.values()))
        if missing := sorted(referenced - params_by_name.keys()):
            raise ValueError(
                f"Legacy config 'fallbacks' references unknown model name(s)"
                f" {missing}; they are absent from 'model_list'."
            )

        primary_name = model_list[0]["model_name"]
        ordered: list[str] = [primary_name, *fallback_map.get(primary_name, [])]
        for name in params_by_name:
            if name not in ordered:
                ordered.append(name)

        router_kwargs = legacy.get("router_kwargs") or {}
        return [
            ModelSpec.from_legacy_params(params_by_name[name], router_kwargs)
            for name in ordered
        ]


def _unsupported_params(name: str, params: dict[str, Any]) -> set[str]:
    """Return the gated params (e.g. logprobs) the named model doesn't support.

    Consults litellm's per-model capability table, so OpenAI and Gemini both
    accept `logprobs`/`top_logprobs` while Anthropic does not. If litellm can't
    resolve the model, returns an empty set and lets the request itself surface
    any provider rejection downstream.
    """
    gated = {p for p in _PROVIDER_GATED_PARAMS if p in params}
    if not gated:
        return set()
    try:
        supported = set(litellm.get_supported_openai_params(model=name) or [])
    except litellm.BadRequestError:
        return set()
    return gated - supported
