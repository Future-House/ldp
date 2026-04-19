from typing import Any

import pytest
from pydantic import BaseModel, Field, SecretStr, ValidationError

from lmi.config import LLMConfig, LLMConfigField, ModelSpec
from lmi.constants import DEFAULT_VERTEX_SAFETY_SETTINGS
from lmi.llms import CommonLLMNames, LiteLLMModel


class TestModelSpec:
    def test_minimal(self) -> None:
        spec = ModelSpec(name="gpt-4o-mini")
        assert spec.name == "gpt-4o-mini"
        assert spec.timeout == 60.0
        assert spec.max_retries == 3
        assert not spec.extra_params
        assert spec.api_key is None
        assert spec.api_base is None

    def test_to_litellm_kwargs_strips_none_and_unwraps_secret(self) -> None:
        spec = ModelSpec(
            name="gpt-4o-mini",
            api_base="https://example.com",
            api_key=SecretStr("sk-abc"),
            timeout=10.0,
            extra_params={"temperature": 0.3},
        )
        assert spec.to_litellm_kwargs() == {
            "model": "gpt-4o-mini",
            "timeout": 10.0,
            "temperature": 0.3,
            "api_base": "https://example.com",
            "api_key": "sk-abc",  # pragma: allowlist secret
        }

    def test_to_litellm_kwargs_omits_unset_optional_fields(self) -> None:
        kwargs = ModelSpec(name="gpt-4o-mini").to_litellm_kwargs()
        assert "api_key" not in kwargs
        assert "api_base" not in kwargs

    def test_to_litellm_kwargs_strips_litellm_retry_kwargs(self) -> None:
        # `num_retries` / `max_retries` are LiteLLM's internal retry knobs; LMI
        # owns retries, so they must not flow through to litellm.acompletion
        # even when callers stuff them into extra_params via legacy configs.
        spec = ModelSpec(
            name="gpt-4o-mini",
            extra_params={"num_retries": 5, "max_retries": 7, "temperature": 0.3},
        )
        kwargs = spec.to_litellm_kwargs()
        assert "num_retries" not in kwargs
        assert "max_retries" not in kwargs
        assert kwargs["temperature"] == 0.3

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            ModelSpec(name="gpt-4o-mini", unknown_field=1)  # type: ignore[call-arg]

    def test_from_name_openai_defaults(self) -> None:
        spec = ModelSpec.from_name("gpt-4o-mini")
        assert spec.extra_params == {"temperature": 1.0}

    def test_from_name_gemini_adds_safety_settings(self) -> None:
        spec = ModelSpec.from_name("gemini-1.5-pro")
        assert spec.extra_params["safety_settings"] == DEFAULT_VERTEX_SAFETY_SETTINGS

    def test_from_name_rejects_logprobs_for_non_openai(self) -> None:
        with pytest.raises(ValueError, match="only supported on OpenAI"):
            ModelSpec.from_name("claude-3-5-sonnet-20241022", logprobs=True)
        with pytest.raises(ValueError, match="only supported on OpenAI"):
            ModelSpec.from_name("claude-3-5-sonnet-20241022", top_logprobs=5)

    def test_from_name_keeps_logprobs_for_openai(self) -> None:
        spec = ModelSpec.from_name("gpt-4o-mini", logprobs=True, top_logprobs=5)
        assert spec.extra_params["logprobs"] is True
        assert spec.extra_params["top_logprobs"] == 5

    def test_from_name_override_wins_over_default(self) -> None:
        spec = ModelSpec.from_name("gpt-4o-mini", temperature=0.0)
        assert spec.extra_params["temperature"] == 0.0

    def test_from_name_routes_spec_field_overrides(self) -> None:
        spec = ModelSpec.from_name("gpt-4o-mini", timeout=5.0, max_retries=7)
        assert spec.timeout == 5.0
        assert spec.max_retries == 7
        assert "timeout" not in spec.extra_params
        assert "max_retries" not in spec.extra_params


class TestLLMConfigFromLegacy:
    def test_requires_model_list(self) -> None:
        with pytest.raises(ValueError, match="empty or missing model_list"):
            LLMConfig.from_legacy_dict({})

    def test_single_model_primary_only(self) -> None:
        legacy = {
            "model_list": [
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {
                        "model": "gpt-4o-mini",
                        "temperature": 0.5,
                        "max_tokens": 200,
                    },
                }
            ]
        }
        cfg = LLMConfig.from_legacy_dict(legacy)
        assert len(cfg.models) == 1
        assert cfg.models[0].name == "gpt-4o-mini"
        assert cfg.models[0].extra_params == {"temperature": 0.5, "max_tokens": 200}

    def test_fallback_chain_ordered_from_primary(self) -> None:
        legacy = {
            "model_list": [
                {"model_name": "A", "litellm_params": {"model": "gpt-4o-mini"}},
                {
                    "model_name": "B",
                    "litellm_params": {"model": "claude-3-5-sonnet-20241022"},
                },
                {
                    "model_name": "C",
                    "litellm_params": {"model": "gemini-1.5-pro"},
                },
            ],
            "fallbacks": [{"A": ["B", "C"]}],
        }
        cfg = LLMConfig.from_legacy_dict(legacy)
        assert [s.name for s in cfg.models] == [
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
            "gemini-1.5-pro",
        ]

    def test_unreferenced_models_appended(self) -> None:
        legacy = {
            "model_list": [
                {"model_name": "A", "litellm_params": {"model": "gpt-4o-mini"}},
                {
                    "model_name": "B",
                    "litellm_params": {"model": "claude-3-5-sonnet-20241022"},
                },
                {
                    "model_name": "C",
                    "litellm_params": {"model": "gemini-1.5-pro"},
                },
            ],
            "fallbacks": [{"A": ["B"]}],
        }
        cfg = LLMConfig.from_legacy_dict(legacy)
        # A -> B (from fallbacks), then C is orphan and appended at end.
        assert [s.name for s in cfg.models] == [
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
            "gemini-1.5-pro",
        ]

    def test_router_kwargs_feed_timeout_and_retries(self) -> None:
        legacy = {
            "model_list": [
                {"model_name": "A", "litellm_params": {"model": "gpt-4o-mini"}}
            ],
            "router_kwargs": {"timeout": 12.5, "num_retries": 7},
        }
        cfg = LLMConfig.from_legacy_dict(legacy)
        assert cfg.models[0].timeout == 12.5
        assert cfg.models[0].max_retries == 7

    def test_per_model_overrides_beat_router_defaults(self) -> None:
        legacy = {
            "model_list": [
                {
                    "model_name": "A",
                    "litellm_params": {
                        "model": "gpt-4o-mini",
                        "timeout": 3.0,
                        "max_retries": 1,
                    },
                }
            ],
            "router_kwargs": {"timeout": 12.5, "num_retries": 7},
        }
        cfg = LLMConfig.from_legacy_dict(legacy)
        assert cfg.models[0].timeout == 3.0
        assert cfg.models[0].max_retries == 1

    def test_api_key_lifts_to_secret(self) -> None:
        legacy = {
            "model_list": [
                {
                    "model_name": "A",
                    "litellm_params": {
                        "model": "gpt-4o-mini",
                        "api_key": "sk-xyz",  # pragma: allowlist secret
                    },
                }
            ]
        }
        cfg = LLMConfig.from_legacy_dict(legacy)
        assert isinstance(cfg.models[0].api_key, SecretStr)
        assert cfg.models[0].api_key.get_secret_value() == "sk-xyz"
        assert "api_key" not in cfg.models[0].extra_params


class TestLiteLLMModelPopulatesLLMConfig:
    """New and legacy construction paths should propagate to the same `llm_config`."""

    def test_bare_name_produces_single_model_chain(self) -> None:
        model = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value)
        assert model.llm_config is not None
        assert len(model.llm_config.models) == 1
        assert model.llm_config.models[0].name == CommonLLMNames.OPENAI_TEST.value

    def test_legacy_dict_and_explicit_llm_config_agree(self) -> None:
        name = CommonLLMNames.OPENAI_TEST.value
        legacy_model = LiteLLMModel(
            config={
                "model_list": [
                    {
                        "model_name": name,
                        "litellm_params": {
                            "model": name,
                            "temperature": 0.3,
                            "max_tokens": 128,
                        },
                    }
                ]
            }
        )

        explicit_model = LiteLLMModel(
            name=name,
            llm_config=LLMConfig(
                models=[
                    ModelSpec(
                        name=name,
                        extra_params={"temperature": 0.3, "max_tokens": 128},
                    )
                ]
            ),
        )

        assert legacy_model.llm_config is not None
        assert explicit_model.llm_config is not None
        legacy_specs = legacy_model.llm_config.models
        explicit_specs = explicit_model.llm_config.models
        assert len(legacy_specs) == len(explicit_specs) == 1
        assert legacy_specs[0].name == explicit_specs[0].name
        assert legacy_specs[0].extra_params == explicit_specs[0].extra_params

    def test_legacy_fallbacks_become_ordered_chain(self) -> None:
        model = LiteLLMModel(
            config={
                "model_list": [
                    {
                        "model_name": "primary",
                        "litellm_params": {"model": CommonLLMNames.OPENAI_TEST.value},
                    },
                    {
                        "model_name": "backup",
                        "litellm_params": {
                            "model": CommonLLMNames.ANTHROPIC_TEST.value
                        },
                    },
                ],
                "fallbacks": [{"primary": ["backup"]}],
            }
        )
        assert model.llm_config is not None
        assert [s.name for s in model.llm_config.models] == [
            CommonLLMNames.OPENAI_TEST.value,
            CommonLLMNames.ANTHROPIC_TEST.value,
        ]

    def test_config_dict_preserved_alongside_llm_config(self) -> None:
        model = LiteLLMModel(name=CommonLLMNames.OPENAI_TEST.value)
        assert "model_list" in model.config
        assert model.llm_config is not None
        # Both sources of truth should reference the same primary name.
        assert (
            model.config["model_list"][0]["model_name"]
            == model.llm_config.models[0].name
        )

    @pytest.mark.parametrize(
        ("config_overrides", "expected_temp", "expected_max_tokens"),
        [
            pytest.param(
                {"temperature": 0, "max_tokens": 56},
                0,
                56,
                id="explicit_zero_temperature",
            ),
            pytest.param(
                {"temperature": 0.5, "max_tokens": 100},
                0.5,
                100,
                id="nonzero_overrides",
            ),
        ],
    )
    def test_config_overrides_propagate_to_llm_config(
        self,
        config_overrides: dict[str, Any],
        expected_temp: float,
        expected_max_tokens: int,
    ) -> None:
        model = LiteLLMModel(
            name=CommonLLMNames.OPENAI_TEST.value, config=config_overrides
        )
        assert model.llm_config is not None
        extras = model.llm_config.models[0].extra_params
        assert extras["temperature"] == expected_temp
        assert extras["max_tokens"] == expected_max_tokens


class TestLLMConfigCoerce:
    def test_passthrough_llmconfig_instance(self) -> None:
        cfg = LLMConfig(models=[ModelSpec(name="gpt-4o-mini")])
        assert LLMConfig.coerce(cfg) is cfg

    def test_typed_dict(self) -> None:
        cfg = LLMConfig.coerce({"models": [{"name": "gpt-4o-mini"}]})
        assert [m.name for m in cfg.models] == ["gpt-4o-mini"]

    def test_legacy_router_dict(self) -> None:
        cfg = LLMConfig.coerce({
            "model_list": [
                {
                    "model_name": "primary",
                    "litellm_params": {"model": "gpt-4o-mini", "temperature": 0.5},
                },
            ]
        })
        assert cfg.models[0].name == "gpt-4o-mini"
        assert cfg.models[0].extra_params == {"temperature": 0.5}

    def test_name_plus_flat_kwargs(self) -> None:
        cfg = LLMConfig.coerce({
            "name": "gpt-4o-mini",
            "temperature": 0.1,
            "timeout": 30.0,
        })
        spec = cfg.models[0]
        assert spec.name == "gpt-4o-mini"
        assert spec.timeout == 30.0
        assert spec.extra_params["temperature"] == 0.1

    def test_name_plus_flat_does_not_mutate_input(self) -> None:
        src = {"name": "gpt-4o-mini", "temperature": 0.1}
        LLMConfig.coerce(src)
        # coerce() must not mutate the caller's dict.
        assert src == {"name": "gpt-4o-mini", "temperature": 0.1}

    def test_empty_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="Can't infer"):
            LLMConfig.coerce({})

    def test_non_dict_non_llmconfig_raises(self) -> None:
        with pytest.raises(TypeError, match="Cannot build"):
            LLMConfig.coerce(42)


class TestLLMConfigField:
    """Pydantic fields annotated with `LLMConfigField` coerce all supported input shapes."""

    class _Holder(BaseModel):
        cfg: LLMConfigField = Field(
            default_factory=lambda: LLMConfig(models=[ModelSpec(name="gpt-4o-mini")])
        )

    def test_accepts_instance(self) -> None:
        expected = LLMConfig(models=[ModelSpec(name="x")])
        h = self._Holder(cfg=expected)
        # Pydantic copies the instance through BeforeValidator but preserves contents.
        assert [m.name for m in h.cfg.models] == ["x"]

    def test_accepts_typed_dict(self) -> None:
        h = self._Holder(cfg={"models": [{"name": "y"}]})
        assert [m.name for m in h.cfg.models] == ["y"]

    def test_accepts_legacy_dict(self) -> None:
        h = self._Holder(
            cfg={
                "model_list": [
                    {
                        "model_name": "primary",
                        "litellm_params": {"model": "gpt-4o-mini"},
                    }
                ]
            }
        )
        assert h.cfg.models[0].name == "gpt-4o-mini"

    def test_accepts_name_plus_flat(self) -> None:
        h = self._Holder(cfg={"name": "gpt-4o-mini", "temperature": 0.1})
        assert h.cfg.models[0].extra_params["temperature"] == 0.1

    def test_rejects_unrecognized_dict(self) -> None:
        with pytest.raises(ValidationError):
            self._Holder(cfg={"unknown_key": "value"})

    def test_rejects_wrong_type(self) -> None:
        # `TypeError` raised inside `BeforeValidator` propagates unwrapped.
        with pytest.raises(TypeError, match="Cannot build"):
            self._Holder(cfg=42)
