from enum import StrEnum
from logging import getLogger
from typing import Any, Self, TypeVar, no_type_check

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = getLogger(__name__)
TModel = TypeVar("TModel", bound=PreTrainedModel)


class ModelMode(StrEnum):
    training = "training"
    inference = "inference"


class TorchDType(StrEnum):
    bf16 = "bfloat16"
    fp16 = "float16"
    fp32 = "float32"
    auto = "auto"


class LMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        description="Name of the model to load. Must be available "
        "on the Huggingface Hub or the path to a local directory."
    )
    load_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Passed as Model.from_pretrained(self.model, **load_args)",
    )
    tokenizer_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Passed as AutoTokenizer.from_pretrained(self.model, **tokenizer_args)",
    )

    device: str | int | None = None
    dtype: TorchDType = Field(
        default=TorchDType.auto,
        description=(
            "Will pass torch_dtype=getattr(torch, self.dtype) if dtype is not auto."
        ),
    )
    mode: ModelMode | None = None
    eos_as_pad_fallback: bool = Field(
        default=True,
        description=(
            "If the tokenizer is missing a pad token, "
            "automatically set it to the EOS token. Defaults to True."
        ),
    )
    gradient_checkpointing: bool = False
    compile: bool = False

    # private attribute
    _loaded_model_name: str | None = None

    @model_validator(mode="after")
    def check_load_args(self) -> Self:
        if (
            self.device != "cpu"
            and torch.cuda.is_available()
            and self.dtype in {TorchDType.bf16, TorchDType.fp16}
        ):
            # FA2 is a good default if provided
            self.load_args.setdefault("attn_implementation", "flash_attention_2")
        if "torch_dtype" in self.load_args:
            raise ValueError("Do not set torch_dtype in load_args. Use dtype instead.")
        return self

    def get_causal_lm(self) -> tuple[PreTrainedTokenizer, AutoModelForCausalLM]:
        return self.get_model(AutoModelForCausalLM)

    def get_regression_lm(
        self,
    ) -> tuple[PreTrainedTokenizer, AutoModelForSequenceClassification]:
        # num_labels=1 puts it in regression mode (MSE loss, single output)
        # https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/auto#transformers.AutoModelForSequenceClassification
        return self.get_classification_lm(num_labels=1)

    def get_classification_lm(
        self, num_labels: int
    ) -> tuple[PreTrainedTokenizer, AutoModelForSequenceClassification]:
        return self.get_model(AutoModelForSequenceClassification, num_labels=num_labels)

    # huggingface autotypes make type annotations messy, so disable mypy
    # for this function
    @no_type_check
    def get_model(
        self,
        model_cls: type[TModel],
        _compile_enabled: bool = True,
        **kwargs,
    ) -> tuple[PreTrainedTokenizer, TModel]:
        model = self._load_pretrained_model(self.model, model_cls, **kwargs)
        tokenizer = self.get_tokenizer()

        # Make consistent in case _load_tokenizer changed the pad token
        model.config.pad_token_id = tokenizer.pad_token_id

        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()  # will raise if not supported

        if _compile_enabled and self.compile:
            model = torch.compile(model)

        logger.debug(f"Model:\n{model}")

        return tokenizer, model

    @no_type_check
    def get_tokenizer(self) -> PreTrainedTokenizer:
        model_name = self.model
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **self.tokenizer_args)
        tokenizer.padding_side = "right"

        if (not tokenizer.pad_token) and self.eos_as_pad_fallback:
            logger.warning("Tokenizer does not have a pad token. Using EOS token.")
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    @no_type_check
    def _load_pretrained_model(
        self, model_name: str, model_cls: type[TModel], **kwargs
    ) -> TModel:
        kwargs.update(self.load_args)
        if "quantization_config" in kwargs:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                **kwargs["quantization_config"]
            )
        if self.dtype != TorchDType.auto:
            kwargs["torch_dtype"] = getattr(torch, self.dtype.value)

        if self.device is not None:
            device_map = self.device
            logger.info(f"Loading model from {model_name} to {self.device}")
        else:
            device_map = None
            logger.info(f"Loading model from {model_name}")
        model = model_cls.from_pretrained(
            model_name,
            device_map=device_map,
            **kwargs,
        )

        if "torch_dtype" in kwargs:
            # ValueHead models put the value head in fp32, so homogenize here
            model = model.to(kwargs["torch_dtype"])

        return model
