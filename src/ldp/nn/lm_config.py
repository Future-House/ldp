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


class TorchDType(StrEnum):
    bf16 = "bfloat16"
    fp16 = "float16"
    fp32 = "float32"
    auto = "auto"


class LMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        description=(
            "Name of the model to load. Must be available "
            "on the Huggingface Hub or the path to a local directory."
        )
    )
    load_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Passed as Model.from_pretrained(self.model, **load_args)",
    )
    tokenizer_args: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Passed as AutoTokenizer.from_pretrained(self.model, **tokenizer_args)"
        ),
    )
    chat_template: str | None = Field(
        default=None,
        description=(
            "Name of a jinja file defining a chat template. "
            "Leave as None to not use a chat template."
        ),
    )

    device: str | int | None = None
    dtype: TorchDType = Field(
        default=TorchDType.auto,
        description=(
            "Will pass torch_dtype=getattr(torch, self.dtype) if dtype is not auto."
        ),
    )
    pad_token: str | None = Field(
        default=None,
        description=(
            "If set, will override the pad token in the tokenizer. "
            "Must be a valid token already in the vocabulary. Should be primarily "
            "used for tokenizers that do not predefine a pad token; will throw a "
            "warning otherwise."
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

    def get_causal_lm(self) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
        return self.get_model(AutoModelForCausalLM)

    def get_regression_lm(
        self,
    ) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
        # num_labels=1 puts it in regression mode (MSE loss, single output)
        # https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/auto#transformers.AutoModelForSequenceClassification
        return self.get_classification_lm(num_labels=1)

    def get_classification_lm(
        self, num_labels: int
    ) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
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
        assert self._loaded_model_name is not None, (
            "Call resolve_model_location() before get_*() methods."
        )
        model = self._load_pretrained_model(
            self._loaded_model_name, model_cls, **kwargs
        )
        tokenizer = self._load_tokenizer(self._loaded_model_name)

        # Make consistent in case _load_tokenizer changed the pad token
        model.config.pad_token_id = tokenizer.pad_token_id

        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()  # will raise if not supported

        if _compile_enabled and self.compile:
            model = torch.compile(model)

        logger.debug(f"Model:\n{model}")

        return tokenizer, model

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self._load_tokenizer(self._loaded_model_name or self.model)

    @no_type_check
    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, **self.tokenizer_args)
        tokenizer.padding_side = "right"

        if self.pad_token is not None:
            if self.pad_token not in tokenizer.vocab:
                raise ValueError(
                    f"pad_token {self.pad_token!r} not in tokenizer vocabulary"
                )
            if tokenizer.pad_token is not None:
                logger.warning(
                    f"Overriding tokenizer pad token {tokenizer.pad_token!r} with"
                    f" {self.pad_token!r}"
                )
            tokenizer.pad_token = self.pad_token

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

    def resolve_model_location(self, is_main_process: bool = True):
        """This method can be overridden in subclasses to load models from custom storage.

        This class automatically supports models in (a) Huggingface Hub or (b) the local
        filesystem.
        """
        self._loaded_model_name = self.model
