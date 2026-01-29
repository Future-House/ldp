import asyncio
from abc import ABC, abstractmethod
from collections import Counter
from enum import StrEnum
from itertools import chain
from typing import Any

import litellm
import tiktoken
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lmi.constants import CHARACTERS_PER_TOKEN_ASSUMPTION, MODEL_COST_MAP
from lmi.cost_tracker import track_costs
from lmi.llms import PassThroughRouter
from lmi.rate_limiter import GLOBAL_LIMITER
from lmi.utils import get_litellm_retrying_config, is_encoded_image

URL_ENCODED_IMAGE_TOKEN_ESTIMATE = 85  # tokens


def estimate_tokens(
    document: str
    | list[str]
    | list[litellm.ChatCompletionImageObject]
    | list[litellm.types.llms.vertex_ai.PartType],
) -> float:
    """Estimate token count for rate limiting purposes."""
    if isinstance(document, str):  # Text or a data URL
        return (
            URL_ENCODED_IMAGE_TOKEN_ESTIMATE
            if is_encoded_image(document)
            else len(document) / CHARACTERS_PER_TOKEN_ASSUMPTION
        )
    # For multimodal content, estimate based on text parts and add fixed cost for images
    token_count = 0.0
    for part in document:
        if isinstance(part, str):  # Part of a batch of text or data URLs
            token_count += estimate_tokens(part)
        # Handle different multimodal formats
        elif part.get("type") == "image_url":  # OpenAI format
            token_count += URL_ENCODED_IMAGE_TOKEN_ESTIMATE
        elif (  # Gemini text format -- https://ai.google.dev/api#text-only-prompt
            "text" in part
        ):
            token_count += len(part["text"]) / CHARACTERS_PER_TOKEN_ASSUMPTION  # type: ignore[typeddict-item]
    return token_count


class EmbeddingModes(StrEnum):
    """Enum representing the different modes of an embedding model."""

    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingModel(ABC, BaseModel):
    name: str
    ndim: int | None = None
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional `rate_limit` key, value must be a RateLimitItem or RateLimitItem"
            " string for parsing"
        ),
    )

    def set_mode(self, mode: EmbeddingModes) -> None:
        """Several embedding models have a 'mode' or prompt which affects output."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""

    async def embed_document(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    @staticmethod
    def from_name(embedding: str, **kwargs) -> "EmbeddingModel":
        if embedding.startswith("hybrid"):
            dense_model = LiteLLMEmbeddingModel(name="-".join(embedding.split("-")[1:]))
            return HybridEmbeddingModel(
                name=embedding, models=[dense_model, SparseEmbeddingModel(**kwargs)]
            )
        if embedding == "sparse":
            return SparseEmbeddingModel(**kwargs)
        return LiteLLMEmbeddingModel(name=embedding, **kwargs)

    async def check_rate_limit(self, token_count: float, **kwargs) -> None:
        if "rate_limit" in self.config:
            await GLOBAL_LIMITER.try_acquire(
                ("client", self.name),
                self.config["rate_limit"],
                weight=max(int(token_count), 1),
                **kwargs,
            )


class LiteLLMEmbeddingModel(EmbeddingModel):
    name: str = Field(default="text-embedding-3-small")
    ndim: int | None = Field(
        default=None,
        description=(
            "The length an embedding will have. If left unspecified, we attempt to"
            " infer an un-truncated length via LiteLLM's internal model map. If this"
            " inference fails, the embedding will be un-truncated."
        ),
    )
    embed_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs to pass to litellm.aembedding.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,  # See below field_validator for injection of kwargs
        description=(
            "The optional `rate_limit` key's value must be a RateLimitItem or"
            " RateLimitItem string for parsing. The optional `kwargs` key is keyword"
            " arguments to pass to the litellm.aembedding function. Note that LiteLLM's"
            " Router is not used here."
        ),
    )
    _router: litellm.Router | None = None

    @property
    def router(self) -> litellm.Router:
        if self._router is None:
            router_kwargs: dict = self.config.get("router_kwargs", {})
            if (
                self.config.get("pass_through_router")  # Explicit opt-out of Router
                or "model_list" not in self.config  # Router requires model_list
            ):
                self._router = PassThroughRouter(**router_kwargs)
            else:
                self._router = litellm.Router(
                    model_list=self.config["model_list"], **router_kwargs
                )
        return self._router

    @model_validator(mode="before")
    @classmethod
    def infer_dimensions(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("ndim") is not None:
            return data
        # Let's infer the dimensions
        config: dict[str, dict[str, Any]] = litellm.get_model_cost_map(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/model_prices_and_context_window_backup.json"
        )
        output_vector_size: int | None = config.get(  # noqa: FURB184
            data.get("name", ""), {}
        ).get("output_vector_size")
        if output_vector_size:
            data["ndim"] = output_vector_size
        return data

    @field_validator("config", mode="before")
    @classmethod
    def set_up_default_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        if "kwargs" not in value:
            value["kwargs"] = get_litellm_retrying_config(
                timeout=120,  # 2-min timeout seemed reasonable
            )
        return value

    def _truncate_if_large(self, texts: list[str]) -> list[str]:
        """Truncate texts if they are too large by using litellm cost map."""
        if self.name not in MODEL_COST_MAP:
            return texts
        max_tokens = MODEL_COST_MAP[self.name]["max_input_tokens"]
        # heuristic about ratio of tokens to characters
        conservative_char_token_ratio = 3
        maybe_too_large = max_tokens * conservative_char_token_ratio
        if any(len(t) > maybe_too_large for t in texts if not is_encoded_image(t)):
            try:
                enct = tiktoken.encoding_for_model("cl100k_base")
                enc_batch = enct.encode_ordinary_batch(texts)
                return [enct.decode(t[:max_tokens]) for t in enc_batch]
            except KeyError:
                return [t[: max_tokens * conservative_char_token_ratio] for t in texts]

        return texts

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = self._truncate_if_large(texts)
        batch_size = self.config.get("batch_size", 16)
        N = len(texts)
        embeddings = []
        for i in range(0, N, batch_size):
            batch = texts[i : i + batch_size]
            await self.check_rate_limit(sum(estimate_tokens(t) for t in batch))

            response = await track_costs(self.router.aembedding)(
                model=self.name,
                input=batch,
                dimensions=self.ndim,
                **self.config.get("kwargs", {}),
            )
            embeddings.extend([e["embedding"] for e in response.data])

        return embeddings


class SparseEmbeddingModel(EmbeddingModel):
    """This is a very simple keyword search model - probably best to be mixed with others."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "sparse"
    ndim: int = 256
    enc: tiktoken.Encoding = Field(
        default_factory=lambda: tiktoken.get_encoding("cl100k_base")
    )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        enc_batch = self.enc.encode_ordinary_batch(texts)
        # now get frequency of each token rel to length
        return [
            [token_counts.get(xi, 0) / len(x) for xi in range(self.ndim)]
            for x in enc_batch
            if (token_counts := Counter(xi % self.ndim for xi in x))
        ]


class HybridEmbeddingModel(EmbeddingModel):
    name: str = "hybrid-embed"
    models: list[EmbeddingModel]

    @model_validator(mode="before")
    @classmethod
    def infer_dimensions(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("ndim") is not None:
            raise ValueError(f"Don't specify dimensions to {cls.__name__}.")
        if not data.get("models") or any(m.ndim is None for m in data["models"]):
            return data
        data["ndim"] = sum(m.ndim for m in data["models"])
        return data

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeds = await asyncio.gather(*[
            m.embed_documents(texts) for m in self.models
        ])

        return [
            list(chain.from_iterable(embed_group))
            for embed_group in zip(*all_embeds, strict=True)
        ]

    def set_mode(self, mode: EmbeddingModes) -> None:
        # Set mode for all component models
        for model in self.models:
            model.set_mode(mode)


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """An embedding model using SentenceTransformers."""

    name: str = Field(default="multi-qa-MiniLM-L6-cos-v1")
    config: dict[str, Any] = Field(default_factory=dict)
    _model: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import numpy as np  # noqa: F401
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Please install lmi[local] to use SentenceTransformerEmbeddingModel."
            ) from exc

        self._model = SentenceTransformer(self.name)

    def set_mode(self, mode: EmbeddingModes) -> None:
        # SentenceTransformer does not support different modes.
        pass

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Asynchronously embed a list of documents using SentenceTransformer.

        Args:
            texts: A list of text documents to embed.

        Returns:
            A list of embedding vectors.
        """
        import numpy as np

        # Extract additional configurations if needed
        batch_size = self.config.get("batch_size", 32)
        device = self.config.get("device", "cpu")

        # Update the model's device if necessary
        if device:
            self._model.to(device)

        # Run the synchronous encode method in a thread pool to avoid blocking the event loop.
        embeddings = await asyncio.to_thread(
            lambda: self._model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,  # Disabled progress bar
                batch_size=batch_size,
                device=device,
            ),
        )
        # If embeddings are returned as numpy arrays, convert them to lists.
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        return embeddings


def embedding_model_factory(embedding: str, **kwargs) -> EmbeddingModel:
    """
    Factory function to create an appropriate EmbeddingModel based on the embedding string.

    Supports:
    - SentenceTransformer models prefixed with "st-" (e.g., "st-multi-qa-MiniLM-L6-cos-v1")
    - LiteLLM models (default if no prefix is provided)
    - Hybrid embeddings prefixed with "hybrid-", contains a sparse and a dense model

    Args:
        embedding: The embedding model identifier. Supports prefixes like "st-" for SentenceTransformer
                   and "hybrid-" for combining multiple embedding models.
        **kwargs: Additional keyword arguments for the embedding model.
    """
    embedding = embedding.strip()  # Remove any leading/trailing whitespace

    if embedding.startswith("hybrid-"):
        # Extract the component embedding identifiers after "hybrid-"
        dense_name = embedding[len("hybrid-") :]

        if not dense_name:
            raise ValueError(
                "Hybrid embedding must contain at least one component embedding."
            )

        # Recursively create each component embedding model
        dense_model = embedding_model_factory(dense_name, **kwargs)
        sparse_model = SparseEmbeddingModel(**kwargs)

        return HybridEmbeddingModel(models=[dense_model, sparse_model])

    if embedding.startswith("st-"):
        # Extract the SentenceTransformer model name after "st-"
        model_name = embedding[len("st-") :].strip()
        if not model_name:
            raise ValueError(
                "SentenceTransformer model name must be specified after 'st-'."
            )

        return SentenceTransformerEmbeddingModel(
            name=model_name,
            config=kwargs,
        )

    if embedding.startswith("litellm-"):
        # Extract the LiteLLM model name after "litellm-"
        model_name = embedding[len("litellm-") :].strip()
        if not model_name:
            raise ValueError("model name must be specified after 'litellm-'.")

        return LiteLLMEmbeddingModel(
            name=model_name,
            config=kwargs,
        )

    if embedding == "sparse":
        return SparseEmbeddingModel(**kwargs)

    # Default to LiteLLMEmbeddingModel if no special prefix is found
    return LiteLLMEmbeddingModel(name=embedding, config=kwargs)
