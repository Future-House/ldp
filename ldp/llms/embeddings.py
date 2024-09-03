import asyncio
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import litellm
import numpy as np
import tiktoken
from pydantic import BaseModel, ConfigDict, Field, model_validator


class EmbeddingModes(StrEnum):
    """Enum representing the different modes of an embedding model."""

    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingModel(ABC, BaseModel):
    name: str
    dimensions: int | None = None

    def set_mode(self, mode: EmbeddingModes) -> None:
        """Several embedding models have a 'mode' or prompt which affects output."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        pass

    async def embed_text(self, text: str) -> np.ndarray:
        return (await self.embed_texts([text]))[0]

    @staticmethod
    def from_name(embedding: str, **kwargs) -> "EmbeddingModel":
        if embedding.startswith("hybrid"):
            dense_model = LiteEmbeddingModel(name="-".join(embedding.split("-")[1:]))
            return HybridEmbeddingModel(
                name=embedding, models=[dense_model, SparseEmbeddingModel(**kwargs)]
            )
        if embedding == "sparse":
            return SparseEmbeddingModel(**kwargs)
        return LiteEmbeddingModel(name=embedding, **kwargs)


class LiteEmbeddingModel(EmbeddingModel):
    name: str = Field(default="text-embedding-3-small")
    dimensions: int | None = Field(
        default=None,
        description=(
            "The length an embedding will have. If left unspecified, we attempt to"
            " infer an un-truncated length via LiteLLM's internal model map. If this"
            " inference fails, the embedding will be un-truncated."
        ),
    )
    batch_size: int = 16
    embed_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs to pass to litellm.aembedding.",
    )

    @model_validator(mode="before")
    @classmethod
    def infer_dimensions(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("dimensions") is not None:
            return data
        # Let's infer the dimensions
        config: dict[str, dict[str, Any]] = litellm.get_model_cost_map(
            url="https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/model_prices_and_context_window_backup.json"
        )
        output_vector_size: int | None = config.get(data.get("name", ""), {}).get(  # noqa: FURB184
            "output_vector_size"
        )
        if output_vector_size:
            data["dimensions"] = output_vector_size
        return data

    async def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
        embeddings = []
        # Before you get excited to asyncio.gather this:
        # The point of this is to not hit the API rate limit
        for i in range(0, len(texts), self.batch_size):
            response = await litellm.aembedding(
                model=self.name,
                input=texts[i : i + self.batch_size],
                encoding_format="float",
                dimensions=self.dimensions,
                **self.embed_kwargs,
            )
            embeddings.extend([
                np.array(e["embedding"], dtype=np.float32) for e in response.data
            ])
        return embeddings


class SparseEmbeddingModel(EmbeddingModel):
    """This is a very simple keyword search model - probably best to be mixed with others."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "sparse"
    dimensions: int = 256
    enc: tiktoken.Encoding = Field(
        default_factory=lambda: tiktoken.get_encoding("cl100k_base")
    )

    async def embed_texts(self, texts) -> list[np.ndarray]:
        enc_batch = self.enc.encode_ordinary_batch(texts)
        # now get frequency of each token rel to length
        return [
            np.bincount(
                [xi % self.dimensions for xi in x], minlength=self.dimensions
            ).astype(np.float32)
            / len(x)
            for x in enc_batch
        ]


class HybridEmbeddingModel(EmbeddingModel):
    name: str = "hybrid-embed"
    models: list[EmbeddingModel]

    @model_validator(mode="before")
    @classmethod
    def infer_dimensions(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("dimensions") is not None:
            raise ValueError(f"Don't specify dimensions to {cls.__name__}.")
        if not data.get("models") or any(m.dimensions is None for m in data["models"]):
            return data
        data["dimensions"] = sum(m.dimensions for m in data["models"])
        return data

    async def embed_texts(self, texts):
        all_embeds = await asyncio.gather(*[m.embed_texts(texts) for m in self.models])
        return np.concatenate(all_embeds, axis=1)
