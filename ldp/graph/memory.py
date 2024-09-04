import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from usearch.index import Index

from ldp.llms import EmbeddingModel


class Memory(BaseModel):
    """A single memory about an input, output, and value tuple.

    A memory is a record of an input, output, and resulting value. Typically used
    for prompting a language model. Or, it could be about a whole forward pass where
    the input is the observation and the output is the action taken.
    The query is optionally different and used for
    retrieving the memory. For example, it could be much larger because it won't
    be formatted in the resulting prompt.
    """

    query: str = Field(
        description="String to be embedded into a retrieval key for a memory index."
    )
    input: str | None = Field(
        default=None,
        description=(
            "Some input (e.g. prompt to LLM, observation). If None (default), the input"
            " is set to match the query."
        ),
    )
    output: str = Field(description="Some output (e.g. tool selection).")
    value: float = Field(description="Measure of the output's quality (e.g. loss).")
    run_id: UUID | None = Field(
        default=None,
        description=(
            "Associate run_id for debugging purposes to trace "
            "which forward pass generated the memory."
        ),
    )
    template: str = "Input: {input}\nOutput: {output}\nValue: {value}"

    @model_validator(mode="before")
    @classmethod
    def ensure_query(cls, data: Any) -> Any:
        """Copy input to match the query if input is None."""
        if isinstance(data, dict) and data.get("input") is None:
            data["input"] = data["query"]
        return data

    def __str__(self) -> str:
        return self.template.format(**self.model_dump())


TIndex = TypeVar("TIndex")


class MemoryModel(BaseModel, Generic[TIndex], ABC):
    """A collection of memories with retrieval."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding_model: EmbeddingModel = Field(
        default_factory=lambda: EmbeddingModel.from_name(
            "hybrid-text-embedding-3-small"
        )
    )
    memories: dict[int, Memory] = Field(default_factory=dict)
    _index: TIndex
    _index_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    @field_validator("memories")
    @classmethod
    def enforce_empty(cls, v: dict) -> dict:
        if v:
            raise ValueError("Memories must be empty at construction time.")
        return v

    async def add_memory(self, memory: Memory) -> None:
        key = await self._add_to_index(
            embedding=await self.embedding_model.embed_text(memory.query)
        )
        self.memories[key] = memory

    DEFAULT_MEMORY_MATCHES: ClassVar[int] = 3

    async def get_memory(
        self, query: str, matches: int = DEFAULT_MEMORY_MATCHES
    ) -> list[Memory]:
        return await self._search_index(
            embedding=await self.embedding_model.embed_text(query), matches=matches
        )

    def __len__(self) -> int:
        return len(self.memories)

    @asynccontextmanager
    async def safe_access_index(self) -> AsyncIterator[TIndex]:
        """Get the internal Index under the protection of an internal Lock."""
        # pylint bug, SEE: https://github.com/pylint-dev/pylint/issues/9813
        async with self._index_lock:  # pylint: disable=not-async-context-manager
            yield self._index

    @abstractmethod
    async def _add_to_index(self, embedding: np.ndarray) -> int:
        """Add an embedding to the internal Index and return its key."""

    @abstractmethod
    async def _search_index(
        self, embedding: np.ndarray, matches: int = DEFAULT_MEMORY_MATCHES
    ) -> list[Memory]:
        """Search the internal Index, returning a 'matches' amount of Memories."""


class UIndexMemoryModel(MemoryModel[Index]):
    """Memory model using a U-Search index."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.embedding_model.dimensions:
            raise TypeError("Specify dimensions to the embedding model.")
        self._index = Index(ndim=self.embedding_model.dimensions)

    async def _add_to_index(self, embedding: np.ndarray) -> int:
        async with self.safe_access_index() as index:
            return int(index.add(len(self.memories), embedding))

    async def _search_index(
        self, embedding: np.ndarray, matches: int = MemoryModel.DEFAULT_MEMORY_MATCHES
    ) -> list[Memory]:
        async with self.safe_access_index() as index:
            search_matches = index.search(embedding, matches)
        # mypy doesn't respect "old style" __getitem__/__len__ as iterable,
        # so we have this ignore. SEE: https://github.com/python/mypy/issues/9737
        return [self.memories[m.key] for m in search_matches]  # type: ignore[union-attr]