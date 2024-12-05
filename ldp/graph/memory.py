import asyncio
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar, cast
from uuid import UUID

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    field_validator,
    model_validator,
)
from usearch.index import Index

from ldp.llms import EmbeddingModel

if TYPE_CHECKING:
    from .common_ops import MemoryOp
    from .op_utils import CallID
    from .ops import Op, OpResult, TOutput


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
    value: float | str = Field(
        description="Measure of the output's quality (e.g. loss)."
    )
    metadata: JsonValue = Field(
        default_factory=dict,  # type: ignore[arg-type] # SEE: https://github.com/pydantic/pydantic/issues/10950
        description=(
            "Optional JSON metadata to store with the memory. An example is storing"
            " information an optimizer can use at training time."
        ),
    )
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

    @classmethod
    def from_ops(
        cls,
        mem_op: "MemoryOp",
        mem_call_id: "CallID",
        output_op: "Op[TOutput]",
        output_call_id: "CallID",
        value: float,
        **kwargs,
    ) -> Self:
        """Create from a MemoryOp, output Op, and their call IDs."""
        query: str = mem_op.ctx.get(mem_call_id, "query")
        memory_input: str | None = mem_op.ctx.get(mem_call_id, "memory_input")
        output_result: OpResult[TOutput] = output_op.ctx.get(output_call_id, "output")
        return cls(
            query=query,
            input=memory_input if memory_input is not None else query,
            output=str(output_result.value),
            value=value,
            run_id=output_call_id.run_id,
            **kwargs,
        )


class MemoryModel(BaseModel, ABC):
    """A storage mechanism for memories that supports retrieval."""

    memories: dict[int, Memory] = Field(default_factory=dict)

    @abstractmethod
    async def add_memory(self, memory: Memory) -> None:
        pass

    DEFAULT_MATCHES: ClassVar[int] = 3

    @abstractmethod
    async def get_memory(
        self, query: str, matches: int = DEFAULT_MATCHES
    ) -> list[Memory]:
        pass

    def __len__(self) -> int:
        return len(self.memories)

    # Lock to make memory additions concurrency safe
    _memories_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)


class ValueMemoryModel(MemoryModel):
    """Memory model backed by the value of memories."""

    epsilon: float = Field(
        default=0.0,
        description="Epsilon value for random memories (e.g. if 0.1, 10% of matches will be randomly selected).",
    )
    sort: bool = Field(default=False, description="Whether to sort memories by value.")

    @staticmethod
    def default_extract_value(value: Memory | str | float) -> float:
        if isinstance(value, Memory):
            return float(value.value)
        return float(value)

    extract_value: Callable[[Memory | str | float], float] = Field(
        default=default_extract_value,
        description=(
            "Function we can use to quantify a memory. The default function simply"
            " assumes float values."
        ),
        exclude=True,
    )

    async def add_memory(self, memory: Memory) -> None:
        async with self._memories_lock:
            self.memories[len(self.memories)] = memory

    async def get_memory(  # TODO: think about the fact this doesn't need to be async
        self,
        query: str,  # TODO: think about the fact this is dead code
        matches: int = MemoryModel.DEFAULT_MATCHES,
    ) -> list[Memory]:
        num_random = int(self.epsilon * matches)
        num_nonrandom = matches - num_random
        to_return: list[Memory] = []
        others: list[Memory] = []
        for m in self.memories.values():
            if len(to_return) < num_nonrandom:
                to_return.append(m)
                continue
            argmin_memory = min(
                range(len(to_return)), key=lambda x: self.extract_value(to_return[x])
            )
            if self.extract_value(to_return[argmin_memory]) < self.extract_value(m):
                others.append(to_return.pop(argmin_memory))
                to_return.append(m)
        if num_random > 0:
            to_return += [
                others[i]
                for i in random.choices(list(range(len(others))), k=num_random)
            ]
        if self.sort:
            to_return.sort(key=self.extract_value)
        return to_return


TIndex = TypeVar("TIndex")


class EmbeddedMemoryModel(MemoryModel, ABC, Generic[TIndex]):
    """Memory storage backed by an embedding-based index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding_model: EmbeddingModel = Field(
        default_factory=lambda: EmbeddingModel.from_name(
            "hybrid-text-embedding-3-small"
        )
    )
    _index: TIndex

    @field_validator("memories")
    @classmethod
    def enforce_empty(cls, v: dict) -> dict:
        if v:
            raise ValueError(
                "Memories must be empty at construction time, so we can add memories to"
                " the index."
            )
        return v

    async def add_memory(self, memory: Memory) -> None:
        key = await self._add_to_index(
            embedding=await self.embedding_model.embed_text(memory.query)
        )
        self.memories[key] = memory

    async def get_memory(
        self, query: str, matches: int = MemoryModel.DEFAULT_MATCHES
    ) -> list[Memory]:
        return await self._search_index(
            embedding=await self.embedding_model.embed_text(query), matches=matches
        )

    @asynccontextmanager
    async def safe_access_index(self) -> AsyncIterator[TIndex]:
        """Get the internal Index under the protection of an internal Lock."""
        # pylint bug, SEE: https://github.com/pylint-dev/pylint/issues/9813
        async with self._memories_lock:  # pylint: disable=not-async-context-manager
            yield self._index

    @abstractmethod
    async def _add_to_index(self, embedding: np.ndarray) -> int:
        """Add an embedding to the internal Index and return its key."""

    @abstractmethod
    async def _search_index(
        self, embedding: np.ndarray, matches: int = MemoryModel.DEFAULT_MATCHES
    ) -> list[Memory]:
        """Search the internal Index, returning a 'matches' amount of Memories."""


class UIndexMemoryModel(EmbeddedMemoryModel[Index]):
    """Embedding-based memory model using a U-Search index."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.embedding_model.dimensions:
            raise TypeError("Specify dimensions to the embedding model.")
        self._index = Index(ndim=self.embedding_model.dimensions)

    async def _add_to_index(self, embedding: np.ndarray) -> int:
        async with self.safe_access_index() as index:
            added_value = cast(
                npt.NDArray[np.int_], index.add(len(self.memories), embedding)
            )
            return added_value.item()

    async def _search_index(
        self, embedding: np.ndarray, matches: int = MemoryModel.DEFAULT_MATCHES
    ) -> list[Memory]:
        async with self.safe_access_index() as index:
            search_matches = index.search(embedding, matches)
        # mypy doesn't respect "old style" __getitem__/__len__ as iterable,
        # so we have this ignore. SEE: https://github.com/python/mypy/issues/9737
        return [self.memories[m.key] for m in search_matches]  # type: ignore[union-attr]
