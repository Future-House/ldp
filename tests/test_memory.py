import pytest
from lmi import EmbeddingModel
from pytest_subtests import SubTests

from ldp.graph import Memory
from ldp.graph.memory import UIndexMemoryModel


@pytest.fixture(name="sample_memory")
def fixture_sample_memory() -> Memory:
    return Memory(
        query="sample string representation", output="observation", value=42.0
    )


class TestUIndexMemoryModel:
    def test_initialization_serialization(self, subtests: SubTests) -> None:
        with subtests.test(msg="default-model-specified"):
            model = UIndexMemoryModel()
            assert isinstance(model.embedding_model, EmbeddingModel), (
                "Default embedding model should be set"
            )
            model.model_dump()  # Check we can serialize

        with subtests.test(msg="nondefault-model-specified"):
            model_custom = UIndexMemoryModel(
                embedding_model=EmbeddingModel.from_name("text-embedding-3-small")
            )
            model_custom.model_dump()  # Check we can serialize

    @pytest.mark.asyncio
    async def test_add_then_get_memory(self, sample_memory: Memory) -> None:
        memory_model = UIndexMemoryModel(
            embedding_model=EmbeddingModel.from_name("text-embedding-3-small")
        )
        async with memory_model.safe_access_index() as index:
            assert len(index) == 0, "Should have no memories"
        await memory_model.add_memory(sample_memory)
        async with memory_model.safe_access_index() as index:
            assert len(index) == 1, "Should have one memory"
        assert memory_model.memories[0] == sample_memory
        result = await memory_model.get_memory("sample query", matches=1)
        assert len(result) == 1
        assert result[0] == sample_memory
