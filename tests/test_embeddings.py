import asyncio
from unittest.mock import MagicMock, patch

import litellm
import numpy as np
import pytest
from litellm.caching import Cache, InMemoryCache
from pytest_subtests import SubTests

from ldp.llms import (
    EmbeddingModel,
    HybridEmbeddingModel,
    LiteEmbeddingModel,
    SparseEmbeddingModel,
)


class TestLiteEmbeddingModel:
    @pytest.mark.asyncio
    async def test_embed_texts(self) -> None:
        texts = ["Hello", "World"]
        batch_size = 1  # NOTE: this affects the mock below
        model = LiteEmbeddingModel(name="stub", batch_size=1)
        with patch(
            "litellm.aembedding",
            autospec=True,
            side_effect=[
                MagicMock(data=[{"embedding": [1.0, 2.0]}]),
                MagicMock(data=[{"embedding": [3.0, 4.0]}]),
            ],
        ) as mock_aembedding:
            embeddings = await model.embed_texts(texts)

        assert np.allclose(embeddings[0], [1.0, 2.0])
        assert np.allclose(embeddings[1], [3.0, 4.0])
        assert mock_aembedding.call_count == len(texts) / batch_size

    @pytest.mark.parametrize(
        ("model_name", "expected_dimensions"),
        [
            ("stub", None),
            ("text-embedding-ada-002", 1536),
            ("text-embedding-3-small", 1536),
        ],
    )
    def test_model_dimension_inference(
        self, model_name: str, expected_dimensions: int | None
    ) -> None:
        assert LiteEmbeddingModel(name=model_name).dimensions == expected_dimensions

    @pytest.mark.asyncio
    async def test_can_change_dimension(self) -> None:
        """We run this one for real, because want to test end to end."""
        stub_texts = ["test1", "test2"]

        model = LiteEmbeddingModel(name="text-embedding-3-small")
        assert model.dimensions == 1536

        model = LiteEmbeddingModel(name="text-embedding-3-small", dimensions=8)
        assert model.dimensions == 8
        etext1, etext2 = await model.embed_texts(stub_texts)
        assert len(etext1) == len(etext2) == 8

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_caching(self) -> None:
        model = LiteEmbeddingModel(
            name="text-embedding-3-small", dimensions=8, embed_kwargs={"caching": True}
        )
        # Make sure there is no existing cache.
        with patch("litellm.cache", None):
            # now create a new cache
            litellm.cache = Cache()
            assert isinstance(litellm.cache.cache, InMemoryCache)
            assert len(litellm.cache.cache.cache_dict) == 0

            _ = await model.embed_texts(["test1"])
            # need to do this to see the data propagated to cache
            await asyncio.sleep(0.0)

            # Check the cache entry was made
            assert len(litellm.cache.cache.cache_dict) == 1


@pytest.mark.asyncio
async def test_sparse_embedding_model(subtests: SubTests):
    with subtests.test("1D sparse"):
        ndim = 1
        expected_output = [[1.0], [1.0]]

        model = SparseEmbeddingModel(dimensions=ndim)
        result = await model.embed_texts(["test1", "test2"])

        assert result == expected_output

    with subtests.test("large sparse"):
        ndim = 1024

        model = SparseEmbeddingModel(dimensions=ndim)
        result = await model.embed_texts(["hello test", "go hello"])

        assert max(result[0]) == max(result[1]) == 0.5

    with subtests.test("default sparse"):
        model = SparseEmbeddingModel()
        result = await model.embed_texts(["test1 hello", "test2 hello"])

        assert pytest.approx(sum(result[0]), abs=1e-6) == pytest.approx(
            sum(result[1]), abs=1e-6
        )


@pytest.mark.asyncio
async def test_hybrid_embedding_model() -> None:
    hybrid_model = HybridEmbeddingModel(
        models=[LiteEmbeddingModel(), SparseEmbeddingModel()]
    )

    # Mock the embedded documents of Lite and Sparse models
    with (
        patch.object(LiteEmbeddingModel, "embed_texts", return_value=[[1.0], [2.0]]),
        patch.object(SparseEmbeddingModel, "embed_texts", return_value=[[3.0], [4.0]]),
    ):
        result = await hybrid_model.embed_texts(["hello", "world"])
    assert result.tolist() == [[1.0, 3.0], [2.0, 4.0]]


@pytest.mark.asyncio
async def test_class_constructor() -> None:
    original_name = "hybrid-text-embedding-3-small"
    model = EmbeddingModel.from_name(original_name)
    assert isinstance(model, HybridEmbeddingModel)
    assert model.name == original_name
    dense_model, sparse_model = model.models
    assert dense_model.name == "text-embedding-3-small"
    assert dense_model.dimensions == 1536
    assert sparse_model.name == "sparse"
    assert sparse_model.dimensions == 256
    assert model.dimensions == 1792
