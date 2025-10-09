import asyncio
from typing import Any, ClassVar
from unittest.mock import Mock, call, patch

import litellm
import pytest
import tiktoken
from litellm.caching import Cache, InMemoryCache
from pytest_subtests import SubTests

from lmi.embeddings import (
    MODEL_COST_MAP,
    EmbeddingModel,
    HybridEmbeddingModel,
    LiteLLMEmbeddingModel,
    SentenceTransformerEmbeddingModel,
    SparseEmbeddingModel,
    embedding_model_factory,
    estimate_tokens,
)
from lmi.utils import VCR_DEFAULT_MATCH_ON, encode_image_as_url


def test_estimate_tokens(subtests: SubTests, png_image: bytes) -> None:
    with subtests.test(msg="text only"):
        text_only = "Hello world"
        text_only_estimated_token_count = estimate_tokens(text_only)
        assert text_only_estimated_token_count == 2.75, (
            "Expected a reasonable token estimate"
        )
        text_only_actual_token_count = len(
            tiktoken.get_encoding("cl100k_base").encode(text_only)
        )
        assert text_only_estimated_token_count == pytest.approx(
            text_only_actual_token_count, abs=1
        ), "Estimation should be within one token of what tiktoken"

    # Test multimodal (text + image)
    with subtests.test(msg="multimodal"):  # Text + image
        multimodal = [
            "What is in this image?",
            encode_image_as_url(image_type="png", image_data=png_image),
        ]
        assert estimate_tokens(multimodal) == 90.5, (
            "Expected a reasonable token estimate"
        )


class TestLiteLLMEmbeddingModel:
    @pytest.fixture
    def embedding_model(self) -> LiteLLMEmbeddingModel:
        return LiteLLMEmbeddingModel()

    @pytest.mark.asyncio
    async def test_embed_documents(self, embedding_model):
        texts = ["short text", "another short text"]
        mock_response = Mock(
            data=[{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]
        )

        with (
            patch(
                "lmi.embeddings.LiteLLMEmbeddingModel._truncate_if_large",
                return_value=texts,
            ),
            patch(
                "lmi.embeddings.LiteLLMEmbeddingModel.check_rate_limit",
                return_value=None,
            ),
            patch("litellm.aembedding", side_effect=[mock_response]),
        ):
            embeddings = await embedding_model.embed_documents(texts)
        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.asyncio
    async def test_embedding_batches(
        self, embedding_model: LiteLLMEmbeddingModel
    ) -> None:
        stub_input = ["my", "name", "is", "neo"]
        with patch.object(
            litellm, "aembedding", autospec=True, side_effect=litellm.aembedding
        ) as mock_aembedding:
            embeddings = await embedding_model.embed_documents(stub_input)
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(stub_input)
        mock_aembedding.assert_awaited_once_with(
            model="text-embedding-3-small", input=stub_input, dimensions=None
        )

        embedding_model.config["batch_size"] = 2
        with patch.object(
            litellm, "aembedding", autospec=True, side_effect=litellm.aembedding
        ) as mock_aembedding:
            embeddings = await embedding_model.embed_documents(stub_input)
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(stub_input)
        mock_aembedding.assert_has_awaits([
            call(model="text-embedding-3-small", input=["my", "name"], dimensions=None),
            call(model="text-embedding-3-small", input=["is", "neo"], dimensions=None),
        ])

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
        assert LiteLLMEmbeddingModel(name=model_name).ndim == expected_dimensions

    @pytest.mark.asyncio
    async def test_can_change_dimension(self) -> None:
        """We run this one for real, because want to test end to end."""
        stub_texts = ["test1", "test2"]

        model = LiteLLMEmbeddingModel(name="text-embedding-3-small")
        assert model.ndim == 1536

        model = LiteLLMEmbeddingModel(name="text-embedding-3-small", ndim=8)
        assert model.ndim == 8
        etext1, etext2 = await model.embed_documents(stub_texts)
        assert len(etext1) == len(etext2) == 8

    def test_truncate_if_large_no_truncation(self, embedding_model):
        texts = ["short text", "another short text"]
        truncated_texts = embedding_model._truncate_if_large(texts)
        assert truncated_texts == texts

    def test_truncate_if_large_with_truncation(self, embedding_model):
        texts = ["a" * 10000, "b" * 10000]

        mock_encoder = Mock()
        mock_encoder.encode_ordinary_batch.return_value = [[1] * 1000 for _ in texts]
        mock_encoder.decode.return_value = "truncated text"

        with (
            patch.dict(
                MODEL_COST_MAP, {embedding_model.name: {"max_input_tokens": 100}}
            ),
            patch("tiktoken.encoding_for_model", return_value=mock_encoder),
        ):
            truncated_texts = embedding_model._truncate_if_large(texts)
        assert truncated_texts == ["truncated text", "truncated text"]

    def test_truncate_if_large_key_error(self, embedding_model):
        texts = ["a" * 10000, "b" * 10000]
        with (
            patch.dict(
                MODEL_COST_MAP, {embedding_model.name: {"max_input_tokens": 100}}
            ),
            patch("tiktoken.encoding_for_model", side_effect=KeyError),
        ):
            truncated_texts = embedding_model._truncate_if_large(texts)
            assert truncated_texts == ["a" * 300, "b" * 300]

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_caching(self) -> None:
        model = LiteLLMEmbeddingModel(
            name="text-embedding-3-small", dimensions=8, embed_kwargs={"caching": True}
        )
        # Make sure there is no existing cache.
        with patch("litellm.cache", None):
            # now create a new cache
            litellm.cache = Cache()
            assert isinstance(litellm.cache.cache, InMemoryCache)
            assert len(litellm.cache.cache.cache_dict) == 0

            _ = await model.embed_documents(["test1"])
            # need to do this to see the data propagated to cache
            await asyncio.sleep(0.0)

            # Check the cache entry was made
            assert len(litellm.cache.cache.cache_dict) == 1

    def test_default_config_injection(self, embedding_model):
        # field_validator is only triggered if the attribute is passed
        embedding_model = LiteLLMEmbeddingModel(config={})

        config = embedding_model.config
        assert "kwargs" in config
        assert config["kwargs"]["timeout"] == 120

    SENTINEL_TIMEOUT: ClassVar[float] = 15.0

    @pytest.mark.parametrize(
        ("config", "used_router"),
        [
            pytest.param(
                {
                    "model_list": [
                        {
                            "model_name": "text-embedding-3-small",
                            "litellm_params": {
                                "model": "text-embedding-3-small",
                                "timeout": SENTINEL_TIMEOUT,
                            },
                        }
                    ],
                    "kwargs": {},  # Ensure we don't set up retrying config overrides
                },
                True,
                id="with-router",
            ),
            pytest.param(
                {
                    "pass_through_router": True,
                    "router_kwargs": {"timeout": SENTINEL_TIMEOUT},
                    "kwargs": {},  # Ensure we don't set up retrying config overrides
                },
                False,
                id="without-router",
            ),
        ],
    )
    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_router_usage(
        self, config: dict[str, Any], used_router: bool
    ) -> None:
        model = LiteLLMEmbeddingModel(
            name="text-embedding-3-small", config=config, ndim=8
        )

        with (
            patch.object(
                litellm.Router,
                "aembedding",
                side_effect=litellm.Router.aembedding,
                autospec=True,
            ) as mock_Router_aembedding,
            patch.object(
                litellm, "aembedding", side_effect=litellm.aembedding, autospec=True
            ) as mock_aembedding,
        ):
            embeddings = await model.embed_documents(["test"])

        # Check embeddings are expected
        (embedding,) = embeddings
        assert len(embedding) == model.ndim

        # Check we acquired the embeddings as expected
        if used_router:
            mock_Router_aembedding.assert_awaited_once_with(
                model.router,
                model="text-embedding-3-small",
                input=["test"],
                dimensions=8,
            )
        else:
            mock_Router_aembedding.assert_not_awaited()
        mock_aembedding.assert_awaited_once()
        # Confirm use of the sentinel timeout in the Router's model_list or pass through
        assert mock_aembedding.call_args.kwargs["timeout"] == self.SENTINEL_TIMEOUT

    @pytest.mark.asyncio
    async def test_multimodal_embedding(
        self, subtests: SubTests, png_image_gcs: str
    ) -> None:
        multimodal_model = LiteLLMEmbeddingModel(
            name=f"{litellm.LlmProviders.VERTEX_AI.value}/multimodalembedding@001"
        )

        with subtests.test(msg="text or image only"):
            embedding_text_only = await multimodal_model.embed_document("Some text")
            assert len(embedding_text_only) == 1408
            assert all(isinstance(x, float) for x in embedding_text_only)

            embedding_image_only = await multimodal_model.embed_document(png_image_gcs)
            assert len(embedding_image_only) == 1408
            assert all(isinstance(x, float) for x in embedding_image_only)

            assert embedding_image_only != embedding_text_only

        with (
            subtests.test(msg="denies two texts"),
            pytest.raises(litellm.BadRequestError, match="one instance"),
        ):
            # This is more of a confirmation/demonstration that Vertex AI denies any
            # embedding request containing >1 text or >1 image
            await multimodal_model.embed_documents(["A", "B"])

        with subtests.test(msg="text and image mixing"):
            (embedding_image_text,) = await multimodal_model.embed_documents([
                "What is in this image?",
                png_image_gcs,
            ])
            assert len(embedding_image_text) == 1408
            assert all(isinstance(x, float) for x in embedding_image_text)

            (embedding_two_images,) = await multimodal_model.embed_documents([
                png_image_gcs,
                png_image_gcs,
            ])
            assert len(embedding_two_images) == 1408
            assert all(isinstance(x, float) for x in embedding_two_images)

            assert embedding_image_text != embedding_two_images

        with subtests.test(msg="batching"):
            multimodal_model.config["batch_size"] = 1
            embeddings = await multimodal_model.embed_documents([
                "Some text",
                png_image_gcs,
            ])
            assert len(embeddings) == 2
            for embedding in embeddings:
                assert len(embedding) == 1408
                assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_sparse_embedding_model(subtests: SubTests):
    with subtests.test("1D sparse"):
        ndim = 1
        expected_output = [[1.0], [1.0]]

        model = SparseEmbeddingModel(ndim=ndim)
        result = await model.embed_documents(["test1", "test2"])

        assert result == expected_output

    with subtests.test("large sparse"):
        ndim = 1024

        model = SparseEmbeddingModel(dimensions=ndim)
        result = await model.embed_documents(["hello test", "go hello"])

        assert max(result[0]) == max(result[1]) == 0.5

    with subtests.test("default sparse"):
        model = SparseEmbeddingModel()
        result = await model.embed_documents(["test1 hello", "test2 hello"])

        assert pytest.approx(sum(result[0]), abs=1e-6) == pytest.approx(
            sum(result[1]), abs=1e-6
        )


@pytest.mark.asyncio
async def test_hybrid_embedding_model() -> None:
    hybrid_model = HybridEmbeddingModel(
        models=[LiteLLMEmbeddingModel(), SparseEmbeddingModel()]
    )

    # Mock the embedded documents of Lite and Sparse models
    with (
        patch.object(
            LiteLLMEmbeddingModel, "embed_documents", return_value=[[1.0], [2.0]]
        ),
        patch.object(
            SparseEmbeddingModel, "embed_documents", return_value=[[3.0], [4.0]]
        ),
    ):
        result = await hybrid_model.embed_documents(["hello", "world"])
    assert result == [[1.0, 3.0], [2.0, 4.0]]


def test_class_constructor() -> None:
    original_name = "hybrid-text-embedding-3-small"
    model = EmbeddingModel.from_name(original_name)
    assert isinstance(model, HybridEmbeddingModel)
    assert model.name == original_name
    dense_model, sparse_model = model.models
    assert dense_model.name == "text-embedding-3-small"
    assert dense_model.ndim == 1536
    assert sparse_model.name == "sparse"
    assert sparse_model.ndim == 256
    assert model.ndim == 1792


@pytest.mark.asyncio
async def test_embedding_model_factory_sentence_transformer() -> None:
    """Test that the factory creates a SentenceTransformerEmbeddingModel when given an 'st-' prefix."""
    embedding = "st-multi-qa-MiniLM-L6-cos-v1"
    model = embedding_model_factory(embedding)
    assert isinstance(model, SentenceTransformerEmbeddingModel), (
        "Factory did not create SentenceTransformerEmbeddingModel"
    )
    assert model.name == "multi-qa-MiniLM-L6-cos-v1", "Incorrect model name assigned"

    # Test embedding functionality
    texts = ["Hello world", "Test sentence"]
    embeddings = await model.embed_documents(texts)
    assert len(embeddings) == 2, "Incorrect number of embeddings returned"
    assert all(isinstance(embed, list) for embed in embeddings), (
        "Embeddings are not in list format"
    )
    assert all(len(embed) > 0 for embed in embeddings), "Embeddings should not be empty"


@pytest.mark.asyncio
async def test_embedding_model_factory_hybrid_with_sentence_transformer() -> None:
    """Test that the factory creates a HybridEmbeddingModel containing a SentenceTransformerEmbeddingModel."""
    embedding = "hybrid-st-multi-qa-MiniLM-L6-cos-v1"
    model = embedding_model_factory(embedding)
    assert isinstance(model, HybridEmbeddingModel), (
        "Factory did not create HybridEmbeddingModel"
    )
    assert len(model.models) == 2, "Hybrid model should contain two component models"
    assert isinstance(model.models[0], SentenceTransformerEmbeddingModel), (
        "First component should be SentenceTransformerEmbeddingModel"
    )
    assert isinstance(model.models[1], SparseEmbeddingModel), (
        "Second component should be SparseEmbeddingModel"
    )

    # Test embedding functionality
    texts = ["Hello world", "Test sentence"]
    embeddings = await model.embed_documents(texts)
    assert len(embeddings) == 2, "Incorrect number of embeddings returned"
    expected_length = len((await model.models[0].embed_documents(texts))[0]) + len(
        (await model.models[1].embed_documents(texts))[0]
    )
    assert all(len(embed) == expected_length for embed in embeddings), (
        "Embeddings do not match expected combined length"
    )


def test_embedding_model_factory_invalid_st_prefix() -> None:
    """Test that the factory raises a ValueError when 'st-' prefix is provided without a model name."""
    embedding = "st-"
    with pytest.raises(
        ValueError,
        match=r"SentenceTransformer model name must be specified after 'st-'.",
    ):
        embedding_model_factory(embedding)


def test_embedding_model_factory_unknown_prefix() -> None:
    """Test that the factory defaults to LiteLLMEmbeddingModel when an unknown prefix is provided."""
    embedding = "unknown-prefix-model"
    model = embedding_model_factory(embedding)
    assert isinstance(model, LiteLLMEmbeddingModel), (
        "Factory did not default to LiteLLMEmbeddingModel for unknown prefix"
    )
    assert model.name == "unknown-prefix-model", "Incorrect model name assigned"


def test_embedding_model_factory_sparse() -> None:
    """Test that the factory creates a SparseEmbeddingModel when 'sparse' is provided."""
    embedding = "sparse"
    model = embedding_model_factory(embedding)
    assert isinstance(model, SparseEmbeddingModel), (
        "Factory did not create SparseEmbeddingModel"
    )
    assert model.name == "sparse", "Incorrect model name assigned"


def test_embedding_model_factory_litellm() -> None:
    """Test that the factory creates a LiteLLMEmbeddingModel when 'litellm-' prefix is provided."""
    embedding = "litellm-text-embedding-3-small"
    model = embedding_model_factory(embedding)
    assert isinstance(model, LiteLLMEmbeddingModel), (
        "Factory did not create LiteLLMEmbeddingModel"
    )
    assert model.name == "text-embedding-3-small", "Incorrect model name assigned"


def test_embedding_model_factory_default() -> None:
    """Test that the factory defaults to LiteLLMEmbeddingModel when no known prefix is provided."""
    embedding = "default-model"
    model = embedding_model_factory(embedding)
    assert isinstance(model, LiteLLMEmbeddingModel), (
        "Factory did not default to LiteLLMEmbeddingModel"
    )
    assert model.name == "default-model", "Incorrect model name assigned"
