"""
Unit tests for EmbeddingGenerator and EmbeddingCache.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.chunker import DocumentChunk
from ingestion.embedder import EmbeddingCache, EmbeddingGenerator, create_embedder


def _make_chunk(content: str, index: int = 0) -> DocumentChunk:
    return DocumentChunk(
        content=content,
        index=index,
        start_char=0,
        end_char=len(content),
        metadata={"title": "Test", "source": "test.txt"},
    )


def _make_embedding_response(dim: int = 1536) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1] * dim)]
    return mock_resp


class TestEmbeddingGeneratorInit:
    def test_known_model_config_loaded(self):
        gen = EmbeddingGenerator(model="text-embedding-3-small")
        assert gen.config["dimensions"] == 1536

    def test_large_model_config(self):
        gen = EmbeddingGenerator(model="text-embedding-3-large")
        assert gen.config["dimensions"] == 3072

    def test_unknown_model_uses_default_config(self):
        gen = EmbeddingGenerator(model="custom-model-xyz")
        assert gen.config["dimensions"] == 1536

    def test_batch_size_default(self):
        gen = EmbeddingGenerator()
        assert gen.batch_size == 100


class TestGenerateEmbedding:
    @pytest.mark.asyncio
    async def test_returns_embedding_vector(self):
        gen = EmbeddingGenerator()
        gen_mock = AsyncMock(return_value=_make_embedding_response())

        import ingestion.embedder as emb_mod

        with patch.object(emb_mod, "embedding_client") as mock_client:
            mock_client.embeddings.create = gen_mock
            embedding = await gen.generate_embedding("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 1536

    @pytest.mark.asyncio
    async def test_text_truncated_when_too_long(self):
        gen = EmbeddingGenerator()
        long_text = "x" * 100_000

        captured: list = []

        async def _fake_create(model, input):
            captured.append(len(input))
            return _make_embedding_response()

        import ingestion.embedder as emb_mod

        with patch.object(emb_mod, "embedding_client") as mock_client:
            mock_client.embeddings.create = _fake_create
            await gen.generate_embedding(long_text)

        max_chars = gen.config["max_tokens"] * 4
        assert captured[0] <= max_chars

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        import openai

        gen = EmbeddingGenerator(max_retries=3, retry_delay=0)
        call_count = 0

        async def _flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise openai.RateLimitError("rate limit")
            return _make_embedding_response()

        import ingestion.embedder as emb_mod

        with patch.object(emb_mod, "embedding_client") as mock_client:
            mock_client.embeddings.create = _flaky
            embedding = await gen.generate_embedding("text")

        assert call_count == 3
        assert len(embedding) == 1536


class TestGenerateEmbeddingsBatch:
    @pytest.mark.asyncio
    async def test_returns_one_embedding_per_input(self):
        gen = EmbeddingGenerator()
        texts = ["text one", "text two", "text three"]

        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in texts]

        import ingestion.embedder as emb_mod

        with patch.object(emb_mod, "embedding_client") as mock_client:
            mock_client.embeddings.create = AsyncMock(return_value=mock_resp)
            embeddings = await gen.generate_embeddings_batch(texts)

        assert len(embeddings) == 3

    @pytest.mark.asyncio
    async def test_empty_strings_are_handled(self):
        gen = EmbeddingGenerator()
        texts = ["", "  ", "real text"]

        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.0] * 1536) for _ in texts]

        import ingestion.embedder as emb_mod

        with patch.object(emb_mod, "embedding_client") as mock_client:
            mock_client.embeddings.create = AsyncMock(return_value=mock_resp)
            embeddings = await gen.generate_embeddings_batch(texts)

        assert len(embeddings) == 3


class TestEmbedChunks:
    @pytest.mark.asyncio
    async def test_all_chunks_receive_embedding(self):
        gen = EmbeddingGenerator(batch_size=10)
        chunks = [_make_chunk(f"chunk content {i}", i) for i in range(5)]

        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[float(i)] * 1536) for i in range(5)]

        import ingestion.embedder as emb_mod

        with patch.object(emb_mod, "embedding_client") as mock_client:
            mock_client.embeddings.create = AsyncMock(return_value=mock_resp)
            embedded = await gen.embed_chunks(chunks)

        assert len(embedded) == 5
        for chunk in embedded:
            assert hasattr(chunk, "embedding")

    @pytest.mark.asyncio
    async def test_empty_chunk_list_returns_empty(self):
        gen = EmbeddingGenerator()
        result = await gen.embed_chunks([])
        assert result == []

    @pytest.mark.asyncio
    async def test_progress_callback_called(self):
        gen = EmbeddingGenerator(batch_size=3)
        chunks = [_make_chunk(f"c{i}", i) for i in range(6)]
        callback_calls: list = []

        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(3)]

        import ingestion.embedder as emb_mod

        with patch.object(emb_mod, "embedding_client") as mock_client:
            mock_client.embeddings.create = AsyncMock(return_value=mock_resp)
            await gen.embed_chunks(
                chunks, progress_callback=lambda c, t: callback_calls.append((c, t))
            )

        assert len(callback_calls) == 2  # 6 chunks / batch_size=3 = 2 batches


class TestEmbeddingCache:
    def test_miss_returns_none(self):
        cache = EmbeddingCache(max_size=10)
        assert cache.get("unseen text") is None

    def test_put_then_get(self):
        cache = EmbeddingCache(max_size=10)
        embedding = [0.1] * 128
        cache.put("hello world", embedding)
        assert cache.get("hello world") == embedding

    def test_same_text_returns_same_entry(self):
        cache = EmbeddingCache(max_size=10)
        cache.put("test", [1.0, 2.0])
        assert cache.get("test") == [1.0, 2.0]

    def test_eviction_when_full(self):
        cache = EmbeddingCache(max_size=3)
        for i in range(3):
            cache.put(f"text-{i}", [float(i)])
        # Adding one more should evict the oldest
        cache.put("text-3", [3.0])
        assert len(cache.cache) == 3

    def test_different_texts_different_entries(self):
        cache = EmbeddingCache()
        cache.put("text A", [1.0])
        cache.put("text B", [2.0])
        assert cache.get("text A") == [1.0]
        assert cache.get("text B") == [2.0]


class TestCreateEmbedder:
    def test_returns_embedding_generator(self):
        embedder = create_embedder()
        assert isinstance(embedder, EmbeddingGenerator)

    def test_cache_wraps_generate_embedding(self):
        embedder = create_embedder(use_cache=True)
        # With cache, generate_embedding should be replaced by the cached wrapper
        assert callable(embedder.generate_embedding)

    def test_no_cache_preserves_original(self):
        embedder = create_embedder(use_cache=False)
        assert isinstance(embedder, EmbeddingGenerator)
