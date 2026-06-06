"""
Unit tests for the cross-encoder rerank stage (agent.reranker).

The reranker is fail-open: without a key/flag it must degrade to a no-op
truncation, and any provider error must fall back to the retriever's order.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent import reranker
from agent.models import ChunkResult


def _chunk(cid: str, content: str, cos: float = 0.5) -> ChunkResult:
    return ChunkResult(
        chunk_id=cid,
        document_id="doc-1",
        content=content,
        score=0.1,
        vector_similarity=cos,
        metadata={},
        document_title="Doc",
        document_source="s3://b/d.pdf",
    )


class TestRerankEnabled:
    def test_disabled_without_key(self, monkeypatch):
        monkeypatch.setenv("RERANK_ENABLED", "true")
        monkeypatch.setenv("RERANK_PROVIDER", "voyage")
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        assert reranker.rerank_enabled() is False

    def test_disabled_by_flag_even_with_key(self, monkeypatch):
        monkeypatch.setenv("RERANK_ENABLED", "false")
        monkeypatch.setenv("VOYAGE_API_KEY", "k")
        assert reranker.rerank_enabled() is False

    def test_enabled_with_flag_and_key(self, monkeypatch):
        monkeypatch.setenv("RERANK_ENABLED", "true")
        monkeypatch.setenv("RERANK_PROVIDER", "voyage")
        monkeypatch.setenv("VOYAGE_API_KEY", "k")
        assert reranker.rerank_enabled() is True


class TestCandidatePool:
    def test_widens_when_enabled(self, monkeypatch):
        monkeypatch.setattr(reranker, "RERANK_CANDIDATES", 50)
        with patch("agent.reranker.rerank_enabled", return_value=True):
            assert reranker.candidate_pool_size(8) == 50

    def test_no_widen_when_disabled(self):
        with patch("agent.reranker.rerank_enabled", return_value=False):
            assert reranker.candidate_pool_size(8) == 8


class TestRerank:
    @pytest.mark.asyncio
    async def test_noop_when_disabled_truncates(self):
        chunks = [_chunk(f"c{i}", f"t{i}") for i in range(5)]
        with patch("agent.reranker.rerank_enabled", return_value=False):
            out = await reranker.rerank("q", chunks, top_n=3)
        assert [c.chunk_id for c in out] == ["c0", "c1", "c2"]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        assert await reranker.rerank("q", [], top_n=3) == []

    @pytest.mark.asyncio
    async def test_reorders_by_relevance_and_sets_score(self, monkeypatch):
        chunks = [_chunk("c0", "a"), _chunk("c1", "b"), _chunk("c2", "c")]
        # Voyage returns index 2 best, then 0 — relevance scores reused as score.
        with (
            patch("agent.reranker.rerank_enabled", return_value=True),
            patch(
                "agent.reranker._voyage_rerank",
                AsyncMock(return_value=[(2, 0.91), (0, 0.42)]),
            ),
        ):
            out = await reranker.rerank("q", chunks, top_n=2)
        assert [c.chunk_id for c in out] == ["c2", "c0"]
        assert out[0].score == pytest.approx(0.91)
        # vector_similarity must be untouched (the cosine gate depends on it)
        assert out[0].vector_similarity == 0.5

    @pytest.mark.asyncio
    async def test_fail_open_on_provider_error(self):
        chunks = [_chunk("c0", "a"), _chunk("c1", "b")]
        with (
            patch("agent.reranker.rerank_enabled", return_value=True),
            patch(
                "agent.reranker._voyage_rerank",
                AsyncMock(side_effect=RuntimeError("api down")),
            ),
        ):
            out = await reranker.rerank("q", chunks, top_n=2)
        assert [c.chunk_id for c in out] == ["c0", "c1"]
