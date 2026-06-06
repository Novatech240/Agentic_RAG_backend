"""
Unit tests for the HybridRetriever (Vespa-first, Postgres fallback).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.retriever import HybridRetriever


def _vespa_row(chunk_id: str, cosine: float = 0.7) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "content": f"Content of {chunk_id}",
        "similarity": cosine,
        "cosine_similarity": cosine,
        "combined_score": 0.5,
        "bm25_score": 1.2,
        "document_title": "Test Doc",
        "document_source": "s3://bucket/test.pdf",
        "metadata": {},
    }


def _pg_row(chunk_id: str, similarity: float = 0.6) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc-1",
        "content": f"PG content of {chunk_id}",
        "similarity": similarity,
        "combined_score": similarity,
        "document_title": "Test Doc",
        "document_source": "s3://bucket/test.pdf",
        "metadata": {},
    }


class TestNoOpIndexing:
    """build_index / rebuild_index are retained as harmless no-ops."""

    @pytest.mark.asyncio
    async def test_build_index_noop(self):
        r = HybridRetriever()
        assert await r.build_index() is None

    @pytest.mark.asyncio
    async def test_rebuild_index_noop(self):
        r = HybridRetriever()
        assert await r.rebuild_index() is None


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_uses_vespa_when_enabled(self):
        r = HybridRetriever()
        with (
            patch("agent.vespa_client.vespa_enabled", return_value=True),
            patch(
                "agent.vespa_client.search",
                AsyncMock(return_value=[_vespa_row("c1"), _vespa_row("c2")]),
            ) as mock_search,
            patch("agent.db_utils.hybrid_search", AsyncMock()) as mock_pg,
        ):
            result = await r.retrieve(query="AI", embedding=[0.1] * 3072, limit=5)

        mock_search.assert_awaited_once()
        mock_pg.assert_not_awaited()
        assert [x["chunk_id"] for x in result] == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_falls_back_to_postgres_on_vespa_error(self):
        r = HybridRetriever()
        with (
            patch("agent.vespa_client.vespa_enabled", return_value=True),
            patch(
                "agent.vespa_client.search",
                AsyncMock(side_effect=RuntimeError("Vespa down")),
            ),
            patch(
                "agent.db_utils.hybrid_search",
                AsyncMock(return_value=[_pg_row("c1")]),
            ) as mock_pg,
        ):
            result = await r.retrieve(query="AI", embedding=[0.1] * 3072, limit=5)

        mock_pg.assert_awaited_once()
        assert result[0]["chunk_id"] == "c1"
        # fallback must expose cosine_similarity for the confidence gate
        assert result[0]["cosine_similarity"] == 0.6

    @pytest.mark.asyncio
    async def test_falls_back_when_vespa_returns_empty(self):
        r = HybridRetriever()
        with (
            patch("agent.vespa_client.vespa_enabled", return_value=True),
            patch("agent.vespa_client.search", AsyncMock(return_value=[])),
            patch(
                "agent.db_utils.hybrid_search",
                AsyncMock(return_value=[_pg_row("c9")]),
            ) as mock_pg,
        ):
            result = await r.retrieve(query="AI", embedding=[0.1] * 3072)

        mock_pg.assert_awaited_once()
        assert result[0]["chunk_id"] == "c9"

    @pytest.mark.asyncio
    async def test_uses_postgres_when_vespa_disabled(self):
        r = HybridRetriever()
        with (
            patch("agent.vespa_client.vespa_enabled", return_value=False),
            patch("agent.vespa_client.search", AsyncMock()) as mock_search,
            patch(
                "agent.db_utils.hybrid_search",
                AsyncMock(return_value=[_pg_row("c1")]),
            ) as mock_pg,
        ):
            result = await r.retrieve(query="AI", embedding=[0.1] * 3072)

        mock_search.assert_not_awaited()
        mock_pg.assert_awaited_once()
        assert result[0]["chunk_id"] == "c1"
