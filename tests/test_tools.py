"""
Unit tests for agent search tools.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agent.models import ChunkResult, DocumentMetadata, GraphSearchResult
from agent.tools import (
    DocumentListInput,
    GraphSearchInput,
    HybridSearchInput,
    VectorSearchInput,
    CONFIDENCE_MIN_SIMILARITY,
    graph_search_tool,
    hybrid_search_tool,
    is_low_confidence,
    list_documents_tool,
    max_cosine_similarity,
    vector_search_tool,
)


def _result(vector_similarity: float) -> ChunkResult:
    return ChunkResult(
        chunk_id="c",
        document_id="d",
        content="x",
        score=0.03,
        vector_similarity=vector_similarity,
        document_title="t",
        document_source="s",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _chunk_row(**overrides):
    base = {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "content": "Sample content",
        "similarity": 0.9,
        "combined_score": 0.85,
        "metadata": {},
        "document_title": "Test Doc",
        "document_source": "s3://bucket/test.pdf",
    }
    base.update(overrides)
    return base


def _graph_row(**overrides):
    base = {
        "fact": "Company X acquired Company Y",
        "uuid": "uuid-001",
        "valid_at": "2023-01-01",
        "invalid_at": None,
        "source_node_uuid": None,
    }
    base.update(overrides)
    return base


# ── Input model validation ────────────────────────────────────────────────────


class TestInputModels:
    def test_vector_search_input_defaults(self):
        v = VectorSearchInput(query="test")
        assert v.limit == 10
        assert v.user_id is None

    def test_hybrid_search_input_weights(self):
        h = HybridSearchInput(query="test")
        assert 0 <= h.text_weight <= 1

    def test_graph_search_input_query_required(self):
        from pydantic import ValidationError

        with pytest.raises((ValidationError, TypeError)):
            GraphSearchInput()  # type: ignore[call-arg]

    def test_document_list_defaults(self):
        d = DocumentListInput()
        assert d.limit == 20
        assert d.offset == 0


# ── Vector search tool ────────────────────────────────────────────────────────


class TestVectorSearchTool:
    @pytest.mark.asyncio
    async def test_returns_chunk_results(self):
        embedding = [0.1] * 1536
        db_rows = [_chunk_row()]

        with (
            patch("agent.tools.generate_embedding", AsyncMock(return_value=embedding)),
            patch("agent.tools.vector_search", AsyncMock(return_value=db_rows)),
        ):
            results = await vector_search_tool(VectorSearchInput(query="AI research"))

        assert len(results) == 1
        assert isinstance(results[0], ChunkResult)
        assert results[0].score == 0.9

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self):
        with patch(
            "agent.tools.generate_embedding",
            AsyncMock(side_effect=RuntimeError("API down")),
        ):
            results = await vector_search_tool(VectorSearchInput(query="test"))

        assert results == []

    @pytest.mark.asyncio
    async def test_passes_user_id_to_db(self):
        captured: list = []

        async def _fake_vector_search(embedding, limit, user_id):
            captured.append(user_id)
            return []

        with (
            patch(
                "agent.tools.generate_embedding", AsyncMock(return_value=[0.0] * 1536)
            ),
            patch("agent.tools.vector_search", _fake_vector_search),
        ):
            await vector_search_tool(VectorSearchInput(query="q", user_id="alice"))

        assert captured[0] == "alice"


# ── Graph search tool ─────────────────────────────────────────────────────────


class TestGraphSearchTool:
    @pytest.mark.asyncio
    async def test_returns_graph_results(self):
        graph_rows = [_graph_row(), _graph_row(uuid="uuid-002")]

        with patch(
            "agent.tools.search_knowledge_graph", AsyncMock(return_value=graph_rows)
        ):
            results = await graph_search_tool(GraphSearchInput(query="acquisitions"))

        assert len(results) == 2
        assert isinstance(results[0], GraphSearchResult)

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self):
        with patch(
            "agent.tools.search_knowledge_graph",
            AsyncMock(side_effect=Exception("Neo4j down")),
        ):
            results = await graph_search_tool(GraphSearchInput(query="test"))

        assert results == []

    @pytest.mark.asyncio
    async def test_maps_fact_field(self):
        graph_rows = [_graph_row(fact="Google founded in 1998")]

        with patch(
            "agent.tools.search_knowledge_graph", AsyncMock(return_value=graph_rows)
        ):
            results = await graph_search_tool(GraphSearchInput(query="Google"))

        assert results[0].fact == "Google founded in 1998"


# ── Hybrid search tool ────────────────────────────────────────────────────────


class TestHybridSearchTool:
    @pytest.mark.asyncio
    async def test_returns_chunk_results(self):
        embedding = [0.2] * 1536
        db_rows = [_chunk_row(combined_score=0.88)]

        with (
            patch("agent.tools.generate_embedding", AsyncMock(return_value=embedding)),
            patch(
                "agent.retriever.hybrid_retriever.retrieve",
                AsyncMock(return_value=db_rows),
            ),
            patch("agent.tools.hybrid_search", AsyncMock(return_value=db_rows)),
        ):
            results = await hybrid_search_tool(
                HybridSearchInput(query="machine learning")
            )

        assert len(results) == 1
        assert results[0].score == 0.88

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self):
        with patch(
            "agent.tools.generate_embedding",
            AsyncMock(side_effect=ValueError("bad input")),
        ):
            results = await hybrid_search_tool(HybridSearchInput(query="test"))

        assert results == []


# ── List documents tool ───────────────────────────────────────────────────────


class TestListDocumentsTool:
    @pytest.mark.asyncio
    async def test_returns_document_metadata(self):
        db_rows = [
            {
                "id": "doc-1",
                "title": "Annual Report 2023",
                "source": "s3://reports/2023.pdf",
                "metadata": {"pages": 50},
                "created_at": "2023-12-01T00:00:00",
                "updated_at": "2023-12-01T00:00:00",
                "chunk_count": 120,
            }
        ]

        with patch("agent.tools.list_documents", AsyncMock(return_value=db_rows)):
            results = await list_documents_tool(DocumentListInput(limit=5))

        assert len(results) == 1
        assert isinstance(results[0], DocumentMetadata)
        assert results[0].title == "Annual Report 2023"
        assert results[0].chunk_count == 120

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self):
        with patch("agent.tools.list_documents", AsyncMock(side_effect=RuntimeError)):
            results = await list_documents_tool(DocumentListInput())

        assert results == []


# ── Confidence gate (cosine-similarity based) ─────────────────────────────────


class TestConfidenceGate:
    def test_max_cosine_empty_is_zero(self):
        assert max_cosine_similarity([]) == 0.0

    def test_max_cosine_picks_highest(self):
        chunks = [_result(0.20), _result(0.55), _result(0.31)]
        assert max_cosine_similarity(chunks) == 0.55

    def test_low_confidence_when_no_chunks(self):
        assert is_low_confidence([]) is True

    def test_low_confidence_below_threshold(self):
        # off-domain: top cosine well under the gate
        assert is_low_confidence([_result(0.10)]) is True

    def test_high_confidence_above_threshold(self):
        # relevant: top cosine above the gate
        above = CONFIDENCE_MIN_SIMILARITY + 0.2
        assert is_low_confidence([_result(above)]) is False

    def test_threshold_is_sane(self):
        # Must sit in the gap between off-domain (~0.1) and relevant (~0.4+).
        assert 0.1 < CONFIDENCE_MIN_SIMILARITY < 0.4
