"""
Unit tests for Pydantic data models.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from agent.models import (
    ChatRequest,
    Chunk,
    ChunkResult,
    ErrorResponse,
    GraphSearchResult,
    HealthStatus,
    IngestionConfig,
    MessageRole,
    SearchRequest,
    SearchResponse,
    SearchType,
    ToolCall,
)


class TestEnumerations:
    def test_message_role_values(self):
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"

    def test_search_type_values(self):
        assert SearchType.VECTOR == "vector"
        assert SearchType.HYBRID == "hybrid"
        assert SearchType.GRAPH == "graph"


class TestChatRequest:
    def test_required_message(self):
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.session_id is None
        assert req.user_id is None
        assert req.search_type == SearchType.HYBRID

    def test_custom_search_type(self):
        req = ChatRequest(message="Hi", search_type=SearchType.VECTOR)
        assert req.search_type == SearchType.VECTOR

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest()  # type: ignore[call-arg]

    def test_metadata_defaults_to_empty(self):
        req = ChatRequest(message="test")
        assert req.metadata == {}


class TestSearchRequest:
    def test_default_limit(self):
        req = SearchRequest(query="test")
        assert req.limit == 10

    def test_limit_clamps_lower(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=0)

    def test_limit_clamps_upper(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=51)

    def test_limit_valid_boundary(self):
        assert SearchRequest(query="q", limit=1).limit == 1
        assert SearchRequest(query="q", limit=50).limit == 50


class TestChunkResult:
    def test_score_clamped_above_one(self):
        result = ChunkResult(
            chunk_id="c1",
            document_id="d1",
            content="text",
            score=1.5,
            document_title="title",
            document_source="src",
        )
        assert result.score == 1.0

    def test_score_clamped_below_zero(self):
        result = ChunkResult(
            chunk_id="c1",
            document_id="d1",
            content="text",
            score=-0.3,
            document_title="title",
            document_source="src",
        )
        assert result.score == 0.0

    def test_score_in_range_unchanged(self):
        result = ChunkResult(
            chunk_id="c1",
            document_id="d1",
            content="text",
            score=0.75,
            document_title="title",
            document_source="src",
        )
        assert result.score == 0.75


class TestGraphSearchResult:
    def test_minimal_fields(self):
        r = GraphSearchResult(fact="Some fact", uuid="uuid-123")
        assert r.fact == "Some fact"
        assert r.valid_at is None
        assert r.invalid_at is None

    def test_with_temporal_fields(self):
        r = GraphSearchResult(
            fact="Fact with time",
            uuid="uuid-456",
            valid_at="2023-01-01",
            invalid_at="2024-01-01",
        )
        assert r.valid_at == "2023-01-01"


class TestIngestionConfig:
    def test_defaults(self):
        cfg = IngestionConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.use_semantic_chunking is True

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValidationError):
            IngestionConfig(chunk_size=500, chunk_overlap=500)

    def test_overlap_just_below_chunk_size(self):
        cfg = IngestionConfig(chunk_size=500, chunk_overlap=499)
        assert cfg.chunk_overlap == 499

    def test_skip_graph_building_flag(self):
        cfg = IngestionConfig(skip_graph_building=True)
        assert cfg.skip_graph_building is True


class TestChunk:
    def test_valid_embedding(self):
        chunk = Chunk(
            document_id="d1",
            content="text",
            chunk_index=0,
            embedding=[0.1] * 768,
        )
        assert len(chunk.embedding) == 768

    def test_empty_embedding_raises(self):
        with pytest.raises(ValidationError):
            Chunk(document_id="d1", content="text", chunk_index=0, embedding=[])

    def test_none_embedding_allowed(self):
        chunk = Chunk(document_id="d1", content="text", chunk_index=0)
        assert chunk.embedding is None


class TestToolCall:
    def test_default_args(self):
        tc = ToolCall(tool_name="vector_search")
        assert tc.args == {}
        assert tc.tool_call_id is None

    def test_with_args(self):
        tc = ToolCall(tool_name="search", args={"query": "AI"}, tool_call_id="call-1")
        assert tc.args["query"] == "AI"


class TestSearchResponse:
    def test_defaults(self):
        resp = SearchResponse(
            total_results=0,
            search_type=SearchType.VECTOR,
            query_time_ms=12.5,
        )
        assert resp.results == []
        assert resp.graph_results == []


class TestHealthStatus:
    def test_healthy_status(self):
        h = HealthStatus(
            status="healthy",
            database=True,
            graph_database=True,
            llm_connection=True,
            version="0.1.0",
            timestamp=datetime.now(),
        )
        assert h.status == "healthy"

    def test_invalid_status_raises(self):
        with pytest.raises(ValidationError):
            HealthStatus(
                status="unknown",  # type: ignore[arg-type]
                database=True,
                graph_database=True,
                llm_connection=True,
                version="0.1.0",
                timestamp=datetime.now(),
            )


class TestErrorResponse:
    def test_minimal(self):
        err = ErrorResponse(error="Something went wrong", error_type="RuntimeError")
        assert err.details is None
        assert err.request_id is None
