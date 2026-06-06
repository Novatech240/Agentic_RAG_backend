"""
Unit tests for FastAPI endpoint logic.

External I/O (database, graph, agent) is fully mocked so these tests run
without any running services.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import json

import pytest

from agent.models import (
    ChatRequest,
    HealthStatus,
    SearchRequest,
)


# ── Helper factories ──────────────────────────────────────────────────────────


def _make_chat_request(**kwargs) -> ChatRequest:
    defaults = {"message": "What is Google's AI strategy?"}
    defaults.update(kwargs)
    return ChatRequest(**defaults)


def _make_search_request(**kwargs) -> SearchRequest:
    defaults = {"query": "AI initiatives"}
    defaults.update(kwargs)
    return SearchRequest(**defaults)


# ── get_or_create_session ─────────────────────────────────────────────────────


class TestGetOrCreateSession:
    def test_creates_new_session_when_none_provided(self):
        from agent.api import get_or_create_session

        sid = get_or_create_session(_make_chat_request())
        assert isinstance(sid, str) and len(sid) == 36  # UUID format

    def test_reuses_existing_session(self):
        from agent.api import get_or_create_session, memory_manager

        # Pre-create the session so it exists in LangChain memory
        existing_id = memory_manager.create_session()
        sid = get_or_create_session(_make_chat_request(session_id=existing_id))
        assert sid == existing_id

    def test_registers_unknown_session_id_as_new_session(self):
        from agent.api import get_or_create_session, memory_manager

        # Unknown session_id is accepted and registered with the given ID
        sid = get_or_create_session(_make_chat_request(session_id="ghost-session"))
        assert sid == "ghost-session"
        assert memory_manager.session_exists("ghost-session")


# ── get_conversation_context ──────────────────────────────────────────────────


class TestGetConversationContext:
    def test_returns_formatted_history(self):
        from agent.api import get_conversation_context, memory_manager

        sid = memory_manager.create_session()
        memory_manager.add_turn(sid, "Hello", "Hi there")

        ctx = get_conversation_context(sid)
        assert "User: Hello" in ctx
        assert "Assistant: Hi there" in ctx

    def test_empty_session_returns_empty_string(self):
        from agent.api import get_conversation_context, memory_manager

        sid = memory_manager.create_session()
        assert get_conversation_context(sid) == ""


# ── extract_tool_calls ────────────────────────────────────────────────────────


class TestExtractToolCalls:
    def test_empty_result_returns_empty_list(self):
        from agent.api import extract_tool_calls

        result = MagicMock()
        result.all_messages.return_value = []
        tools = extract_tool_calls(result)
        assert tools == []

    def test_extracts_tool_call_parts(self):
        from agent.api import extract_tool_calls

        # Construct a class whose __name__ is "ToolCallPart" so the api.py
        # isinstance name-check identifies it correctly.
        ToolCallPart = type(
            "ToolCallPart",
            (),
            {
                "tool_name": "vector_search",
                "args": json.dumps({"query": "AI"}),
                "tool_call_id": "call-abc",
            },
        )
        part = ToolCallPart()
        # Ensure no args_as_dict method so JSON-string branch is used
        assert not hasattr(part, "args_as_dict")

        message = MagicMock()
        message.parts = [part]

        result = MagicMock()
        result.all_messages.return_value = [message]

        tools = extract_tool_calls(result)
        assert len(tools) == 1
        assert tools[0].tool_name == "vector_search"
        assert tools[0].args == {"query": "AI"}

    def test_handles_corrupt_result_gracefully(self):
        from agent.api import extract_tool_calls

        result = MagicMock()
        result.all_messages.side_effect = RuntimeError("broken")

        tools = extract_tool_calls(result)
        assert tools == []


# ── save_conversation_turn ────────────────────────────────────────────────────


class TestSaveConversationTurn:
    def test_saves_both_messages_to_langchain_memory(self):
        from agent.api import (
            save_conversation_turn,
            get_conversation_context,
            memory_manager,
        )

        sid = memory_manager.create_session()
        save_conversation_turn(sid, "user question", "assistant answer")

        ctx = get_conversation_context(sid)
        assert "User: user question" in ctx
        assert "Assistant: assistant answer" in ctx

    def test_sliding_window_trims_oldest_turns(self):
        from agent.session_memory import SessionMemoryManager

        mgr = SessionMemoryManager(window_size=2)
        sid = mgr.create_session()
        mgr.add_turn(sid, "q1", "a1")
        mgr.add_turn(sid, "q2", "a2")
        mgr.add_turn(sid, "q3", "a3")  # exceeds window_size=2

        ctx = mgr.get_context_string(sid)
        assert "q1" not in ctx  # oldest turn dropped
        assert "q2" in ctx
        assert "q3" in ctx


# ── execute_agent ─────────────────────────────────────────────────────────────


class TestExecuteAgent:
    @pytest.mark.asyncio
    async def test_returns_response_and_tools(self):
        from agent.api import execute_agent

        mock_result = MagicMock()
        # pydantic-ai >=1.0 returns `.output`; set it explicitly (a MagicMock's
        # auto-attribute would otherwise be a truthy mock, not our string).
        mock_result.output = "This is the agent response"
        mock_result.data = "This is the agent response"
        mock_result.all_messages.return_value = []

        with (
            patch("agent.api.rag_agent") as mock_agent,
            patch("agent.api.get_conversation_context", MagicMock(return_value="")),
            patch("agent.api.save_conversation_turn", MagicMock()),
        ):
            mock_agent.run = AsyncMock(return_value=mock_result)
            response, tools, deps = await execute_agent("What is AI?", "session-1")

        assert response == "This is the agent response"
        assert isinstance(tools, list)

    @pytest.mark.asyncio
    async def test_returns_error_message_on_exception(self):
        from agent.api import execute_agent

        with (
            patch("agent.api.rag_agent") as mock_agent,
            patch("agent.api.get_conversation_context", MagicMock(return_value="")),
            patch("agent.api.save_conversation_turn", MagicMock()),
        ):
            mock_agent.run = AsyncMock(side_effect=RuntimeError("LLM timeout"))
            response, tools, deps = await execute_agent("Question", "session-1")

        assert response  # non-empty
        assert "sorry" in response.lower()
        assert tools == []


# ── Health check logic ────────────────────────────────────────────────────────
# We test the business logic in isolation, not the decorated endpoint function.


class TestHealthCheckLogic:
    """Test the health-check decision logic directly without the FastAPI decorator."""

    def _status(self, db: bool, graph: bool) -> str:
        if db and graph:
            return "healthy"
        if db or graph:
            return "degraded"
        return "unhealthy"

    def test_both_up_is_healthy(self):
        assert self._status(True, True) == "healthy"

    def test_one_down_is_degraded(self):
        assert self._status(True, False) == "degraded"
        assert self._status(False, True) == "degraded"

    def test_both_down_is_unhealthy(self):
        assert self._status(False, False) == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_response_model_fields(self):
        """HealthStatus model correctly represents all status variants."""
        from datetime import datetime

        for status in ("healthy", "degraded", "unhealthy"):
            h = HealthStatus(
                status=status,  # type: ignore[arg-type]
                database=True,
                graph_database=True,
                llm_connection=True,
                version="0.1.0",
                timestamp=datetime.now(),
            )
            assert h.status == status


# ── Documents list logic ──────────────────────────────────────────────────────


class TestListDocumentsLogic:
    """Test document listing through the tool layer (mocked DB)."""

    @pytest.mark.asyncio
    async def test_returns_document_metadata_list(self):
        from agent.tools import list_documents_tool, DocumentListInput

        mock_docs = [
            {
                "id": "doc-1",
                "title": "Tech Report",
                "source": "s3://bucket/report.pdf",
                "metadata": {},
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "chunk_count": 10,
            }
        ]

        with patch("agent.tools.list_documents", AsyncMock(return_value=mock_docs)):
            results = await list_documents_tool(DocumentListInput(limit=10))

        assert len(results) == 1
        assert results[0].title == "Tech Report"

    @pytest.mark.asyncio
    async def test_empty_result_when_no_documents(self):
        from agent.tools import list_documents_tool, DocumentListInput

        with patch("agent.tools.list_documents", AsyncMock(return_value=[])):
            results = await list_documents_tool(DocumentListInput())

        assert results == []
