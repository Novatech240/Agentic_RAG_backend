"""
Shared pytest fixtures and sys.modules stubs for unavailable external packages.

All stubs are installed before any project module is imported so that
module-level initialisation (e.g. ``embedding_client = get_embedding_client()``)
receives a well-formed mock object rather than raising ImportError.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Environment ───────────────────────────────────────────────────────────────

os.environ.setdefault("LLM_API_KEY", "test-llm-key")
os.environ.setdefault("EMBEDDING_API_KEY", "test-embed-key")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
# Access-control allow-lists: force the fixture values (not setdefault) so the
# access-control tests are deterministic even when a real .env is present in the
# environment (e.g. running the suite inside the Docker image with env_file).
os.environ["PRIVATE_DOCUMENT_USERS"] = "private_user"
os.environ["ADMIN_USERS"] = "admin_user"
# Session memory: force the in-process backend so tests never need a live Redis.
os.environ["SESSION_MEMORY_BACKEND"] = "memory"
# Reranking: force OFF in the test harness so unit tests are deterministic and
# never call the live Voyage API (a real VOYAGE_API_KEY in .env would otherwise
# be picked up via tools.py's load_dotenv). test_reranker.py patches
# rerank_enabled() directly to exercise the enabled path.
os.environ["RERANK_ENABLED"] = "false"
os.environ.pop("VOYAGE_API_KEY", None)

# ── openai stub ───────────────────────────────────────────────────────────────


def _make_async_openai_instance(*_args: Any, **_kwargs: Any) -> MagicMock:
    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=MagicMock(data=[MagicMock(embedding=[0.1] * 1536)])
    )
    return client


_openai_stub = MagicMock(name="openai")
_openai_stub.AsyncOpenAI = _make_async_openai_instance
_openai_stub.OpenAI = MagicMock(name="OpenAI")
_openai_stub.RateLimitError = type("RateLimitError", (Exception,), {})
_openai_stub.APIError = type("APIError", (Exception,), {})

sys.modules.setdefault("openai", _openai_stub)

# ── neo4j stub ────────────────────────────────────────────────────────────────

_neo4j_stub = MagicMock(name="neo4j")
_neo4j_stub.AsyncGraphDatabase = MagicMock()
_neo4j_stub.GraphDatabase = MagicMock()
sys.modules.setdefault("neo4j", _neo4j_stub)

# ── boto3 / botocore stubs ────────────────────────────────────────────────────

_boto3_stub = MagicMock(name="boto3")
sys.modules.setdefault("boto3", _boto3_stub)

_botocore_stub = MagicMock(name="botocore")
_botocore_exceptions_stub = MagicMock(name="botocore.exceptions")
_botocore_exceptions_stub.ClientError = type("ClientError", (Exception,), {})
sys.modules.setdefault("botocore", _botocore_stub)
sys.modules.setdefault("botocore.exceptions", _botocore_exceptions_stub)

# ── pydantic_ai stub ──────────────────────────────────────────────────────────

_pydantic_ai_stub = MagicMock(name="pydantic_ai")
_pydantic_ai_stub.Agent = MagicMock(name="Agent")
_pydantic_ai_stub.RunContext = MagicMock(name="RunContext")
sys.modules.setdefault("pydantic_ai", _pydantic_ai_stub)
sys.modules.setdefault("pydantic_ai.models", MagicMock())
sys.modules.setdefault("pydantic_ai.models.openai", MagicMock())
sys.modules.setdefault("pydantic_ai.providers", MagicMock())
sys.modules.setdefault("pydantic_ai.providers.openai", MagicMock())
sys.modules.setdefault("pydantic_ai.messages", MagicMock())

# ── fastapi / uvicorn stubs ───────────────────────────────────────────────────
# Route decorators (on both FastAPI app and APIRouter) must pass the decorated
# function through unchanged so tests can import and call the handlers directly.


class _FakeHTTPException(Exception):
    def __init__(self, status_code: int = 500, detail: str = ""):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class _FakeRouter:
    """Stub for both FastAPI app and APIRouter — route decorators are identity."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def _route(self, *args: Any, **kwargs: Any):
        return lambda f: f

    get = post = put = delete = patch = _route  # type: ignore[assignment]

    def exception_handler(self, *args: Any, **kwargs: Any):
        return lambda f: f

    def add_middleware(self, *args: Any, **kwargs: Any) -> None:
        pass

    def include_router(self, *args: Any, **kwargs: Any) -> None:
        pass


_fastapi_stub = MagicMock(name="fastapi")
_fastapi_stub.FastAPI = _FakeRouter
_fastapi_stub.APIRouter = _FakeRouter
_fastapi_stub.HTTPException = _FakeHTTPException
_fastapi_stub.Request = MagicMock(name="Request")
_fastapi_stub.Depends = MagicMock(name="Depends")
_fastapi_stub.BackgroundTasks = MagicMock(name="BackgroundTasks")
sys.modules.setdefault("fastapi", _fastapi_stub)
sys.modules.setdefault("fastapi.responses", MagicMock())
sys.modules.setdefault("fastapi.middleware", MagicMock())
sys.modules.setdefault("fastapi.middleware.cors", MagicMock())
sys.modules.setdefault("fastapi.middleware.gzip", MagicMock())
sys.modules.setdefault("uvicorn", MagicMock())

# ── llama_index stubs ─────────────────────────────────────────────────────────


class _FakeTextNode:
    def __init__(
        self,
        *args: Any,
        id_: str = "",
        text: str = "",
        metadata: Any = None,
        **kwargs: Any,
    ):
        self.node_id = id_
        self.text = text
        self.metadata = metadata or {}
        self.score = 0.0


class _FakeBM25Retriever:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._nodes: list = []

    @classmethod
    def from_defaults(cls, nodes: list, similarity_top_k: int = 10, **kwargs: Any):
        inst = cls()
        inst._nodes = nodes
        return inst

    def retrieve(self, query: str) -> list:
        return self._nodes[:5]


_llama_core_schema = MagicMock()
_llama_core_schema.TextNode = _FakeTextNode

_llama_bm25 = MagicMock()
_llama_bm25.BM25Retriever = _FakeBM25Retriever

sys.modules.setdefault("llama_index", MagicMock())
sys.modules.setdefault("llama_index.core", MagicMock())
sys.modules.setdefault("llama_index.core.schema", _llama_core_schema)
sys.modules.setdefault("llama_index.retrievers", MagicMock())
sys.modules.setdefault("llama_index.retrievers.bm25", _llama_bm25)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_text() -> str:
    return (
        "# Big Tech AI Initiatives\n\n"
        "## Google AI Strategy\n\n"
        "Google has invested heavily in artificial intelligence research. "
        "Their main focus areas include large language models, computer vision, "
        "and natural language processing.\n\n"
        "## Microsoft and OpenAI\n\n"
        "Microsoft's strategic partnership with OpenAI has positioned them as a "
        "leader in the generative AI space. Key products include GitHub Copilot, "
        "Azure OpenAI Service, and Copilot integration across Office 365.\n\n"
        "## Meta AI Research\n\n"
        "Meta AI (formerly FAIR) continues to publish influential research, "
        "including the LLaMA family of open-weight language models."
    )


@pytest.fixture()
def mock_embedding() -> list:
    return [0.1] * 1536


@pytest.fixture()
def mock_chunk_results() -> list:
    from agent.models import ChunkResult

    return [
        ChunkResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Google AI strategy content",
            score=0.95,
            metadata={},
            document_title="Big Tech AI Report",
            document_source="s3://bucket/report.pdf",
        ),
        ChunkResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="Microsoft OpenAI partnership details",
            score=0.87,
            metadata={},
            document_title="Big Tech AI Report",
            document_source="s3://bucket/report.pdf",
        ),
    ]


@pytest.fixture()
def mock_graph_results() -> list:
    from agent.models import GraphSearchResult

    return [
        GraphSearchResult(
            fact="Google acquired DeepMind in 2014",
            uuid="uuid-1",
            valid_at="2014-01-01",
        ),
        GraphSearchResult(
            fact="Microsoft invested $10B in OpenAI in 2023",
            uuid="uuid-2",
            valid_at="2023-01-23",
        ),
    ]
