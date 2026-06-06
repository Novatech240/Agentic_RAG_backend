"""
FastAPI application — all routes live under /api/v1.

Background ingestion (Celery)
-----------------------------
Ingestion runs on a Celery worker pool backed by Redis. Periodic full sweeps
are scheduled by Celery Beat (INGEST_INTERVAL_MINUTES, default 15). Manual and
webhook-driven ingests enqueue tasks and return a task id for status polling.

S3 webhook (real-time)
----------------------
POST /api/v1/ingest/webhook/s3 receives SNS-wrapped S3 event notifications and
enqueues a Celery task to ingest the changed object.
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import APIRouter

from .agent import rag_agent, AgentDependencies
from . import db_utils
from .db_utils import (
    initialize_database,
    close_database,
    test_connection,
)
from .graph_utils import initialize_graph, close_graph, test_graph_connection
from .models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthStatus,
    SearchRequest,
    SearchResponse,
    ToolCall,
)
from .tools import (
    vector_search_tool,
    graph_search_tool,
    hybrid_search_tool,
    list_documents_tool,
    is_low_confidence as retrieval_low_confidence,
    VectorSearchInput,
    GraphSearchInput,
    HybridSearchInput,
    DocumentListInput,
)
from .citations import enforce as enforce_citations

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV", "development")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── Redis-backed session memory (sliding window, no durable DB transcript) ─────
from .session_memory import memory_manager  # noqa: E402

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
if APP_ENV == "development":
    logger.setLevel(logging.DEBUG)

# ── Background ingestion ──────────────────────────────────────────────────────
# Periodic ingestion is now driven by Celery Beat (see worker/celery_app.py),
# not an in-process asyncio loop. This makes scheduling durable and lets us
# scale ingestion across multiple worker containers.


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Agentic RAG API …")
    try:
        await initialize_database()
        logger.info("PostgreSQL ready")
        await initialize_graph()
        logger.info("Neo4j ready")

        # Build BM25 index from existing chunks
        try:
            from .retriever import hybrid_retriever

            await hybrid_retriever.build_index()
            logger.info("BM25 index ready")
        except Exception as exc:
            logger.warning("BM25 index build skipped: %s", exc)

        logger.info("Agentic RAG API started (ingestion runs on Celery workers)")
    except Exception as exc:
        logger.error("Startup failed: %s", exc)
        raise
    yield
    logger.info("Shutting down …")
    try:
        await close_database()
        await close_graph()
        logger.info("Connections closed")
    except Exception as exc:
        logger.error("Shutdown error: %s", exc)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG with Knowledge Graph",
    description=(
        "AI research assistant combining BM25 + vector hybrid search "
        "with a Neo4j knowledge graph. All endpoints are versioned under /api/v1."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Explicit allow-list (never "*" with credentials — that combo is rejected by
# browsers and is a security anti-pattern). Origins come from CORS_ALLOW_ORIGINS
# (comma-separated); defaults cover the local Vite/React dev servers.
_cors_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ALLOW_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Versioned router — all public endpoints live here
v1_router = APIRouter(prefix="/api/v1")

# ── Helper functions (also importable for testing) ────────────────────────────


def get_or_create_session(request: ChatRequest) -> str:
    """Return existing session_id or create a new one via LangChain memory."""
    return memory_manager.get_or_create(request.session_id)


def get_conversation_context(session_id: str) -> str:
    """Return the formatted conversation history string for this session."""
    return memory_manager.get_context_string(session_id)


def extract_tool_calls(result) -> List[ToolCall]:
    tools_used: List[ToolCall] = []
    try:
        for message in result.all_messages():
            if not hasattr(message, "parts"):
                continue
            for part in message.parts:
                if part.__class__.__name__ != "ToolCallPart":
                    continue
                try:
                    tool_name = str(getattr(part, "tool_name", "unknown"))
                    tool_args: Dict[str, Any] = {}
                    raw_args = getattr(part, "args", None)
                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    elif isinstance(raw_args, dict):
                        tool_args = raw_args
                    tool_call_id = (
                        str(part.tool_call_id)
                        if getattr(part, "tool_call_id", None)
                        else None
                    )
                    tools_used.append(
                        ToolCall(
                            tool_name=tool_name,
                            args=tool_args,
                            tool_call_id=tool_call_id,
                        )
                    )
                except Exception as exc:
                    logger.debug("Failed to parse tool call part: %s", exc)
    except Exception as exc:
        logger.warning("Failed to extract tool calls: %s", exc)
    return tools_used


def save_conversation_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Append a turn to LangChain memory (sliding window, no DB write)."""
    memory_manager.add_turn(session_id, user_message, assistant_message)


async def get_optional_current_user(request: Request) -> Optional[Dict]:
    """Optionally extract the current authenticated user from JWT bearer token."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth[7:]
    try:
        from .auth_utils import decode_access_token

        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            return None
        return {
            "id": user_id,
            "email": payload.get("email"),
            "roles": payload.get("roles", []),
        }
    except Exception:
        return None


async def execute_agent(
    message: str,
    session_id: str,
    user_id: Optional[str] = None,
    search_query: Optional[str] = None,
) -> tuple[str, List[ToolCall], AgentDependencies]:
    # ── Input guardrails (fail closed) ────────────────────────────────────────
    from .guardrails import check_input, apply_output_guardrails

    deps = AgentDependencies(session_id=session_id, user_id=user_id)
    deps.retrieved_chunks = []
    deps.graph_facts = []
    deps.selected_retrieval_tool = None

    verdict = check_input(message)
    if not verdict.allowed:
        blocked = (
            verdict.user_message or "Sorry, I can't answer that question."
        )
        save_conversation_turn(session_id, message, blocked)
        return blocked, [], deps
    safe_message = verdict.sanitized_input or message

    try:
        # Build prompt: prepend LangChain memory context if the session has history
        history = get_conversation_context(session_id)
        # Condense follow-ups into a self-contained question so retrieval embeds
        # the full intent, not a context-free fragment ("and the fee?"). The
        # caller may pass a pre-computed standalone query (reused as the cache
        # key + scope input) to avoid a second rewrite call.
        if search_query is None:
            from .query_rewriter import condense

            search_query = await condense(history, safe_message)
        full_prompt = (
            f"Previous conversation:\n{history}\n\nCurrent question: {search_query}"
            if history
            else search_query
        )

        result = await rag_agent.run(full_prompt, deps=deps)
        # pydantic-ai >=1.0 exposes `.output`; older versions used `.data`.
        response: str = getattr(result, "output", None) or getattr(result, "data", "")
        tools_used = extract_tool_calls(result)

        # ── Output guardrails (leak scrub, PII redaction) ─────────────────────
        response = await apply_output_guardrails(response)

        # Persist to LangChain memory (in-process, no DB)
        save_conversation_turn(session_id, message, response)
        return response, tools_used, deps

    except Exception as exc:
        logger.error("Agent execution failed: %s", exc)
        error_response = "Sorry — there was a problem processing your request. Please try again."
        save_conversation_turn(session_id, message, error_response)
        return error_response, [], deps


# ── /api/v1/health ────────────────────────────────────────────────────────────


@v1_router.get("/health", response_model=HealthStatus, tags=["health"])
async def health_check():
    """Service health check — reports database and graph database connectivity."""
    try:
        db_status = await test_connection()
        graph_status = await test_graph_connection()
        if db_status and graph_status:
            status = "healthy"
        elif db_status or graph_status:
            status = "degraded"
        else:
            status = "unhealthy"
        return HealthStatus(
            status=status,
            database=db_status,
            graph_database=graph_status,
            llm_connection=True,
            version="1.0.0",
            timestamp=datetime.now(),
        )
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        raise HTTPException(status_code=500, detail="Health check failed")


# ── /api/v1/chat ──────────────────────────────────────────────────────────────


@v1_router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest, fastapi_req: Request):
    """Non-streaming chat — returns the full agent response in one payload."""
    try:
        session_id = get_or_create_session(request)

        # ── Auth check for admin overlay ──
        user = await get_optional_current_user(fastapi_req)
        is_admin = user is not None and "admin" in user.get("roles", [])

        # ── Canonical query: condense the turn once, reuse for the cache key,
        # the scope check, and retrieval (avoids re-rewriting downstream). ──
        history = get_conversation_context(session_id)
        from .query_rewriter import condense

        canonical_query = await condense(history, request.message)

        # ── Turn-level answer cache (skip for admins who need fresh provenance) ──
        from . import answer_cache

        cached = (
            None if is_admin else await answer_cache.get(canonical_query, request.user_id)
        )
        if cached:
            cached_answer = cached.get("answer", "")
            cached_tools = [ToolCall(**t) for t in cached.get("tools", [])]
            save_conversation_turn(session_id, request.message, cached_answer)
            try:
                await db_utils.ensure_session(session_id, request.user_id)
                await db_utils.add_message(session_id, "user", request.message)
                await db_utils.add_message(
                    session_id,
                    "assistant",
                    cached_answer,
                    metadata={"cache_hit": True},
                )
            except Exception as db_exc:
                logger.warning("Failed to persist cached turn to DB: %s", db_exc)
            return ChatResponse(
                message=cached_answer,
                session_id=session_id,
                tools_used=cached_tools,
                metadata={"search_type": str(request.search_type), "cache_hit": True},
            )

        # ── Out-of-Scope check (Tier 1) — history-aware so follow-ups resolve ──
        from .guardrails import classify_query_scope

        scope_verdict = await classify_query_scope(canonical_query, history=history)

        if scope_verdict == "out_of_scope":
            from .settings_store import get_config

            decline_msg = (await get_config()).out_of_scope_message
            # Log turn to postgres messages
            try:
                await db_utils.ensure_session(session_id, request.user_id)
                await db_utils.add_message(session_id, "user", request.message)
                await db_utils.add_message(
                    session_id,
                    "assistant",
                    decline_msg,
                    metadata={"out_of_scope": True},
                )
            except Exception as db_exc:
                logger.warning("Failed to save out-of-scope turn to DB: %s", db_exc)

            return ChatResponse(
                message=decline_msg,
                session_id=session_id,
                tools_used=[],
                metadata={"out_of_scope": True},
            )

        # ── RAG Agent Turn Execution ──
        response, tools_used, deps = await execute_agent(
            message=request.message,
            session_id=session_id,
            user_id=request.user_id,
            search_query=canonical_query,
        )

        # ── Extract search metrics and build provenance & debug metadata ──
        chunks = getattr(deps, "retrieved_chunks", []) or []
        graph_facts = getattr(deps, "graph_facts", []) or []
        selected_tool = getattr(deps, "selected_retrieval_tool", "none") or "none"

        prov_sources = []
        for chunk in chunks:
            meta = chunk.metadata or {}
            page_sec = (
                meta.get("page") or meta.get("section") or meta.get("page_number")
            )
            page_sec_str = f"Page/Section {page_sec}" if page_sec else None
            prov_sources.append(
                {
                    "source_document_name": chunk.document_title,
                    "chunk_id": chunk.chunk_id,
                    "page_section": page_sec_str,
                    "retrieval_method_used": selected_tool,
                    "confidence_score": float(chunk.score),
                }
            )

        debug_metadata = {
            "provenance": {
                "sources": prov_sources,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "user_query": request.message,
                "final_answer": response,
            },
            "debug": {
                "selected_retrieval_tool": selected_tool,
                "retrieved_chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "score": float(c.score),
                        "document_title": c.document_title,
                        "document_source": c.document_source,
                    }
                    for c in chunks
                ],
                "neo4j_results": [
                    {"fact": f.fact, "valid_at": f.valid_at} for f in graph_facts
                ],
                "redis_session_context": get_conversation_context(session_id),
                "final_generated_prompt_summary": f"Previous conversation context + current question: {request.message}",
                "final_answer": response,
                "guardrail_result": {
                    "input_allowed": True,
                    "input_reason": None,
                    "output_applied": True,
                },
            },
        }

        # ── Confidence Gate (Tier 2) — stable cosine-similarity threshold ──
        is_low_confidence = retrieval_low_confidence(chunks)
        cacheable = False  # only a verified-good answer is memoized (see below)

        if is_low_confidence:
            # Trigger background Celery task
            from worker.tasks import notify_admin_weak_context_task

            notify_admin_weak_context_task.delay(session_id, request.message)

            response = (
                "Sorry, I couldn't find a reliable answer to your question in the "
                "knowledge base. I've alerted an admin, who will help you shortly."
            )
            debug_metadata["debug"]["low_confidence_triggered"] = True
        elif selected_tool == "hybrid_search":
            # ── Citation enforcement (Tier 3) — abstain on fabricated/uncited claims
            response, cite_reason = enforce_citations(response, chunks)
            debug_metadata["debug"]["citation_check"] = cite_reason

            # ── Groundedness (Tier 4) — remediate (strip unsupported claims) or
            # abstain. Catches confident mis-citations the regex gate can't see.
            if cite_reason == "ok":
                from . import groundedness

                response, gstatus = await groundedness.enforce(response, chunks)
                debug_metadata["debug"]["groundedness"] = gstatus
                cacheable = gstatus in groundedness.CACHEABLE_STATUSES

            debug_metadata["provenance"]["final_answer"] = response
            debug_metadata["debug"]["final_answer"] = response

        # ── Memoize verified-good answers for the turn-level cache ──
        if cacheable and not is_admin:
            await answer_cache.set(
                canonical_query,
                request.user_id,
                response,
                tools=[{"tool_name": t.tool_name, "args": t.args} for t in tools_used],
            )

        # ── Persistent Supabase Logs (Phase 2) ──
        try:
            await db_utils.ensure_session(session_id, request.user_id)
            await db_utils.add_message(session_id, "user", request.message)
            await db_utils.add_message(
                session_id, "assistant", response, metadata=debug_metadata
            )
        except Exception as exc:
            logger.warning("Failed to persist web chat turn to DB: %s", exc)

        # ── Admin-Only Debug Return (Admin Debug Mode) ──
        res_metadata = {"search_type": str(request.search_type)}
        if is_admin:
            res_metadata.update(debug_metadata)

        return ChatResponse(
            message=response,
            session_id=session_id,
            tools_used=tools_used,
            metadata=res_metadata,
        )
    except Exception as exc:
        logger.error("Chat endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.post("/chat/stream", tags=["chat"])
async def chat_stream(request: ChatRequest, fastapi_req: Request):
    """Streaming chat via Server-Sent Events (SSE)."""
    try:
        session_id = get_or_create_session(request)

        # ── Auth check for admin overlay ──
        user = await get_optional_current_user(fastapi_req)
        is_admin = user is not None and "admin" in user.get("roles", [])

        async def generate_stream():
            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

                # Ensure the session row exists so every downstream persist
                # branch (scope/low-confidence/final) satisfies the messages FK.
                try:
                    await db_utils.ensure_session(session_id, request.user_id)
                except Exception as db_exc:
                    logger.warning("ensure_session (stream) skipped: %s", db_exc)

                # ── Canonical query (history-resolved) — cache key + scope + retrieval
                history = get_conversation_context(session_id)
                from .query_rewriter import condense

                canonical_query = await condense(history, request.message)

                # ── Turn-level answer cache (skip for admins; fresh provenance) ──
                from . import answer_cache

                cached = (
                    None
                    if is_admin
                    else await answer_cache.get(canonical_query, request.user_id)
                )
                if cached:
                    cached_answer = cached.get("answer", "")
                    yield f"data: {json.dumps({'type': 'text', 'content': cached_answer})}\n\n"
                    save_conversation_turn(session_id, request.message, cached_answer)
                    try:
                        await db_utils.add_message(session_id, "user", request.message)
                        await db_utils.add_message(
                            session_id,
                            "assistant",
                            cached_answer,
                            metadata={"cache_hit": True},
                        )
                    except Exception as db_exc:
                        logger.warning("Failed to persist cached stream turn: %s", db_exc)
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    return

                # ── Out-of-Scope check (Tier 1) — history-aware ──
                from .guardrails import classify_query_scope

                scope_verdict = await classify_query_scope(
                    canonical_query, history=history
                )

                if scope_verdict == "out_of_scope":
                    from .settings_store import get_config

                    decline_msg = (await get_config()).out_of_scope_message
                    yield f"data: {json.dumps({'type': 'text', 'content': decline_msg})}\n\n"
                    # Log turns
                    try:
                        await db_utils.add_message(session_id, "user", request.message)
                        await db_utils.add_message(
                            session_id,
                            "assistant",
                            decline_msg,
                            metadata={"out_of_scope": True},
                        )
                    except Exception as db_exc:
                        logger.warning(
                            "Failed to save stream out-of-scope to DB: %s", db_exc
                        )
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    return

                # ── Confidence Gate Pre-check (Tier 2) ──
                from .tools import hybrid_search_tool, HybridSearchInput

                pre_chunks = []
                try:
                    pre_chunks = await hybrid_search_tool(
                        HybridSearchInput(
                            query=canonical_query, limit=10, user_id=request.user_id
                        )
                    )
                except Exception as search_exc:
                    logger.warning("Pre-check hybrid search failed: %s", search_exc)

                is_low_confidence = retrieval_low_confidence(pre_chunks)

                if is_low_confidence:
                    # Suppress response. Return notified fallback.
                    from worker.tasks import notify_admin_weak_context_task

                    notify_admin_weak_context_task.delay(session_id, request.message)

                    fallback_msg = (
                        "Sorry, I couldn't find a reliable answer to your question in the "
                        "knowledge base. I've alerted an admin, who will help you shortly."
                    )
                    yield f"data: {json.dumps({'type': 'text', 'content': fallback_msg})}\n\n"

                    # Log to Supabase
                    debug_metadata = {
                        "provenance": {
                            "sources": [],
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "user_query": request.message,
                            "final_answer": fallback_msg,
                        },
                        "debug": {
                            "selected_retrieval_tool": "none",
                            "retrieved_chunks": [],
                            "neo4j_results": [],
                            "redis_session_context": "",
                            "final_answer": fallback_msg,
                            "low_confidence_triggered": True,
                        },
                    }
                    try:
                        await db_utils.add_message(session_id, "user", request.message)
                        await db_utils.add_message(
                            session_id,
                            "assistant",
                            fallback_msg,
                            metadata=debug_metadata,
                        )
                    except Exception as db_exc:
                        logger.warning(
                            "Failed to save stream low-confidence turn to DB: %s",
                            db_exc,
                        )

                    if is_admin:
                        yield f"data: {json.dumps({'type': 'tools', 'tools': [], 'metadata': debug_metadata})}\n\n"
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    return

                # ── Normal Stream RAG ──
                from .guardrails import check_input, apply_output_guardrails

                verdict = check_input(request.message)
                if not verdict.allowed:
                    blocked = (
                        verdict.user_message
                        or "Sorry, I can't answer that question."
                    )
                    yield f"data: {json.dumps({'type': 'text', 'content': blocked})}\n\n"
                    save_conversation_turn(session_id, request.message, blocked)
                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                    return

                deps = AgentDependencies(session_id=session_id, user_id=request.user_id)
                deps.retrieved_chunks = []
                deps.graph_facts = []
                deps.selected_retrieval_tool = None

                # Reuse the canonical (history-resolved) query as the question so
                # the streamed agent retrieves on full intent, not a fragment.
                full_prompt = (
                    f"Previous conversation:\n{history}\n\nCurrent question: {canonical_query}"
                    if history
                    else canonical_query
                )

                full_response = ""

                async with rag_agent.iter(full_prompt, deps=deps) as run:
                    async for node in run:
                        if rag_agent.is_model_request_node(node):
                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    from pydantic_ai.messages import (
                                        PartStartEvent,
                                        PartDeltaEvent,
                                        TextPartDelta,
                                    )

                                    if (
                                        isinstance(event, PartStartEvent)
                                        and event.part.part_kind == "text"
                                    ):
                                        delta = event.part.content
                                        yield f"data: {json.dumps({'type': 'text', 'content': delta})}\n\n"
                                        full_response += delta
                                    elif isinstance(
                                        event, PartDeltaEvent
                                    ) and isinstance(event.delta, TextPartDelta):
                                        delta = event.delta.content_delta
                                        yield f"data: {json.dumps({'type': 'text', 'content': delta})}\n\n"
                                        full_response += delta

                tools_used = extract_tool_calls(run.result)

                # Assemble provenance and debug info
                chunks = deps.retrieved_chunks or []
                graph_facts = deps.graph_facts or []
                selected_tool = deps.selected_retrieval_tool or "hybrid_search"

                guarded = await apply_output_guardrails(full_response)
                # ── Citation enforcement (Tier 3) then groundedness (Tier 4) —
                # remediate/abstain. The full answer is buffered, so any change is
                # pushed to the client as a single `replace` event.
                cite_reason = "skipped"
                ground_status = "skipped"
                cacheable = False
                if selected_tool == "hybrid_search":
                    guarded, cite_reason = enforce_citations(guarded, chunks)
                    if cite_reason == "ok":
                        from . import groundedness

                        guarded, ground_status = await groundedness.enforce(
                            guarded, chunks
                        )
                        cacheable = ground_status in groundedness.CACHEABLE_STATUSES
                if guarded != full_response:
                    yield f"data: {json.dumps({'type': 'replace', 'content': guarded})}\n\n"

                save_conversation_turn(session_id, request.message, guarded)

                # Memoize verified-good answers for the turn-level cache.
                if cacheable and not is_admin:
                    await answer_cache.set(
                        canonical_query,
                        request.user_id,
                        guarded,
                        tools=[
                            {"tool_name": t.tool_name, "args": t.args}
                            for t in tools_used
                        ],
                    )

                prov_sources = []
                for chunk in chunks:
                    meta = chunk.metadata or {}
                    page_sec = (
                        meta.get("page")
                        or meta.get("section")
                        or meta.get("page_number")
                    )
                    page_sec_str = f"Page/Section {page_sec}" if page_sec else None
                    prov_sources.append(
                        {
                            "source_document_name": chunk.document_title,
                            "chunk_id": chunk.chunk_id,
                            "page_section": page_sec_str,
                            "retrieval_method_used": selected_tool,
                            "confidence_score": float(chunk.score),
                        }
                    )

                debug_metadata = {
                    "provenance": {
                        "sources": prov_sources,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "user_query": request.message,
                        "final_answer": guarded,
                    },
                    "debug": {
                        "selected_retrieval_tool": selected_tool,
                        "retrieved_chunks": [
                            {
                                "chunk_id": c.chunk_id,
                                "content": c.content,
                                "score": float(c.score),
                                "document_title": c.document_title,
                                "document_source": c.document_source,
                            }
                            for c in chunks
                        ],
                        "neo4j_results": [
                            {"fact": f.fact, "valid_at": f.valid_at}
                            for f in graph_facts
                        ],
                        "redis_session_context": get_conversation_context(session_id),
                        "final_generated_prompt_summary": f"Previous conversation context + current question: {request.message}",
                        "final_answer": guarded,
                        "citation_check": cite_reason,
                        "groundedness": ground_status,
                        "guardrail_result": {
                            "input_allowed": True,
                            "input_reason": None,
                            "output_applied": True,
                        },
                    },
                }

                # Save turns durably
                try:
                    await db_utils.add_message(session_id, "user", request.message)
                    await db_utils.add_message(
                        session_id, "assistant", guarded, metadata=debug_metadata
                    )
                except Exception as exc:
                    logger.warning("Failed to persist stream turn: %s", exc)

                # If the user is an admin, stream the final metadata event containing debug info
                if is_admin:
                    yield f"data: {json.dumps({'type': 'tools', 'tools': [t.model_dump() for t in tools_used], 'metadata': debug_metadata})}\n\n"
                elif tools_used:
                    yield f"data: {json.dumps({'type': 'tools', 'tools': [t.model_dump() for t in tools_used]})}\n\n"

                yield f"data: {json.dumps({'type': 'end'})}\n\n"

            except Exception as exc:
                logger.error("Stream error: %s", exc)
                yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except Exception as exc:
        logger.error("Streaming chat failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── /api/v1/search ────────────────────────────────────────────────────────────


@v1_router.post("/search/vector", response_model=SearchResponse, tags=["search"])
async def search_vector(request: SearchRequest):
    """Pure vector (semantic) search using pgvector cosine similarity."""
    try:
        t0 = datetime.now()
        results = await vector_search_tool(
            VectorSearchInput(
                query=request.query,
                limit=request.limit,
                user_id=request.filters.get("user_id"),
            )
        )
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="vector",
            query_time_ms=(datetime.now() - t0).total_seconds() * 1000,
        )
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.post("/search/graph", response_model=SearchResponse, tags=["search"])
async def search_graph(request: SearchRequest):
    """Knowledge graph full-text search."""
    try:
        t0 = datetime.now()
        results = await graph_search_tool(GraphSearchInput(query=request.query))
        return SearchResponse(
            graph_results=results,
            total_results=len(results),
            search_type="graph",
            query_time_ms=(datetime.now() - t0).total_seconds() * 1000,
        )
    except Exception as exc:
        logger.error("Graph search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.post("/search/hybrid", response_model=SearchResponse, tags=["search"])
async def search_hybrid(request: SearchRequest):
    """Hybrid search: BM25 (LlamaIndex) + pgvector fused with Reciprocal Rank Fusion."""
    try:
        t0 = datetime.now()
        results = await hybrid_search_tool(
            HybridSearchInput(
                query=request.query,
                limit=request.limit,
                user_id=request.filters.get("user_id"),
            )
        )
        return SearchResponse(
            results=results,
            total_results=len(results),
            search_type="hybrid",
            query_time_ms=(datetime.now() - t0).total_seconds() * 1000,
        )
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── /api/v1/documents ────────────────────────────────────────────────────────


@v1_router.get("/documents", tags=["documents"])
async def list_documents_endpoint(
    limit: int = 20,
    offset: int = 0,
    user_id: Optional[str] = None,
):
    """List ingested documents with optional user-based access filtering."""
    try:
        documents = await list_documents_tool(
            DocumentListInput(limit=limit, offset=offset, user_id=user_id)
        )
        return {
            "documents": documents,
            "total": len(documents),
            "limit": limit,
            "offset": offset,
        }
    except Exception as exc:
        logger.error("Document listing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.get("/documents/{document_id}", tags=["documents"])
async def get_document_endpoint(document_id: str):
    """Retrieve a single document's full details and content."""
    try:
        from .tools import get_document_tool, DocumentInput

        doc = await get_document_tool(DocumentInput(document_id=document_id))
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.delete("/documents/{document_id}", tags=["documents"])
async def delete_document_endpoint(document_id: str):
    """Delete an ingested document and its chunks."""
    try:
        from .db_utils import delete_document, get_document

        doc = await get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        await delete_document(document_id)
        # Rebuild BM25 index after deletion!
        try:
            from agent.retriever import hybrid_retriever

            await hybrid_retriever.rebuild_index()
        except Exception as exc:
            logger.warning("BM25 rebuild skipped: %s", exc)
        return {"message": "Document deleted successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document deletion failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@v1_router.post("/documents/upload", tags=["documents"])
async def upload_document_endpoint(
    file: UploadFile = File(...),
    access_level: str = Form("public"),
):
    """Upload and ingest a document file directly."""
    try:
        import tempfile
        from pathlib import Path
        from ingestion.file_parsers import parse_document

        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        try:
            content, file_meta = parse_document(tmp_path)
            if not content.strip():
                raise HTTPException(
                    status_code=400, detail="Document contains no parseable text"
                )

            # Unique source
            source = f"upload://{uuid.uuid4()}/{file.filename}"
            title = Path(file.filename).stem

            # Parsing is fast and done inline; the heavy chunk→embed→upsert→graph
            # work is offloaded to a Celery worker so the request returns quickly.
            from worker.tasks import ingest_document_task

            async_result = ingest_document_task.delay(
                content=content,
                source=source,
                title=title,
                metadata={**file_meta, "uploaded": True, "file_name": file.filename},
                access_level=access_level,
            )

            return {
                "status": "queued",
                "task_id": async_result.id,
                "title": title,
                "source": source,
                "message": "Document parsed and queued for indexing. Poll /api/v1/ingest/status/{task_id}.",
            }
        finally:
            import os

            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Document upload/ingest failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── /api/v1/sessions ─────────────────────────────────────────────────────────


@v1_router.get("/sessions/{session_id}", tags=["sessions"])
async def get_session_info(session_id: str):
    """Return in-memory session metadata (turn count, window size)."""
    if not memory_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return memory_manager.session_info(session_id)


# ── /api/v1/ingest ───────────────────────────────────────────────────────────


@v1_router.post("/ingest/trigger", tags=["ingest"])
async def trigger_ingest():
    """
    Enqueue a full S3 ingestion run on the Celery worker pool.

    Returns immediately with the Celery task id — poll
    ``GET /api/v1/ingest/status/{task_id}`` for progress and the final result.
    """
    from worker.tasks import ingest_all_task

    async_result = ingest_all_task.delay()
    logger.info("Enqueued full ingest (task_id=%s)", async_result.id)
    return {
        "status": "queued",
        "task_id": async_result.id,
        "message": "Ingestion queued on Celery. Poll /api/v1/ingest/status/{task_id}.",
    }


@v1_router.post("/ingest/s3", tags=["ingest"])
async def ingest_s3_bucket(bucket_type: str = "private", prefix: str = ""):
    """Enqueue ingestion of a specific S3 bucket / prefix on the worker pool."""
    from worker.tasks import ingest_bucket_task

    async_result = ingest_bucket_task.delay(bucket_type=bucket_type, prefix=prefix)
    logger.info(
        "Enqueued bucket ingest '%s/%s' (task_id=%s)",
        bucket_type,
        prefix,
        async_result.id,
    )
    return {
        "status": "queued",
        "task_id": async_result.id,
        "bucket_type": bucket_type,
        "prefix": prefix,
    }


@v1_router.get("/ingest/status/{task_id}", tags=["ingest"])
async def ingest_status(task_id: str):
    """Return the live state and result of a queued ingestion task."""
    from worker.celery_app import celery_app

    res = celery_app.AsyncResult(task_id)
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "state": res.state,  # PENDING | STARTED | SUCCESS | FAILURE | RETRY
        "ready": res.ready(),
        "successful": res.successful() if res.ready() else None,
    }
    if res.ready():
        # `.result` is the return value on success or the exception on failure.
        payload["result"] = res.result if res.successful() else str(res.result)
    return payload


@v1_router.post("/ingest/webhook/s3", tags=["ingest"])
async def s3_event_webhook(request: Request):
    """
    Receive AWS S3 event notifications (delivered via SNS HTTP subscription).

    S3 → SNS topic → HTTPS subscription → this endpoint

    Handles both SNS SubscriptionConfirmation and Notification message types.
    New or modified objects are enqueued for ingestion on the Celery workers.
    """
    body = await request.json()

    # SNS subscription confirmation handshake
    if body.get("Type") == "SubscriptionConfirmation":
        import urllib.request

        urllib.request.urlopen(body["SubscribeURL"])
        return {"status": "confirmed"}

    # S3 event notification
    if body.get("Type") == "Notification":
        try:
            message = json.loads(body.get("Message", "{}"))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid SNS message body")

        from worker.tasks import ingest_single_s3_task

        records = message.get("Records", [])
        queued = 0
        for record in records:
            s3_info = record.get("s3", {})
            bucket_name = s3_info.get("bucket", {}).get("name", "")
            object_key = s3_info.get("object", {}).get("key", "")
            event_name = record.get("eventName", "")

            if not object_key:
                continue

            if "ObjectCreated" in event_name or "ObjectModified" in event_name:
                logger.info(
                    "S3 event: %s → %s/%s (enqueuing)",
                    event_name,
                    bucket_name,
                    object_key,
                )
                ingest_single_s3_task.delay(s3_key=object_key, bucket_name=bucket_name)
                queued += 1

        return {"status": "accepted", "records_queued": queued}

    return {"status": "ignored"}


# ── Exception handler ─────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log the full detail server-side; return a generic, non-leaking payload.
    request_id = str(uuid.uuid4())
    logger.error("Unhandled exception (request_id=%s): %s", request_id, exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_type="InternalServerError",
            request_id=request_id,
        ).model_dump(),
    )


# ── Mount versioned router ────────────────────────────────────────────────────

app.include_router(v1_router)

from .auth_router import router as auth_router  # noqa: E402
from .users_router import router as users_router  # noqa: E402
from .settings_router import router as settings_router  # noqa: E402

app.include_router(auth_router)
app.include_router(users_router)
app.include_router(settings_router)


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "agent.api:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=APP_ENV == "development",
        log_level=LOG_LEVEL.lower(),
    )
