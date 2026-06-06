"""
Supabase database utilities (PostgREST via supabase-py).

All queries go through Supabase's REST API using the anon key.
RLS policies grant the anon role full access in this dev configuration.
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_client = None  # supabase.AsyncClient


async def initialize_database() -> None:
    """Create the Supabase async client."""
    global _client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_ANON_KEY must be set.")
    from supabase import acreate_client

    _client = await acreate_client(url, key)
    logger.info("Supabase client ready (%s)", url)


async def close_database() -> None:
    """Close the Supabase async client."""
    global _client
    if _client:
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = None
        logger.info("Supabase client closed")


async def test_connection() -> bool:
    """Return True if Supabase is reachable."""
    if not _client:
        return False
    try:
        await _client.table("documents").select("id").limit(1).execute()
        return True
    except Exception as exc:
        logger.error("Supabase connection test failed: %s", exc)
        return False


# ── Message persistence (web-chat transcript) ─────────────────────────────────


async def ensure_session(
    session_id: str, user_id: Optional[str] = None
) -> None:
    """Idempotently create the session row so message inserts satisfy the FK.

    ``messages.session_id`` references ``sessions.id``; web sessions live in
    Redis and were never written to Postgres, so the first ``add_message`` for a
    new session failed the foreign key. Upserting on the primary key here is a
    no-op for existing sessions and makes transcript persistence work for new
    ones. Best-effort: callers already guard ``add_message`` in try/except.
    """
    await (
        _client.table("sessions")
        .upsert({"id": session_id, "user_id": user_id}, on_conflict="id")
        .execute()
    )


async def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    message_id = str(uuid.uuid4())
    await (
        _client.table("messages")
        .insert(
            {
                "id": message_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "metadata": metadata or {},
            }
        )
        .execute()
    )
    await (
        _client.table("sessions")
        .update({"updated_at": "now()"})
        .eq("id", session_id)
        .execute()
    )
    return message_id


# ── Search ────────────────────────────────────────────────────────────────────


async def vector_search(
    embedding: List[float],
    limit: int = 10,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantic similarity search via the match_chunks RPC function."""
    try:
        result = await _client.rpc(
            "match_chunks",
            {
                "query_embedding": embedding,
                "match_count": limit,
                "filter_user_id": user_id,
            },
        ).execute()
        return result.data or []
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        return []


async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Combined vector + full-text search via the hybrid_search RPC function."""
    try:
        result = await _client.rpc(
            "hybrid_search",
            {
                "query_text": query_text,
                "query_embedding": embedding,
                "match_count": limit,
                "text_weight": text_weight,
                "filter_user_id": user_id,
            },
        ).execute()
        return result.data or []
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        return []


# ── Document upsert (deduplication) ──────────────────────────────────────────


async def get_document_by_source(source: str) -> Optional[Dict[str, Any]]:
    result = (
        await _client.table("documents")
        .select("*")
        .eq("source", source)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


async def upsert_document(
    source: str,
    title: str,
    content: str,
    content_hash: str,
    metadata: Optional[Dict[str, Any]] = None,
    access_level: str = "public",
) -> str:
    existing = await get_document_by_source(source)
    if existing:
        doc_id = existing["id"]
        await (
            _client.table("documents")
            .update(
                {
                    "title": title,
                    "content": content,
                    "content_hash": content_hash,
                    "metadata": metadata or {},
                }
            )
            .eq("id", doc_id)
            .execute()
        )
        return doc_id

    doc_id = str(uuid.uuid4())
    await (
        _client.table("documents")
        .insert(
            {
                "id": doc_id,
                "title": title,
                "source": source,
                "content": content,
                "content_hash": content_hash,
                "metadata": metadata or {},
                "access_level": access_level,
            }
        )
        .execute()
    )
    return doc_id


async def delete_document_chunks(document_id: str) -> None:
    await _client.table("chunks").delete().eq("document_id", document_id).execute()


async def delete_chunks_by_ids(chunk_ids: list) -> None:
    """Delete specific chunks by their UUID list."""
    if not chunk_ids:
        return
    await _client.table("chunks").delete().in_("id", chunk_ids).execute()


async def save_chunks(document_id: str, embedded_chunks: list) -> List[Dict[str, Any]]:
    """Insert chunk rows into Postgres and return them (with generated ids).

    Ids are generated here (rather than relying on the DB default) so the caller
    can feed Vespa the *same* chunk_id — keeping the two stores consistent.
    """
    if not embedded_chunks:
        return []
    import hashlib

    rows = []
    for idx, chunk in enumerate(embedded_chunks):
        embedding = getattr(chunk, "embedding", None)
        content_hash = hashlib.sha256(
            chunk.content.encode("utf-8", errors="replace")
        ).hexdigest()
        rows.append(
            {
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": chunk.content,
                "content_hash": content_hash,
                "embedding": embedding,
                "chunk_index": getattr(chunk, "index", idx),
                "token_count": getattr(chunk, "token_count", None),
                "metadata": getattr(chunk, "metadata", {}),
            }
        )
    await _client.table("chunks").insert(rows).execute()
    return rows


# ── Documents ─────────────────────────────────────────────────────────────────


async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    result = (
        await _client.table("documents")
        .select("*")
        .eq("id", document_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    return rows[0] if rows else None


async def list_documents(
    limit: int = 20,
    offset: int = 0,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    query = _client.table("documents").select("*, chunks(count)")
    if user_id is None:
        query = query.eq("access_level", "public")
    result = (
        await query.order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    rows = result.data or []
    for r in rows:
        r["chunk_count"] = (
            r.pop("chunks", [{}])[0].get("count", 0) if r.get("chunks") else 0
        )
    return rows


async def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    result = await (
        _client.table("chunks")
        .select("*")
        .eq("document_id", document_id)
        .order("chunk_index")
        .execute()
    )
    return result.data or []


async def delete_graph_document(doc_key: str) -> None:
    """Delete document chunk nodes from Neo4j graph database."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not uri:
        return

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            # Detach and delete all Chunk nodes with the given doc_key
            session.run(
                "MATCH (c:Chunk {doc_key: $doc_key}) DETACH DELETE c", doc_key=doc_key
            )
        driver.close()
    except Exception as exc:
        logger.error("Graph deletion failed for %s: %s", doc_key, exc)


async def delete_document(document_id: str) -> None:
    """Delete document chunks from Postgres and Neo4j, then delete document."""
    doc = await get_document(document_id)
    if doc:
        source = doc.get("source")
        if source:
            await delete_graph_document(source)
    # Delete chunks first
    await delete_document_chunks(document_id)
    # Remove the same chunks from Vespa (best-effort)
    try:
        from . import vespa_client

        if vespa_client.vespa_enabled():
            await vespa_client.delete_by_document(document_id)
    except Exception as exc:
        logger.warning("Vespa delete-by-document failed for %s: %s", document_id, exc)
    # Delete document
    await _client.table("documents").delete().eq("id", document_id).execute()
