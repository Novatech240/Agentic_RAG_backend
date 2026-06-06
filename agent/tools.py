"""
Tools for the Pydantic AI agent.
"""

import logging
import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field
from pathlib import Path

from dotenv import load_dotenv

from .db_utils import (
    vector_search,
    hybrid_search,
    get_document,
    list_documents,
    get_document_chunks,
)
from .graph_utils import search_knowledge_graph
from .models import ChunkResult, GraphSearchResult, DocumentMetadata
from .providers import get_embedding_client, get_embedding_model

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)

embedding_client = get_embedding_client()
EMBEDDING_MODEL = get_embedding_model()

# Confidence gate: minimum top pgvector cosine similarity for a retrieval to be
# considered grounded. Stable across retrieval paths (unlike fused RRF scores).
# text-embedding-3-large: relevant chunks ~0.4-0.6, noise ~0.1-0.25.
CONFIDENCE_MIN_SIMILARITY = float(os.getenv("CONFIDENCE_MIN_SIMILARITY", "0.25"))


def max_cosine_similarity(chunks: List["ChunkResult"]) -> float:
    """Highest pgvector cosine similarity among retrieved chunks (0.0 if none)."""
    if not chunks:
        return 0.0
    return max((getattr(c, "vector_similarity", 0.0) or 0.0) for c in chunks)


def is_low_confidence(chunks: List["ChunkResult"]) -> bool:
    """True when retrieval is too weak to ground an answer (gate it / escalate)."""
    return max_cosine_similarity(chunks) < CONFIDENCE_MIN_SIMILARITY


async def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using OpenAI.

    Args:
        text: Text to embed

    Returns:
        Embedding vector
    """
    try:
        response = await embedding_client.embeddings.create(
            model=EMBEDDING_MODEL, input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


# Tool Input Models
class VectorSearchInput(BaseModel):
    """Input for vector search tool."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    user_id: Optional[str] = Field(
        default=None, description="User ID for filtering private documents"
    )


class GraphSearchInput(BaseModel):
    """Input for graph search tool."""

    query: str = Field(..., description="Search query")


class HybridSearchInput(BaseModel):
    """Input for hybrid search tool."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    text_weight: float = Field(
        default=0.3, description="Weight for text similarity (0-1)"
    )
    user_id: Optional[str] = Field(
        default=None, description="User ID for filtering private documents"
    )


class DocumentInput(BaseModel):
    """Input for document retrieval."""

    document_id: str = Field(..., description="Document ID to retrieve")


class DocumentListInput(BaseModel):
    """Input for listing documents."""

    limit: int = Field(default=20, description="Maximum number of documents")
    offset: int = Field(default=0, description="Number of documents to skip")
    user_id: Optional[str] = Field(
        default=None, description="User ID for filtering private documents"
    )


# Tool Implementation Functions
async def vector_search_tool(input_data: VectorSearchInput) -> List[ChunkResult]:
    """
    Perform vector similarity search.

    Args:
        input_data: Search parameters

    Returns:
        List of matching chunks
    """
    try:
        # Generate embedding for the query
        embedding = await generate_embedding(input_data.query)

        # Perform vector search
        results = await vector_search(
            embedding=embedding, limit=input_data.limit, user_id=input_data.user_id
        )

        # Convert to ChunkResult models
        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["similarity"],
                vector_similarity=float(r.get("similarity", 0.0) or 0.0),
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"],
            )
            for r in results
        ]

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


async def graph_search_tool(input_data: GraphSearchInput) -> List[GraphSearchResult]:
    """
    Search the knowledge graph.

    Args:
        input_data: Search parameters

    Returns:
        List of graph search results
    """
    try:
        results = await search_knowledge_graph(query=input_data.query)

        # Convert to GraphSearchResult models
        return [
            GraphSearchResult(
                fact=r["fact"],
                uuid=r["uuid"],
                valid_at=r.get("valid_at"),
                invalid_at=r.get("invalid_at"),
                source_node_uuid=r.get("source_node_uuid"),
            )
            for r in results
        ]

    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        return []


async def hybrid_search_tool(input_data: HybridSearchInput) -> List[ChunkResult]:
    """
    Three-stage retrieval: (1) Vespa hybrid recall (BM25 + dense ANN + RRF),
    with a Postgres hybrid fallback, then (2) a cross-encoder rerank that
    reorders a widened candidate pool by true (query, chunk) relevance and
    truncates to ``limit``. Reranking is fail-open (no-op without a key).

    Args:
        input_data: Search parameters

    Returns:
        List of matching chunks, reranked, ranked best-first.
    """
    try:
        from . import reranker, query_cache

        # Stage 0: retrieval cache. A hit returns the final reranked context with
        # zero embedding/Vespa/rerank calls — the main per-turn cost lever.
        cached = await query_cache.get(
            input_data.query, input_data.user_id, input_data.limit
        )
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        embedding = await generate_embedding(input_data.query)
        t_embed = time.perf_counter()
        # Widen the recall pool when a reranker is active so it has room to
        # promote relevant chunks the fused rank buried; otherwise just ``limit``.
        candidate_limit = reranker.candidate_pool_size(input_data.limit)

        # Prefer the Vespa hybrid retriever; fall back to Postgres hybrid.
        try:
            from .retriever import hybrid_retriever

            results = await hybrid_retriever.retrieve(
                query=input_data.query,
                embedding=embedding,
                limit=candidate_limit,
                user_id=input_data.user_id,
            )
            score_key = "combined_score"
        except Exception as retriever_exc:
            logger.warning(
                "HybridRetriever unavailable, falling back: %s", retriever_exc
            )
            results = await hybrid_search(
                embedding=embedding,
                query_text=input_data.query,
                limit=candidate_limit,
                text_weight=input_data.text_weight,
                user_id=input_data.user_id,
            )
            score_key = "combined_score"
        t_recall = time.perf_counter()

        candidates = [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r.get(score_key, r.get("similarity", 0.0)),
                # cosine_similarity from the RRF retriever; `similarity` from the
                # SQL fallback — both are the raw pgvector cosine.
                vector_similarity=float(
                    r.get("cosine_similarity", r.get("similarity", 0.0)) or 0.0
                ),
                metadata=r.get("metadata", {}),
                document_title=r["document_title"],
                document_source=r["document_source"],
            )
            for r in results
        ]

        # Stage 2: cross-encoder rerank candidates down to the requested limit.
        reranked = await reranker.rerank(
            input_data.query, candidates, top_n=input_data.limit
        )
        t_rerank = time.perf_counter()

        # Per-stage observability: latency breakdown + top cosine for the gate.
        logger.info(
            "retrieval pipeline: embed=%.0fms recall=%.0fms(%d cands) "
            "rerank=%.0fms(%s) -> %d results, top_cos=%.3f",
            (t_embed - t0) * 1000,
            (t_recall - t_embed) * 1000,
            len(candidates),
            (t_rerank - t_recall) * 1000,
            "on" if reranker.rerank_enabled() else "off",
            len(reranked),
            max_cosine_similarity(reranked),
        )
        # Memoize the post-rerank context for repeat/near-duplicate questions.
        await query_cache.set(
            input_data.query, input_data.user_id, input_data.limit, reranked
        )
        return reranked

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return []


async def get_document_tool(input_data: DocumentInput) -> Optional[Dict[str, Any]]:
    """
    Retrieve a complete document.

    Args:
        input_data: Document retrieval parameters

    Returns:
        Document data or None
    """
    try:
        document = await get_document(input_data.document_id)

        if document:
            # Also get all chunks for the document
            chunks = await get_document_chunks(input_data.document_id)
            document["chunks"] = chunks

        return document

    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return None


async def list_documents_tool(input_data: DocumentListInput) -> List[DocumentMetadata]:
    """
    List available documents.

    Args:
        input_data: Listing parameters

    Returns:
        List of document metadata
    """
    try:
        documents = await list_documents(
            limit=input_data.limit, offset=input_data.offset, user_id=input_data.user_id
        )

        # Convert to DocumentMetadata models
        return [
            DocumentMetadata(
                id=d["id"],
                title=d["title"],
                source=d["source"],
                metadata=d["metadata"],
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                chunk_count=d.get("chunk_count"),
            )
            for d in documents
        ]

    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        return []
