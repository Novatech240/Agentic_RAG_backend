"""
Hybrid retriever: Vespa (BM25 + dense + RRF) with a Postgres fallback.

Vespa is the primary engine — it fuses true BM25 and dense HNSW retrieval with
Reciprocal Rank Fusion in a single query (see ``vespa_client`` and the
``vespa/`` application package). If Vespa is disabled or unreachable, retrieval
falls back to the Postgres ``hybrid_search`` RPC so the assistant keeps working
during the migration or a Vespa outage.

The module exposes a single ``hybrid_retriever`` singleton used by the agent
tools. ``build_index`` / ``rebuild_index`` are retained as no-ops for
backward-compatible callers — Vespa maintains its own indexes incrementally as
chunks are fed on ingest, so there is no in-process index to (re)build.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Vespa-first hybrid retrieval with a Postgres fallback."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k

    # ── Backward-compatible no-ops (Vespa indexes are maintained on feed) ──────

    async def build_index(self, chunks: Optional[List[Dict[str, Any]]] = None) -> None:
        return None

    async def rebuild_index(self) -> None:
        return None

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        embedding: List[float],
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query Vespa; on failure or empty result, fall back to Postgres."""
        from . import vespa_client

        if vespa_client.vespa_enabled():
            try:
                results = await vespa_client.search(
                    query_text=query,
                    embedding=embedding,
                    limit=limit,
                    user_id=user_id,
                )
                if results:
                    return results
                logger.info("Vespa returned no hits — falling back to Postgres")
            except Exception as exc:
                logger.warning(
                    "Vespa search failed — falling back to Postgres: %s", exc
                )

        return await self._postgres_fallback(query, embedding, limit, user_id)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    async def _postgres_fallback(
        query: str,
        embedding: List[float],
        limit: int,
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Postgres ``hybrid_search`` RPC, normalized to the Vespa output shape."""
        from .db_utils import hybrid_search

        rows = await hybrid_search(
            embedding=embedding,
            query_text=query,
            limit=limit,
            user_id=user_id,
        )
        for r in rows:
            # `similarity` here is the raw pgvector cosine — expose it the same
            # way the Vespa path does so the confidence gate is backend-agnostic.
            r["cosine_similarity"] = float(r.get("similarity", 0.0) or 0.0)
        return rows


# ── Module-level singleton ────────────────────────────────────────────────────

hybrid_retriever = HybridRetriever()
