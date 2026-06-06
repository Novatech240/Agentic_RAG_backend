"""
Cross-encoder reranking stage (precision stage of the retrieval pipeline).

The hybrid retriever (BM25 + dense ANN + RRF) is a high-recall *candidate*
generator: it casts a wide net but orders results only by fused rank. A
cross-encoder reranker then scores each (query, chunk) pair jointly and reorders
the candidate pool by true semantic relevance — the standard two-stage
"retrieve-then-rerank" pattern that drives precision@k up sharply.

This module isolates the reranker behind one swappable async function so the
provider can change (hosted Voyage today → Vespa-native global-phase ONNX later)
without touching callers. It is **fail-open**: with no API key, a disabled flag,
or any provider error, it degrades to a no-op that simply truncates the
candidate pool to ``top_n`` preserving the retriever's order. Retrieval never
breaks because reranking is unavailable.

The reranker reorders and rewrites ``ChunkResult.score`` (now a relevance
score), but deliberately leaves ``vector_similarity`` untouched so the
downstream cosine confidence gate keeps operating on the stable pgvector signal.

Env:
    RERANK_ENABLED     master on/off switch (default true; still no-ops w/o key)
    RERANK_PROVIDER    "voyage" (only provider implemented today)
    RERANK_MODEL       e.g. "rerank-2-lite"
    RERANK_CANDIDATES  candidate pool size to pull before reranking (default 50)
    RERANK_TIMEOUT     per-call HTTP timeout seconds (default 10)
    VOYAGE_API_KEY     hosted Voyage reranker key (absent -> no-op)
"""

from __future__ import annotations

import logging
import os
from typing import List

import httpx

from .models import ChunkResult

logger = logging.getLogger(__name__)

RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "voyage").strip().lower()
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-2-lite").strip()
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "50"))
RERANK_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "10"))

_VOYAGE_URL = "https://api.voyageai.com/v1/rerank"


def _enabled_flag() -> bool:
    return os.getenv("RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes"}


def _voyage_key() -> str:
    return (os.getenv("VOYAGE_API_KEY") or "").strip()


def rerank_enabled() -> bool:
    """True only when reranking is switched on *and* a provider key is present.

    Used by callers to decide whether to widen the candidate pool before
    retrieval. When False, ``rerank`` still works (as a no-op truncation), so
    callers may also call ``rerank`` unconditionally and rely on fail-open.
    """
    if not _enabled_flag():
        return False
    if RERANK_PROVIDER == "voyage":
        return bool(_voyage_key())
    return False


def candidate_pool_size(limit: int) -> int:
    """How many candidates to retrieve before reranking down to ``limit``."""
    if not rerank_enabled():
        return limit
    return max(limit, RERANK_CANDIDATES)


async def _voyage_rerank(
    query: str, documents: List[str], top_n: int
) -> List[tuple[int, float]]:
    """Return [(original_index, relevance_score), ...] ordered best-first."""
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_k": top_n,
        "truncation": True,
    }
    headers = {
        "Authorization": f"Bearer {_voyage_key()}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=RERANK_TIMEOUT) as client:
        resp = await client.post(_VOYAGE_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    # Voyage returns {"data": [{"index": i, "relevance_score": s}, ...]}
    out: List[tuple[int, float]] = []
    for item in data.get("data", []):
        idx = item.get("index")
        score = float(item.get("relevance_score", 0.0) or 0.0)
        if idx is not None:
            out.append((int(idx), score))
    return out


async def rerank(
    query: str, chunks: List[ChunkResult], top_n: int
) -> List[ChunkResult]:
    """Reorder ``chunks`` by cross-encoder relevance and return the top ``top_n``.

    Fail-open: on a disabled flag, missing key, empty input, or any provider
    error, returns ``chunks[:top_n]`` unchanged (retriever order preserved).
    ``ChunkResult.score`` is overwritten with the relevance score for reranked
    results; ``vector_similarity`` is never modified (the cosine gate relies on
    it).
    """
    if not chunks:
        return []
    top_n = max(1, min(top_n, len(chunks)))

    if not rerank_enabled():
        return chunks[:top_n]

    try:
        documents = [c.content for c in chunks]
        if RERANK_PROVIDER == "voyage":
            ranked = await _voyage_rerank(query, documents, top_n)
        else:  # unknown provider -> no-op
            return chunks[:top_n]
    except Exception as exc:  # fail-open: never break retrieval on rerank error
        logger.warning("Rerank failed (%s) — using retriever order: %s",
                       RERANK_PROVIDER, exc)
        return chunks[:top_n]

    if not ranked:
        return chunks[:top_n]

    reordered: List[ChunkResult] = []
    for idx, score in ranked:
        if 0 <= idx < len(chunks):
            c = chunks[idx]
            # Relevance score becomes the surfaced rank score; clamp to [0,1].
            c.score = max(0.0, min(1.0, score))
            reordered.append(c)
    return reordered[:top_n] if reordered else chunks[:top_n]
