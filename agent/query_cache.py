"""
Retrieval cache — the cost lever of the RAG pipeline.

Every chat turn otherwise pays for: one embedding call, one Vespa hybrid query,
and one cross-encoder rerank (an external Voyage round-trip). For repeated or
near-duplicate questions ("what is the tuition fee?" asked a hundred times a day)
that is pure waste. This module memoizes the **post-rerank** chunk list in Redis,
keyed by the normalized query + access scope + limit, so a cache hit returns the
final ranked context with zero model/engine calls.

Design
------
* **Correctness over cleverness.** The key is a hash of the *normalized* query
  text (lowercased, punctuation-stripped, whitespace-collapsed), so trivial
  surface variants collapse to one entry without risking a wrong-answer match.
  True paraphrase (embedding-similarity) caching is intentionally left out — it
  trades a small extra hit-rate for a real correctness risk, and is noted as
  future work in the README hardening roadmap.
* **Scope-safe.** Anonymous (public-only) and authenticated (user_id) callers
  never share a cache entry — the access scope is part of the key, mirroring the
  Vespa ``access_level`` filter so a cached private result can't leak.
* **Self-invalidating on ingest.** A global integer version (``rag:cache:ver``)
  is folded into every key. ``bump_version`` is called after each successful
  ingest, atomically orphaning all prior entries so freshly indexed documents
  are never masked by stale cache for longer than necessary.
* **Fail-open.** Any Redis error (or Redis absent) degrades to a clean miss —
  retrieval simply runs as it does today. The cache can never break a query.

Env
    RAG_QUERY_CACHE_ENABLED   master switch (default "true")
    RAG_CACHE_TTL             per-entry TTL seconds (default 300 = 5 min)
    RAG_CACHE_PREFIX          Redis key prefix (default "uchenab:ragcache:")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import List, Optional

from .models import ChunkResult

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("RAG_QUERY_CACHE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
_TTL = int(os.getenv("RAG_CACHE_TTL", "300"))
_PREFIX = os.getenv("RAG_CACHE_PREFIX", "uchenab:ragcache:")
_VER_KEY = f"{_PREFIX}ver"

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def cache_enabled() -> bool:
    return _ENABLED


def _normalize(query: str) -> str:
    """Collapse trivial surface variants to one canonical form."""
    low = (query or "").strip().lower()
    low = _PUNCT_RE.sub(" ", low)
    return _WS_RE.sub(" ", low).strip()


# Public aliases so the turn-level answer cache shares one normalization scheme
# and one invalidation signal (a single ingest bump orphans both caches).
def normalize_query(query: str) -> str:
    return _normalize(query)


def get_version() -> int:
    return _version()


def _scope(user_id: Optional[str]) -> str:
    # Mirror the Vespa access filter: anon == public-only, else per-user.
    return user_id or "public"


def _version() -> int:
    """Current cache generation; bumped on every ingest. 0 if Redis is down."""
    from .redis_utils import get_redis_client

    client = get_redis_client()
    if client is None:
        return 0
    try:
        raw = client.get(_VER_KEY)
        return int(raw) if raw is not None else 0
    except Exception:
        return 0


def _key(query: str, user_id: Optional[str], limit: int, version: int) -> str:
    raw = f"v{version}|{_scope(user_id)}|{limit}|{_normalize(query)}"
    digest = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()
    return f"{_PREFIX}{digest}"


async def get(
    query: str, user_id: Optional[str], limit: int
) -> Optional[List[ChunkResult]]:
    """Return cached reranked chunks for this query/scope, or None on miss."""
    if not _ENABLED:
        return None
    from .redis_utils import get_redis_client

    client = get_redis_client()
    if client is None:
        return None
    try:
        raw = client.get(_key(query, user_id, limit, _version()))
        if not raw:
            return None
        payload = json.loads(raw)
        chunks = [ChunkResult(**d) for d in payload]
        logger.info("RAG cache HIT (%d chunks) for %r", len(chunks), query[:60])
        return chunks
    except Exception as exc:  # fail-open: treat any error as a miss
        logger.debug("RAG cache get failed (miss): %s", exc)
        return None


async def set(
    query: str, user_id: Optional[str], limit: int, chunks: List[ChunkResult]
) -> None:
    """Memoize the post-rerank chunk list (best-effort)."""
    if not (_ENABLED and chunks):
        return
    from .redis_utils import get_redis_client

    client = get_redis_client()
    if client is None:
        return
    try:
        payload = json.dumps([c.model_dump() for c in chunks])
        client.setex(_key(query, user_id, limit, _version()), _TTL, payload)
    except Exception as exc:  # never break retrieval on a cache write
        logger.debug("RAG cache set failed: %s", exc)


def bump_version() -> None:
    """Invalidate the entire cache (call after a successful ingest). Fail-open."""
    from .redis_utils import get_redis_client

    client = get_redis_client()
    if client is None:
        return
    try:
        client.incr(_VER_KEY)
        logger.info("RAG cache invalidated (version bumped after ingest)")
    except Exception as exc:
        logger.debug("RAG cache version bump failed: %s", exc)
