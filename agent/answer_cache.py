"""
Turn-level answer cache — the real cost lever.

The retrieval cache (``query_cache``) only saves the embedding + Vespa + rerank
calls, and only when the *agent's* tool query repeats verbatim — which it rarely
does, because the LLM rephrases its tool queries every turn. The dominant per-turn
cost is the LLM work itself: the agent loop plus the groundedness judge.

This cache memoizes the **whole turn outcome** (final answer + the tools the agent
used) keyed by the *canonical, self-contained query* — the history-resolved
question produced by ``query_rewriter.condense`` — so it is stable across
phrasings and across conversations that arrive at the same intent. A hit
short-circuits scope classification, the agent, retrieval, the gates, and the
groundedness judge entirely: zero model calls.

Correctness guarantees (enforced by callers + this module):
  * **Only good answers are stored.** Callers call :func:`set` solely for
    grounded / successfully-remediated answers — never abstentions, escalations,
    out-of-scope declines, or blocked input. A transient "I don't know" must
    never be frozen into the cache.
  * **Scope isolation.** Anonymous (public-only) and authenticated (per-user)
    turns never share an entry — the access scope is part of the key, mirroring
    the retrieval access filter, so a private answer can't leak to anon.
  * **Self-invalidating on ingest.** It shares ``query_cache``'s version counter,
    so a single ``bump_version`` after any ingest orphans both caches at once.
  * **Admin bypass.** Callers skip the cache for admin/debug requests so the
    provenance overlay is always freshly computed.
  * **Fail-open.** Redis absent or any error → clean miss / no-op; the turn runs
    normally.

Env
    ANSWER_CACHE_ENABLED   master switch (default "true")
    ANSWER_CACHE_TTL       per-entry TTL seconds (default 600 = 10 min)
    ANSWER_CACHE_PREFIX    Redis key prefix (default "uchenab:answercache:")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

from . import query_cache

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("ANSWER_CACHE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
_TTL = int(os.getenv("ANSWER_CACHE_TTL", "600"))
_PREFIX = os.getenv("ANSWER_CACHE_PREFIX", "uchenab:answercache:")


def cache_enabled() -> bool:
    return _ENABLED


def _scope(user_id: Optional[str]) -> str:
    return user_id or "public"


def _key(canonical_query: str, user_id: Optional[str]) -> str:
    raw = (
        f"v{query_cache.get_version()}|{_scope(user_id)}"
        f"|{query_cache.normalize_query(canonical_query)}"
    )
    digest = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()
    return f"{_PREFIX}{digest}"


async def get(
    canonical_query: str, user_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Return ``{"answer": str, "tools": [{tool_name, args}, ...]}`` or None."""
    if not (_ENABLED and (canonical_query or "").strip()):
        return None
    from .redis_utils import get_redis_client

    client = get_redis_client()
    if client is None:
        return None
    try:
        raw = client.get(_key(canonical_query, user_id))
        if not raw:
            return None
        payload = json.loads(raw)
        logger.info("Answer cache HIT for %r", canonical_query[:60])
        return payload
    except Exception as exc:  # fail-open: treat any error as a miss
        logger.debug("Answer cache get failed (miss): %s", exc)
        return None


async def set(
    canonical_query: str,
    user_id: Optional[str],
    answer: str,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Memoize a verified-good turn outcome (best-effort).

    ``tools`` is a list of ``{"tool_name": str, "args": dict}`` (no ids), kept so
    the cached response reports the same tool usage as a live turn.
    """
    if not (_ENABLED and answer and (canonical_query or "").strip()):
        return
    from .redis_utils import get_redis_client

    client = get_redis_client()
    if client is None:
        return
    try:
        payload = json.dumps({"answer": answer, "tools": tools or []})
        client.setex(_key(canonical_query, user_id), _TTL, payload)
    except Exception as exc:  # never break a turn on a cache write
        logger.debug("Answer cache set failed: %s", exc)
