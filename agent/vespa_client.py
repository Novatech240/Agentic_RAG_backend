"""
Async Vespa client — feed chunks and run hybrid (BM25 + dense + RRF) queries.

Vespa is the **primary** retrieval engine. Postgres remains the durable source
of truth and a fallback (see ``retriever.HybridRetriever``). Chunk ids are
shared between Postgres and Vespa so the two stores stay consistent and Vespa
can be rebuilt from Postgres without re-embedding (``scripts/backfill_vespa.py``).

All write helpers are best-effort (log + continue); ``search`` raises on
transport error so the caller can fall back to Postgres.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

VESPA_ENDPOINT = os.getenv("VESPA_ENDPOINT", "http://localhost:8080").rstrip("/")
VESPA_ENABLED = os.getenv("VESPA_ENABLED", "true").lower() == "true"
VESPA_NAMESPACE = os.getenv("VESPA_NAMESPACE", "chunks")
VESPA_TIMEOUT = float(os.getenv("VESPA_TIMEOUT", "10"))

_DOC_TYPE = "chunk"
_CLUSTER = "chunks"


def vespa_enabled() -> bool:
    return VESPA_ENABLED


def _doc_url(chunk_id: str) -> str:
    return (
        f"{VESPA_ENDPOINT}/document/v1/{VESPA_NAMESPACE}/{_DOC_TYPE}/docid/{chunk_id}"
    )


def _to_doc_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map a Postgres-shaped chunk row to Vespa document fields."""
    embedding = row.get("embedding") or []
    if isinstance(embedding, str):  # PostgREST may serialize halfvec as a string
        try:
            embedding = json.loads(embedding)
        except Exception:
            embedding = []
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, str):
        metadata = json.dumps(metadata)
    return {
        "chunk_id": str(row["id"]),
        "document_id": str(row.get("document_id", "")),
        "document_title": row.get("document_title", "") or "",
        "document_source": row.get("document_source", "") or "",
        "access_level": row.get("access_level", "public") or "public",
        "chunk_index": int(row.get("chunk_index", 0) or 0),
        "content": row.get("content", "") or "",
        "embedding": {"values": list(embedding)},
        "metadata": metadata,
    }


async def feed_chunks(rows: List[Dict[str, Any]]) -> int:
    """Feed/replace chunk documents in Vespa. Returns the number fed."""
    if not (VESPA_ENABLED and rows):
        return 0
    fed = 0
    async with httpx.AsyncClient(timeout=VESPA_TIMEOUT) as client:
        for row in rows:
            cid = str(row["id"])
            try:
                resp = await client.post(
                    _doc_url(cid), json={"fields": _to_doc_fields(row)}
                )
                if resp.status_code < 300:
                    fed += 1
                else:
                    logger.warning(
                        "Vespa feed %s failed: %s %s",
                        cid,
                        resp.status_code,
                        resp.text[:200],
                    )
            except Exception as exc:
                logger.warning("Vespa feed error for %s: %s", cid, exc)
    logger.info("Vespa fed %d/%d chunks", fed, len(rows))
    return fed


async def delete_chunks(chunk_ids: List[str]) -> None:
    """Delete specific chunks by id (best-effort)."""
    if not (VESPA_ENABLED and chunk_ids):
        return
    async with httpx.AsyncClient(timeout=VESPA_TIMEOUT) as client:
        for cid in chunk_ids:
            try:
                await client.delete(_doc_url(str(cid)))
            except Exception as exc:
                logger.warning("Vespa delete %s error: %s", cid, exc)


async def delete_by_document(document_id: str) -> None:
    """Delete every chunk of a document via a paged selection delete."""
    if not VESPA_ENABLED:
        return
    url = f"{VESPA_ENDPOINT}/document/v1/{VESPA_NAMESPACE}/{_DOC_TYPE}/docid"
    base = {
        "selection": f'{_DOC_TYPE}.document_id=="{document_id}"',
        "cluster": _CLUSTER,
    }
    async with httpx.AsyncClient(timeout=VESPA_TIMEOUT) as client:
        cont: Optional[str] = None
        try:
            while True:
                params = dict(base)
                if cont:
                    params["continuation"] = cont
                resp = await client.delete(url, params=params)
                data = resp.json() if resp.content else {}
                cont = data.get("continuation")
                if not cont:
                    break
        except Exception as exc:
            logger.warning(
                "Vespa delete-by-document %s error: %s", document_id, exc
            )


def _norm_features(mf: Dict[str, Any]) -> Dict[str, Any]:
    """Vespa may emit feature keys with/without spaces; normalize by stripping them."""
    return {k.replace(" ", ""): v for k, v in (mf or {}).items()}


async def search(
    query_text: str,
    embedding: List[float],
    limit: int = 10,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Hybrid BM25 + dense query fused by the ``hybrid`` rank profile.

    Returns chunk dicts shaped like the Postgres retriever output so callers are
    backend-agnostic. Raises on transport error so the caller can fall back.
    """
    if not VESPA_ENABLED:
        raise RuntimeError("Vespa disabled")

    # Match existing Postgres semantics: anonymous -> public only; user -> unfiltered.
    access_clause = "" if user_id else ' and access_level contains "public"'
    target = max(limit, 100)
    yql = (
        "select * from chunk where "
        f"(({{targetHits:{target}}}nearestNeighbor(embedding, q)) "
        f"or userInput(@userquery)){access_clause}"
    )
    body = {
        "yql": yql,
        "userquery": query_text,
        "input.query(q)": embedding,
        "ranking.profile": "hybrid",
        "hits": limit,
        "timeout": f"{VESPA_TIMEOUT}s",
    }

    async with httpx.AsyncClient(timeout=VESPA_TIMEOUT) as client:
        resp = await client.post(f"{VESPA_ENDPOINT}/search/", json=body)
        resp.raise_for_status()
        data = resp.json()

    hits = (data.get("root", {}) or {}).get("children", []) or []
    results: List[Dict[str, Any]] = []
    for h in hits:
        fields = h.get("fields", {}) or {}
        mf = _norm_features(fields.get("matchfeatures", {}))

        # Recover true cosine from the angular distance feature (cos of the angle).
        dist = mf.get("distance(field,embedding)")
        if dist is not None:
            try:
                cosine = math.cos(float(dist))
            except Exception:
                cosine = 0.0
        else:
            cosine = float(mf.get("closeness(field,embedding)", 0.0) or 0.0)

        meta = fields.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}

        results.append(
            {
                "chunk_id": fields.get("chunk_id", ""),
                "document_id": fields.get("document_id", ""),
                "content": fields.get("content", ""),
                "similarity": cosine,
                "cosine_similarity": cosine,
                "combined_score": float(h.get("relevance", 0.0) or 0.0),
                "bm25_score": float(mf.get("bm25(content)", 0.0) or 0.0),
                "document_title": fields.get("document_title", ""),
                "document_source": fields.get("document_source", ""),
                "metadata": meta,
            }
        )
    return results


async def healthy() -> bool:
    """True if the Vespa container reports the application as up."""
    if not VESPA_ENABLED:
        return False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{VESPA_ENDPOINT}/ApplicationStatus")
            return r.status_code == 200
    except Exception:
        return False
