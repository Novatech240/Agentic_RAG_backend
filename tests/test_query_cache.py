"""
Unit tests for the retrieval cache (agent.query_cache).

The cache is fail-open and self-invalidating: it must (1) round-trip the
post-rerank chunk list, (2) never share entries across access scopes, (3) miss
cleanly when Redis is absent, and (4) orphan all entries when the version bumps.
"""

from __future__ import annotations

import pytest

from agent import query_cache
from agent.models import ChunkResult


class FakeRedis:
    """Minimal in-memory stand-in for the bits of redis the cache touches."""

    def __init__(self) -> None:
        self.store: dict = {}

    def get(self, k):
        return self.store.get(k)

    def setex(self, k, ttl, v):
        self.store[k] = v

    def incr(self, k):
        self.store[k] = int(self.store.get(k, 0)) + 1
        return self.store[k]


def _chunk(cid: str, content: str = "hello world", cos: float = 0.5) -> ChunkResult:
    return ChunkResult(
        chunk_id=cid,
        document_id="doc-1",
        content=content,
        score=0.4,
        vector_similarity=cos,
        metadata={"page": 3},
        document_title="Doc",
        document_source="s3://b/d.pdf",
    )


@pytest.fixture
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr("agent.redis_utils.get_redis_client", lambda: r)
    monkeypatch.setattr(query_cache, "_ENABLED", True)
    return r


def test_normalize_collapses_surface_variants():
    a = query_cache._normalize("  What IS the Fee?! ")
    b = query_cache._normalize("what is the fee")
    assert a == b == "what is the fee"


@pytest.mark.asyncio
async def test_roundtrip_hit(fake_redis):
    chunks = [_chunk("c1"), _chunk("c2")]
    await query_cache.set("What is the fee?", None, 5, chunks)
    got = await query_cache.get("what is the fee", None, 5)  # variant spelling
    assert got is not None
    assert [c.chunk_id for c in got] == ["c1", "c2"]
    assert got[0].metadata == {"page": 3}


@pytest.mark.asyncio
async def test_scope_isolation(fake_redis):
    await query_cache.set("fees", None, 5, [_chunk("public")])
    # A logged-in user must not read the anonymous (public) cache entry.
    assert await query_cache.get("fees", "user-42", 5) is None
    assert await query_cache.get("fees", None, 5) is not None


@pytest.mark.asyncio
async def test_version_bump_invalidates(fake_redis):
    await query_cache.set("fees", None, 5, [_chunk("c1")])
    assert await query_cache.get("fees", None, 5) is not None
    query_cache.bump_version()  # simulate an ingest
    assert await query_cache.get("fees", None, 5) is None


@pytest.mark.asyncio
async def test_fail_open_without_redis(monkeypatch):
    monkeypatch.setattr(query_cache, "_ENABLED", True)
    monkeypatch.setattr("agent.redis_utils.get_redis_client", lambda: None)
    await query_cache.set("fees", None, 5, [_chunk("c1")])  # no-op, no raise
    assert await query_cache.get("fees", None, 5) is None


@pytest.mark.asyncio
async def test_disabled_is_noop(monkeypatch, fake_redis):
    monkeypatch.setattr(query_cache, "_ENABLED", False)
    await query_cache.set("fees", None, 5, [_chunk("c1")])
    assert await query_cache.get("fees", None, 5) is None
