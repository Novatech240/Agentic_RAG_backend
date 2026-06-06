"""
Unit tests for the turn-level answer cache (agent.answer_cache).

Contract: round-trip a good answer keyed by the canonical query, isolate by
access scope, invalidate on ingest (shared version counter), and fail-open when
Redis is absent or disabled.
"""

from __future__ import annotations

import pytest

from agent import answer_cache, query_cache


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict = {}

    def get(self, k):
        return self.store.get(k)

    def setex(self, k, ttl, v):
        self.store[k] = v

    def incr(self, k):
        self.store[k] = int(self.store.get(k, 0)) + 1
        return self.store[k]


@pytest.fixture
def fake_redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr("agent.redis_utils.get_redis_client", lambda: r)
    monkeypatch.setattr(answer_cache, "_ENABLED", True)
    return r


@pytest.mark.asyncio
async def test_roundtrip_hit(fake_redis):
    await answer_cache.set(
        "What does Phase 2 deliver?",
        None,
        "Phase 2 ships the personalisation engine [1].",
        tools=[{"tool_name": "search_documents", "args": {"query": "phase 2"}}],
    )
    got = await answer_cache.get("what does phase 2 deliver??", None)  # case/punct variant
    assert got is not None
    assert "personalisation engine" in got["answer"]
    assert got["tools"][0]["tool_name"] == "search_documents"


@pytest.mark.asyncio
async def test_scope_isolation(fake_redis):
    await answer_cache.set("fees", None, "Public answer [1].", tools=[])
    assert await answer_cache.get("fees", "user-7") is None  # per-user scope
    assert await answer_cache.get("fees", None) is not None


@pytest.mark.asyncio
async def test_ingest_bump_invalidates(fake_redis):
    await answer_cache.set("fees", None, "Answer [1].", tools=[])
    assert await answer_cache.get("fees", None) is not None
    query_cache.bump_version()  # an ingest invalidates BOTH caches at once
    assert await answer_cache.get("fees", None) is None


@pytest.mark.asyncio
async def test_fail_open_without_redis(monkeypatch):
    monkeypatch.setattr(answer_cache, "_ENABLED", True)
    monkeypatch.setattr("agent.redis_utils.get_redis_client", lambda: None)
    await answer_cache.set("fees", None, "Answer [1].", tools=[])  # no raise
    assert await answer_cache.get("fees", None) is None


@pytest.mark.asyncio
async def test_disabled_is_noop(monkeypatch, fake_redis):
    monkeypatch.setattr(answer_cache, "_ENABLED", False)
    await answer_cache.set("fees", None, "Answer [1].", tools=[])
    assert await answer_cache.get("fees", None) is None


@pytest.mark.asyncio
async def test_empty_query_not_cached(fake_redis):
    await answer_cache.set("   ", None, "Answer [1].", tools=[])
    assert await answer_cache.get("   ", None) is None
