"""
Unit tests for conversational query rewriting (agent.query_rewriter).

Fail-open contract: no history, disabled flag, over-length input, or any LLM
error must yield the original question verbatim.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import query_rewriter


def _llm_returning(text: str):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[MagicMock(message=MagicMock(content=text))]
        )
    )
    return client


@pytest.mark.asyncio
async def test_no_history_returns_original(monkeypatch):
    monkeypatch.setattr(query_rewriter, "_ENABLED", True)
    assert await query_rewriter.condense("", "and the fee?") == "and the fee?"


@pytest.mark.asyncio
async def test_disabled_returns_original(monkeypatch):
    monkeypatch.setattr(query_rewriter, "_ENABLED", False)
    out = await query_rewriter.condense("User: CS admissions", "and the fee?")
    assert out == "and the fee?"


@pytest.mark.asyncio
async def test_condenses_followup(monkeypatch):
    monkeypatch.setattr(query_rewriter, "_ENABLED", True)
    client = _llm_returning("What is the admission fee for Computer Science?")
    monkeypatch.setattr("agent.providers.get_cached_llm", lambda: client)
    monkeypatch.setattr("agent.providers.LLM_CHOICE", "gpt-4o-mini", raising=False)
    out = await query_rewriter.condense(
        "User: Admission requirements for Computer Science?\nAssistant: ...",
        "and the fee?",
    )
    assert out == "What is the admission fee for Computer Science?"


@pytest.mark.asyncio
async def test_fail_open_on_error(monkeypatch):
    monkeypatch.setattr(query_rewriter, "_ENABLED", True)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr("agent.providers.get_cached_llm", lambda: client)
    out = await query_rewriter.condense("User: hi", "and the fee?")
    assert out == "and the fee?"


@pytest.mark.asyncio
async def test_overlong_skips_rewrite(monkeypatch):
    monkeypatch.setattr(query_rewriter, "_ENABLED", True)
    monkeypatch.setattr(query_rewriter, "_MAXLEN", 20)
    long_q = "this is a very long already self contained question about fees"
    assert await query_rewriter.condense("User: hi", long_q) == long_q
