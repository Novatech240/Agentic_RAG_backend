"""
Unit tests for the groundedness judge (agent.groundedness).

Contract: catch ungrounded answers (judge says false), pass grounded ones, skip
trivial/short answers, and fail-open (return grounded=True) on any judge error.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import groundedness
from agent.models import ChunkResult


def _chunk(content: str) -> ChunkResult:
    return ChunkResult(
        chunk_id="c1",
        document_id="d1",
        content=content,
        score=0.6,
        vector_similarity=0.6,
        metadata={},
        document_title="Doc",
        document_source="s3://b/d.pdf",
    )


def _judge_returning(grounded: bool, reason: str = ""):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=json.dumps({"grounded": grounded, "reason": reason})
                    )
                )
            ]
        )
    )
    return client


_ANSWER = "The tuition fee for Computer Science is 5,000 dollars per semester [1]."


@pytest.mark.asyncio
async def test_grounded_passes(monkeypatch):
    monkeypatch.setattr(groundedness, "_ENABLED", True)
    monkeypatch.setattr(
        "agent.providers.get_cached_llm", lambda: _judge_returning(True, "ok")
    )
    ok, _ = await groundedness.is_grounded(_ANSWER, [_chunk("CS tuition is $5,000")])
    assert ok is True


@pytest.mark.asyncio
async def test_ungrounded_flagged(monkeypatch):
    monkeypatch.setattr(groundedness, "_ENABLED", True)
    monkeypatch.setattr(
        "agent.providers.get_cached_llm",
        lambda: _judge_returning(False, "amount not in source"),
    )
    ok, reason = await groundedness.is_grounded(
        _ANSWER, [_chunk("CS admission requires a transcript")]
    )
    assert ok is False
    assert "amount" in reason


@pytest.mark.asyncio
async def test_short_answer_skipped(monkeypatch):
    monkeypatch.setattr(groundedness, "_ENABLED", True)
    # A greeting is below the word threshold — never call the judge.
    called = MagicMock()
    monkeypatch.setattr("agent.providers.get_cached_llm", called)
    ok, reason = await groundedness.is_grounded("Hello there!", [_chunk("x")])
    assert ok is True
    assert reason == "skipped"
    called.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_skips(monkeypatch):
    monkeypatch.setattr(groundedness, "_ENABLED", False)
    ok, reason = await groundedness.is_grounded(_ANSWER, [_chunk("anything")])
    assert ok is True
    assert reason == "skipped"


@pytest.mark.asyncio
async def test_fail_open_on_error(monkeypatch):
    monkeypatch.setattr(groundedness, "_ENABLED", True)
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr("agent.providers.get_cached_llm", lambda: client)
    ok, reason = await groundedness.is_grounded(_ANSWER, [_chunk("x")])
    assert ok is True
    assert reason == "error_failopen"


# ── enforce(): remediation vs abstain ─────────────────────────────────────────


def _judge_full(grounded, reason="", unsupported=None, corrected=""):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content=json.dumps(
                            {
                                "grounded": grounded,
                                "reason": reason,
                                "unsupported_claims": unsupported or [],
                                "corrected": corrected,
                            }
                        )
                    )
                )
            ]
        )
    )
    return client


@pytest.mark.asyncio
async def test_enforce_grounded_unchanged(monkeypatch):
    monkeypatch.setattr(groundedness, "_ENABLED", True)
    monkeypatch.setattr("agent.providers.get_cached_llm", lambda: _judge_full(True))
    out, status = await groundedness.enforce(_ANSWER, [_chunk("CS tuition is $5,000")])
    assert out == _ANSWER
    assert status == groundedness.STATUS_GROUNDED


@pytest.mark.asyncio
async def test_enforce_remediates_strips_unsupported(monkeypatch):
    monkeypatch.setattr(groundedness, "_ENABLED", True)
    monkeypatch.setattr(groundedness, "_REMEDIATION", True)
    monkeypatch.setattr(groundedness, "_REVERIFY", False)
    corrected = "KSA supports mada, STC Pay, Apple Pay, Tabby, and Tamara [1]."
    answer = "KSA supports mada, STC Pay, Apple Pay, Tabby, Tamara, and Google Wallet [1]."
    monkeypatch.setattr(
        "agent.providers.get_cached_llm",
        lambda: _judge_full(False, "google wallet not in source", ["Google Wallet"], corrected),
    )
    out, status = await groundedness.enforce(answer, [_chunk("KSA: mada, STC Pay, Apple Pay, Tabby, Tamara")])
    assert out == corrected
    assert "Google Wallet" not in out
    assert status == groundedness.STATUS_REMEDIATED


@pytest.mark.asyncio
async def test_enforce_abstains_when_nothing_salvageable(monkeypatch):
    from agent.citations import ABSTAIN_MESSAGE

    monkeypatch.setattr(groundedness, "_ENABLED", True)
    monkeypatch.setattr(groundedness, "_REMEDIATION", True)
    # corrected is empty -> nothing to keep -> abstain
    monkeypatch.setattr(
        "agent.providers.get_cached_llm",
        lambda: _judge_full(False, "all fabricated", ["everything"], ""),
    )
    out, status = await groundedness.enforce(_ANSWER, [_chunk("unrelated content here")])
    assert out == ABSTAIN_MESSAGE
    assert status == groundedness.STATUS_ABSTAINED


@pytest.mark.asyncio
async def test_enforce_abstains_when_remediation_disabled(monkeypatch):
    from agent.citations import ABSTAIN_MESSAGE

    monkeypatch.setattr(groundedness, "_ENABLED", True)
    monkeypatch.setattr(groundedness, "_REMEDIATION", False)
    monkeypatch.setattr(
        "agent.providers.get_cached_llm",
        lambda: _judge_full(False, "bad", ["x"], "A grounded remainder [1]."),
    )
    out, status = await groundedness.enforce(_ANSWER, [_chunk("x")])
    assert out == ABSTAIN_MESSAGE
    assert status == groundedness.STATUS_ABSTAINED


@pytest.mark.asyncio
async def test_cacheable_statuses():
    assert groundedness.STATUS_GROUNDED in groundedness.CACHEABLE_STATUSES
    assert groundedness.STATUS_REMEDIATED in groundedness.CACHEABLE_STATUSES
    assert groundedness.STATUS_ABSTAINED not in groundedness.CACHEABLE_STATUSES
    assert groundedness.STATUS_ERROR not in groundedness.CACHEABLE_STATUSES
