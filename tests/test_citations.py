"""
Unit tests for citation enforcement (agent.citations) — the anti-hallucination
backstop that abstains on fabricated or uncited claims.
"""

from __future__ import annotations

from agent.citations import (
    ABSTAIN_MESSAGE,
    enforce,
    extract_citations,
    validate_citations,
)
from agent.models import ChunkResult


def _chunks(n: int) -> list[ChunkResult]:
    return [
        ChunkResult(
            chunk_id=f"c{i}",
            document_id="d",
            content="x",
            score=0.5,
            vector_similarity=0.5,
            metadata={},
            document_title="T",
            document_source="s",
        )
        for i in range(n)
    ]


class TestExtractCitations:
    def test_extracts_all(self):
        assert extract_citations("Fees are $5k [1] and dorms [3].") == [1, 3]

    def test_none(self):
        assert extract_citations("Hello there!") == []


class TestValidate:
    def test_valid_in_range_passes(self):
        ok, reason = validate_citations("Tuition is $5,000 per year [1].", 3)
        assert ok and reason == "ok"

    def test_fabricated_citation_fails(self):
        ok, reason = validate_citations("The deadline is in May [9].", 3)
        assert not ok
        assert reason.startswith("fabricated_citation")

    def test_substantive_uncited_fails(self):
        text = "The admission deadline is the fifteenth of May and fees are due soon."
        ok, reason = validate_citations(text, 3)
        assert not ok
        assert reason == "uncited_substantive_answer"

    def test_abstention_exempt(self):
        ok, reason = validate_citations(
            "I don't have that information; please reach out to the relevant team.", 3
        )
        assert ok and reason == "exempt_non_answer"

    def test_short_greeting_exempt(self):
        ok, reason = validate_citations("Hello! How can I help?", 3)
        assert ok and reason == "exempt_non_answer"

    def test_no_chunks_uncited_ok(self):
        # With nothing retrieved we don't force a citation here (the gate handles it).
        ok, _ = validate_citations("Some general statement that is fairly long here.", 0)
        assert ok


class TestEnforce:
    def test_passes_valid_answer_through(self):
        text = "Tuition is $5,000 per year [1]."
        out, reason = enforce(text, _chunks(2))
        assert out == text and reason == "ok"

    def test_swaps_fabricated_for_abstain(self):
        out, reason = enforce("The deadline is May 15 [7].", _chunks(2))
        assert out == ABSTAIN_MESSAGE
        assert reason.startswith("fabricated_citation")
