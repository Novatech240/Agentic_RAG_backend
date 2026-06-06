"""
Citation enforcement — the anti-hallucination backstop applied after generation.

The agent is instructed (see agent._CITATION_RULES) to mark every factual claim
with an inline ``[n]`` citation referencing a retrieved chunk. Instructions
alone are not a guarantee, so this module *verifies* the produced answer against
the chunks that were actually retrieved this turn and decides whether to let it
through, scrub it, or abstain.

Two failure modes are caught:
  1. **Fabricated citation** — the answer cites ``[n]`` for an n that was never
     returned by the search tool. A model inventing source numbers is a strong
     hallucination signal; we abstain.
  2. **Ungrounded substantive answer** — the answer reads like a factual reply
     yet cites nothing while relevant chunks were available. We abstain.

Conversational/declining replies (greetings, scope refusals, "I don't have
that") are intentionally exempt so the gate doesn't fire on legitimate
non-answers. Whether retrieval was even attempted is decided by the caller; this
module only runs when grounded facts were expected.
"""

from __future__ import annotations

import re
from typing import List

from .models import ChunkResult

_CITATION_RE = re.compile(r"\[(\d+)\]")

# Phrases that mark a non-answer (refusal / abstention / no-info). These are
# allowed to carry no citations.
_ABSTAIN_MARKERS = (
    "don't have",
    "do not have",
    "couldn't find",
    "could not find",
    "no information",
    "not able to",
    "outside my scope",
    "out of scope",
    "can only answer",
    "reach out to",
    "contact the relevant",
)

ABSTAIN_MESSAGE = (
    "I don't have enough grounded information in the knowledge base to answer "
    "that accurately. Please rephrase, or reach out to the relevant team for "
    "the most reliable answer."
)


def extract_citations(text: str) -> List[int]:
    """All bracketed integer citations in order of appearance (may repeat)."""
    return [int(m) for m in _CITATION_RE.findall(text or "")]


def _looks_like_abstention(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _ABSTAIN_MARKERS)


def _is_substantive(text: str) -> bool:
    """Heuristic: a real answer vs. a greeting/short acknowledgement."""
    words = re.findall(r"\w+", text or "")
    return len(words) >= 12


def validate_citations(text: str, n_chunks: int) -> tuple[bool, str]:
    """Check an answer's citations against the ``n_chunks`` retrieved this turn.

    Returns ``(ok, reason)``. ``ok=False`` means the answer should be replaced
    with :data:`ABSTAIN_MESSAGE`. ``reason`` is a short tag for logging.
    """
    cites = extract_citations(text)

    # Fabricated citation: references a source index that doesn't exist.
    out_of_range = [c for c in cites if c < 1 or c > n_chunks]
    if out_of_range:
        return False, f"fabricated_citation:{out_of_range}"

    # Any valid in-range citation means the answer is grounded — pass.
    if cites:
        return True, "ok"

    # Non-answers (refusals/greetings) need no citations.
    if _looks_like_abstention(text) or not _is_substantive(text):
        return True, "exempt_non_answer"

    # A substantive factual answer with chunks available but zero citations is
    # treated as ungrounded.
    if n_chunks > 0:
        return False, "uncited_substantive_answer"

    return True, "ok"


def enforce(
    text: str,
    chunks: List[ChunkResult],
) -> tuple[str, str]:
    """Apply citation validation, returning ``(final_text, reason)``.

    On failure the answer is swapped for :data:`ABSTAIN_MESSAGE`.
    """
    ok, reason = validate_citations(text, len(chunks))
    if not ok:
        return ABSTAIN_MESSAGE, reason
    return text, reason
