"""
Groundedness verification + claim-level remediation — the semantic backstop
against hallucination.

The pipeline already has three anti-hallucination tiers:
  * Tier 2 (confidence gate): is the *retrieval* strong enough? (cosine >= 0.25)
  * Tier 3 (citation enforcement): does the answer *cite* real chunks? (regex)

Both are necessary but neither checks the thing that actually matters: does the
cited chunk **entail** the claim the answer makes? A model can confidently cite
``[2]`` for a number that chunk [2] never states. Regex sees a valid ``[2]`` and
passes it. This module closes that gap with an LLM NLI / faithfulness judge that
reads the answer and the retrieved chunks and decides, per claim, whether each is
supported.

Two responses to an ungrounded answer:
  * **Remediation (default):** strip only the unsupported claims and keep the
    grounded remainder — e.g. drop a fabricated "Google Wallet" from an otherwise
    correct list rather than discarding the whole answer. Far better UX than a
    blanket abstention while remaining strictly grounded.
  * **Abstain:** if nothing survives remediation (or remediation is disabled),
    swap the whole answer for the abstain message — an honest "I don't know".

This is the "agentic verification" step and the highest-leverage move toward
*zero* hallucination. It is the main added per-turn cost (one short LLM call on
answered turns) and is independently gated and **fail-open**: any judge error
keeps the original answer (the cheaper citation gate still applies upstream).

Env
    GROUNDEDNESS_CHECK_ENABLED        master switch (default "true")
    GROUNDEDNESS_REMEDIATION_ENABLED  strip-and-keep vs abstain-only (default "true")
    GROUNDEDNESS_REVERIFY             re-judge the remediated answer (default "false")
    GROUNDEDNESS_MODEL                judge model (default = LLM_CHOICE)
    GROUNDEDNESS_MIN_WORDS            skip short/greeting answers (default 12)
    GROUNDEDNESS_MAX_CHUNKS           evidence chunks to show the judge (default 8)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from .models import ChunkResult

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("GROUNDEDNESS_CHECK_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
_REMEDIATION = os.getenv(
    "GROUNDEDNESS_REMEDIATION_ENABLED", "true"
).strip().lower() in {"1", "true", "yes"}
_REVERIFY = os.getenv("GROUNDEDNESS_REVERIFY", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
_MIN_WORDS = int(os.getenv("GROUNDEDNESS_MIN_WORDS", "12"))
_MAX_CHUNKS = int(os.getenv("GROUNDEDNESS_MAX_CHUNKS", "8"))

# Statuses returned by ``enforce`` (also persisted in message provenance).
STATUS_GROUNDED = "grounded"
STATUS_REMEDIATED = "remediated"
STATUS_ABSTAINED = "ungrounded_abstained"
STATUS_SKIPPED = "skipped"
STATUS_ERROR = "error_failopen"

# Statuses whose answer is safe to memoize in the turn-level answer cache.
CACHEABLE_STATUSES = {STATUS_GROUNDED, STATUS_REMEDIATED, STATUS_SKIPPED}

_SYSTEM = (
    "You are a strict groundedness judge and editor for a retrieval assistant. "
    "Given an ANSWER and the numbered SOURCES it was supposed to be based on, "
    "decide whether EVERY factual claim in the answer is directly supported by "
    "the sources. Numbers, dates, names, amounts, eligibility rules, and list "
    "items must match the sources exactly; an item not present in the sources is "
    "UNSUPPORTED even if plausible. General conversational framing (greetings, "
    "'let me help', offers to clarify) needs no support.\n"
    "Then produce a CORRECTED answer that keeps every supported claim verbatim "
    "(preserving its inline [n] citations) and removes ONLY the unsupported "
    "claims, leaving natural, coherent text. If removing them leaves no "
    "substantive content, set corrected to an empty string.\n"
    'Respond ONLY as JSON: {"grounded": true|false, "reason": "<short>", '
    '"unsupported_claims": ["<claim>", ...], "corrected": "<edited answer or \'\'>"}'
)


@dataclass
class GroundednessVerdict:
    """Structured judge output. ``corrected`` is the answer with unsupported
    claims removed (``""`` when nothing substantive survives)."""

    grounded: bool
    reason: str
    unsupported_claims: List[str] = field(default_factory=list)
    corrected: str = ""


def groundedness_enabled() -> bool:
    return _ENABLED


def _is_substantive(text: str) -> bool:
    return len(re.findall(r"\w+", text or "")) >= _MIN_WORDS


async def assess(answer: str, chunks: List[ChunkResult]) -> GroundednessVerdict:
    """Judge whether ``answer`` is entailed by ``chunks`` and propose a correction.

    Fail-open: returns ``grounded=True`` (reason ``skipped``/``error_failopen``)
    whenever the check is disabled, not applicable, or errors, so a good answer
    is never dropped because the judge was unavailable.
    """
    if not (_ENABLED and chunks and _is_substantive(answer)):
        return GroundednessVerdict(True, STATUS_SKIPPED)

    sources = "\n\n".join(
        f"[{i}] {c.content[:700]}" for i, c in enumerate(chunks[:_MAX_CHUNKS], 1)
    )
    try:
        from .providers import get_cached_llm, LLM_CHOICE

        model = os.getenv("GROUNDEDNESS_MODEL", LLM_CHOICE)
        client = get_cached_llm()
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"SOURCES:\n{sources}\n\nANSWER:\n{answer}"},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        grounded = bool(data.get("grounded", True))
        reason = str(data.get("reason", ""))[:200]
        claims = [str(c)[:300] for c in (data.get("unsupported_claims") or []) if c]
        corrected = str(data.get("corrected", "") or "").strip()
        if not grounded:
            logger.info("Groundedness judge: UNGROUNDED — %s", reason)
        return GroundednessVerdict(grounded, reason or "grounded", claims, corrected)
    except Exception as exc:  # fail-open: never drop a good answer on judge error
        logger.warning("Groundedness check failed (passing through): %s", exc)
        return GroundednessVerdict(True, STATUS_ERROR)


async def is_grounded(answer: str, chunks: List[ChunkResult]) -> Tuple[bool, str]:
    """Back-compat boolean wrapper around :func:`assess` (``(grounded, reason)``)."""
    verdict = await assess(answer, chunks)
    return verdict.grounded, verdict.reason


async def enforce(answer: str, chunks: List[ChunkResult]) -> Tuple[str, str]:
    """Apply the groundedness policy, returning ``(final_answer, status)``.

    * grounded answer → returned unchanged (status ``grounded``).
    * ungrounded + remediation salvages a substantive, still-cited remainder →
      the corrected answer (status ``remediated``).
    * otherwise → the abstain message (status ``ungrounded_abstained``).

    Disabled / not-applicable / judge error all pass the answer through
    unchanged (status ``skipped`` / ``error_failopen``).
    """
    from .citations import ABSTAIN_MESSAGE, validate_citations

    verdict = await assess(answer, chunks)
    if verdict.grounded:
        return answer, verdict.reason if verdict.reason in {
            STATUS_SKIPPED,
            STATUS_ERROR,
        } else STATUS_GROUNDED

    # Ungrounded — try claim-level remediation before abstaining.
    if _REMEDIATION:
        corrected = (verdict.corrected or "").strip()
        from .citations import _looks_like_abstention  # local import: same package

        # Accept the salvaged answer only if it still carries a real, in-range
        # citation (cite reason "ok") — that, not word count, is what proves a
        # grounded remainder survived (a short cited list is a fine answer).
        _, cite_reason = validate_citations(corrected, len(chunks))
        if (
            corrected
            and cite_reason == "ok"
            and not _looks_like_abstention(corrected)
        ):
            # Optionally re-judge the salvaged answer before trusting it.
            if _REVERIFY:
                recheck = await assess(corrected, chunks)
                if not recheck.grounded:
                    logger.info("Remediation re-verify failed — abstaining")
                    return ABSTAIN_MESSAGE, STATUS_ABSTAINED
            logger.info(
                "Groundedness remediation: removed %d unsupported claim(s)",
                len(verdict.unsupported_claims),
            )
            return corrected, STATUS_REMEDIATED

    return ABSTAIN_MESSAGE, STATUS_ABSTAINED
