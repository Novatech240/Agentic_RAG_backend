"""
Reusable agent-turn execution, decoupled from the HTTP layer.

A single turn needs to:
  1. resolve/create a session,
  2. prepend the sliding-window conversation history,
  3. run the RAG agent,
  4. persist the turn back into the Redis-backed session memory.

Keeping it here (instead of in `api.py`) lets non-HTTP callers (e.g. the RAG
evaluation harness in ``scripts/eval_rag.py``) run a turn without importing the
whole FastAPI application.
"""

from __future__ import annotations

import logging
from typing import Optional

from .agent import rag_agent, AgentDependencies
from .citations import ABSTAIN_MESSAGE, enforce as enforce_citations
from .guardrails import check_input, apply_output_guardrails
from .session_memory import memory_manager
from .tools import is_low_confidence, max_cosine_similarity

logger = logging.getLogger(__name__)


async def run_agent_turn(
    message: str,
    session_id: str,
    user_id: Optional[str] = None,
) -> tuple[str, AgentDependencies]:
    """Run one guarded agent turn with memory and return (assistant's reply, deps).

    Applies input guardrails (injection/abuse/length) before the model and
    output guardrails (leak scrub, PII redaction) after.
    """
    session_id = memory_manager.get_or_create(session_id)

    # ── Input guardrails (fail closed) ────────────────────────────────────────
    verdict = check_input(message)
    if not verdict.allowed:
        blocked = (
            verdict.user_message or "Sorry, I can't answer that question."
        )
        memory_manager.add_turn(session_id, message, blocked)
        return blocked, AgentDependencies(session_id=session_id, user_id=user_id)
    safe_message = verdict.sanitized_input or message

    try:
        deps = AgentDependencies(session_id=session_id, user_id=user_id)
        deps.retrieved_chunks = []
        deps.graph_facts = []
        deps.selected_retrieval_tool = None

        history = memory_manager.get_context_string(session_id)
        # Condense follow-ups into a self-contained question so retrieval embeds
        # the full intent, not a context-free fragment ("and the fee?").
        from .query_rewriter import condense

        search_query = await condense(history, safe_message)

        # ── Turn-level answer cache: a hit short-circuits the agent + gates. ──
        from . import answer_cache

        cached = await answer_cache.get(search_query, user_id)
        if cached:
            cached_answer = cached.get("answer", "")
            memory_manager.add_turn(session_id, message, cached_answer)
            return cached_answer, deps

        full_prompt = (
            f"Previous conversation:\n{history}\n\nCurrent question: {search_query}"
            if history
            else search_query
        )
        result = await rag_agent.run(full_prompt, deps=deps)
        # pydantic-ai >=1.0 exposes `.output`; older versions used `.data`.
        response: str = getattr(result, "output", None) or getattr(result, "data", "")

        # ── Anti-hallucination gate (covers BOTH api + worker paths) ──────────
        # Only enforce when the agent actually attempted document retrieval; a
        # pure greeting / scope refusal never hits the knowledge base.
        chunks = deps.retrieved_chunks or []
        cacheable = False  # only a verified-good answer is memoized
        if deps.selected_retrieval_tool == "hybrid_search":
            if is_low_confidence(chunks):
                # Retrieval too weak to ground an answer → abstain rather than
                # risk a fabricated reply.
                logger.info(
                    "Low-confidence retrieval (session=%s, top_cos=%.3f) — abstaining",
                    session_id,
                    max_cosine_similarity(chunks),
                )
                response = ABSTAIN_MESSAGE
            else:
                response, reason = enforce_citations(response, chunks)
                if reason not in {"ok", "exempt_non_answer"}:
                    logger.info(
                        "Citation enforcement tripped (session=%s): %s",
                        session_id,
                        reason,
                    )
                # Tier 4 — groundedness: remediate (strip unsupported claims) or
                # abstain when the cited chunk does not entail the claim.
                elif reason == "ok":
                    from . import groundedness

                    response, gstatus = await groundedness.enforce(response, chunks)
                    if gstatus == groundedness.STATUS_REMEDIATED:
                        logger.info(
                            "Groundedness remediation applied (session=%s)", session_id
                        )
                    elif gstatus == groundedness.STATUS_ABSTAINED:
                        logger.info(
                            "Groundedness gate tripped (session=%s) — abstaining",
                            session_id,
                        )
                    cacheable = gstatus in groundedness.CACHEABLE_STATUSES

        # ── Output guardrails ─────────────────────────────────────────────────
        response = await apply_output_guardrails(response)

        # Memoize verified-good answers (post-guardrail) for the turn cache.
        if cacheable:
            await answer_cache.set(search_query, user_id, response, tools=[])

        memory_manager.add_turn(session_id, message, response)
        return response, deps
    except Exception as exc:
        logger.error("Agent turn failed (session=%s): %s", session_id, exc)
        error_response = "Sorry — there was a problem processing your message. Please try again."
        memory_manager.add_turn(session_id, message, error_response)
        return error_response, AgentDependencies(session_id=session_id, user_id=user_id)
