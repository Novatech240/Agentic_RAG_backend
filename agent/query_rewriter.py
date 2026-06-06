"""
Conversational query rewriting — the multi-turn recall lever.

Retrieval embeds the user's *current* turn. In a conversation that turn is often
a fragment that only makes sense against the history:

    User:      What are the admission requirements for Computer Science?
    Assistant: ... (answers) ...
    User:      And the fee?            <-- embeds to almost nothing useful

Embedding "And the fee?" retrieves poorly because the subject (CS admissions) is
gone. This module condenses (history + follow-up) into a single self-contained
question — "What is the admission fee for Computer Science?" — *before* the
embedding/Vespa/rerank stack runs, so retrieval sees a complete query.

It is:
* **Cheap & bounded.** One short, temperature-0 completion capped at a few dozen
  tokens, and only when history actually exists. First turns skip it entirely.
* **Conservative.** The prompt forbids inventing facts; it may only resolve
  pronouns/ellipsis using the provided history. If the question is already
  self-contained the model returns it unchanged.
* **Fail-open.** Disabled flag, no history, empty/oversized input, or any LLM
  error → the original question is used verbatim. Rewriting can sharpen
  retrieval but can never block a turn.

Env
    QUERY_REWRITE_ENABLED   master switch (default "true")
    QUERY_REWRITE_MAXLEN    skip rewrite above this char length (default 400)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
}
_MAXLEN = int(os.getenv("QUERY_REWRITE_MAXLEN", "400"))

_SYSTEM = (
    "You rewrite a user's latest message into a single, self-contained search "
    "query using the conversation history ONLY to resolve pronouns, ellipsis, "
    "and implicit subjects (e.g. 'and the fee?' -> 'what is the admission fee "
    "for Computer Science?'). Rules: (1) Output ONLY the rewritten query, no "
    "preamble. (2) Never add facts, entities, or constraints not present in the "
    "history or the message. (3) If the message is already self-contained, "
    "return it unchanged. (4) Keep it concise and in the user's language."
)


def rewrite_enabled() -> bool:
    return _ENABLED


async def condense(history: str, question: str) -> str:
    """Return a standalone version of ``question`` resolved against ``history``.

    Fail-open: returns ``question`` unchanged on disable / no history / error.
    """
    q = (question or "").strip()
    if not (_ENABLED and history and q):
        return q
    # Long messages are almost always already self-contained — skip the call.
    if len(q) > _MAXLEN:
        return q

    try:
        from .providers import get_cached_llm, LLM_CHOICE

        client = get_cached_llm()
        resp = await client.chat.completions.create(
            model=LLM_CHOICE,
            temperature=0,
            max_tokens=80,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": f"Conversation so far:\n{history}\n\nLatest message: {q}\n\nRewritten query:",
                },
            ],
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        # Guard against the model echoing the instruction or returning nothing.
        if rewritten and len(rewritten) <= _MAXLEN * 2:
            if rewritten.lower() != q.lower():
                logger.info("Query rewrite: %r -> %r", q[:60], rewritten[:60])
            return rewritten
    except Exception as exc:  # fail-open — original query is always safe
        logger.warning("Query rewrite failed (using original): %s", exc)
    return q
