"""
RAG retrieval evaluation harness.

What it does
------------
1. Loads every chunk from Supabase (the knowledge base).
2. For a sample of chunks, asks the LLM to write a realistic question whose
   answer lives in *that* chunk (synthetic ground-truth Q→chunk pairs).
3. For each question, runs both retrieval legs and checks whether the source
   chunk is retrieved:
     * vector  — pgvector cosine via match_chunks
     * hybrid  — the app's BM25+vector RRF (agent.tools.hybrid_search_tool)
4. Reports recall@1, recall@5, MRR, mean top cosine similarity, and how often
   the confidence gate would (wrongly) suppress a known-relevant answer.

Run (Docker, with the real .env mounted):
    docker run --rm -v "$PWD":/app -w /app uchenab-backend:latest \
        python -m scripts.eval_rag
"""

from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Old (RRF) gate vs new (cosine) gate — see tools.is_low_confidence.
LEGACY_GATE = float(os.getenv("EVAL_LEGACY_GATE", "0.015"))
COSINE_GATE = float(os.getenv("CONFIDENCE_MIN_SIMILARITY", "0.25"))
SAMPLE = int(os.getenv("EVAL_SAMPLE", "0"))  # 0 = all chunks
TOP_K = int(os.getenv("EVAL_TOP_K", "5"))

# Clearly off-domain questions — a good confidence gate should SUPPRESS these.
# These are *corpus-agnostic* noise (cooking / sport / geography / coding):
# whatever domain the indexed documents are in, none of these belong to it, so
# the gate must reject them regardless of what corpus is loaded.
NEGATIVE_QUERIES = [
    "What is the capital of France?",
    "How do I cook chicken biryani at home?",
    "Who won the football world cup in 2018?",
    "Write me a Python function to reverse a linked list.",
    "What's the weather like in Tokyo today?",
    "Give me a recipe for chocolate cake.",
]

# Corpus-agnostic system prompt used ONLY for the end-to-end citation eval. The
# production prompt scopes the assistant to one domain (e.g. university), which
# would make it decline questions about any *other* corpus. For testing we want
# the agent to retrieve+cite from whatever documents are actually indexed, so we
# swap in this neutral prompt — the eval then works for ANY corpus, not just one.
EVAL_SYSTEM_PROMPT = (
    "You are a retrieval QA assistant. For every question, FIRST call the "
    "search_documents tool, then answer strictly from the retrieved chunks, "
    "marking each factual claim with an inline [n] citation referencing the "
    "bracket number of the source chunk. If the retrieved chunks do not contain "
    "the answer, say you don't have that information — never guess."
)


async def _gen_question(client, model: str, content: str) -> str:
    resp = await client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=60,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write ONE short, natural question a university student might "
                    "ask, whose answer is contained in the passage. Output only the "
                    "question, no preamble."
                ),
            },
            {"role": "user", "content": content[:1500]},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _rank_of(target_id: str, rows, id_key) -> int:
    """1-based rank of target_id in rows, or 0 if absent."""
    for i, r in enumerate(rows, 1):
        if str(id_key(r)) == str(target_id):
            return i
    return 0


async def main() -> None:
    from agent.db_utils import initialize_database, vector_search, _client  # noqa
    from agent import db_utils
    from agent.tools import generate_embedding, hybrid_search_tool, HybridSearchInput
    from agent.retriever import hybrid_retriever
    from agent.providers import get_cached_llm, LLM_CHOICE

    await initialize_database()
    # No-op now (Vespa maintains its own indexes); retrieval goes through Vespa
    # with a Postgres fallback. Kept for backward compatibility.
    await hybrid_retriever.build_index()
    client = get_cached_llm()

    rows = (
        await db_utils._client.table("chunks")
        .select("id, content, document_id, documents(title)")
        .execute()
    ).data or []
    rows = [r for r in rows if (r.get("content") or "").strip()]
    if SAMPLE and SAMPLE < len(rows):
        random.seed(42)
        rows = random.sample(rows, SAMPLE)

    print(f"Evaluating against {len(rows)} chunks (top_k={TOP_K})\n")

    vec_hits1 = vec_hits5 = vec_mrr = 0.0
    hyb_hits1 = hyb_hits5 = hyb_mrr = 0.0
    sims: list[float] = []
    hyb_top_scores: list[float] = []
    suppressed = 0
    n = 0

    for r in rows:
        cid = r["id"]
        content = r["content"]
        q = await _gen_question(client, LLM_CHOICE, content)
        if not q:
            continue
        n += 1

        emb = await generate_embedding(q)

        # Vector leg
        vrows = await vector_search(embedding=emb, limit=TOP_K, user_id=None)
        vrank = _rank_of(cid, vrows, lambda x: x["chunk_id"])
        if vrows:
            sims.append(float(vrows[0].get("similarity", 0.0)))
        if vrank == 1:
            vec_hits1 += 1
        if vrank and vrank <= 5:
            vec_hits5 += 1
        if vrank:
            vec_mrr += 1.0 / vrank

        # Hybrid leg (what the agent actually uses)
        hres = await hybrid_search_tool(HybridSearchInput(query=q, limit=TOP_K))
        hrank = _rank_of(cid, hres, lambda x: x.chunk_id)
        top_score = float(hres[0].score) if hres else 0.0
        hyb_top_scores.append(top_score)
        if top_score < LEGACY_GATE:
            suppressed += 1  # known-relevant answer would be gated out
        if hrank == 1:
            hyb_hits1 += 1
        if hrank and hrank <= 5:
            hyb_hits5 += 1
        if hrank:
            hyb_mrr += 1.0 / hrank

    if not n:
        print("No questions generated — aborting.")
        return

    def pct(x):
        return f"{100 * x / n:5.1f}%"

    print("── VECTOR (pgvector cosine) ──")
    print(f"  recall@1 : {pct(vec_hits1)}")
    print(f"  recall@5 : {pct(vec_hits5)}")
    print(f"  MRR      : {vec_mrr / n:.3f}")
    print(f"  mean top cosine similarity : {sum(sims) / max(len(sims), 1):.3f}")
    print()
    print("── HYBRID (BM25 + vector RRF — used by the agent) ──")
    print(f"  recall@1 : {pct(hyb_hits1)}")
    print(f"  recall@5 : {pct(hyb_hits5)}")
    print(f"  MRR      : {hyb_mrr / n:.3f}")
    print(
        f"  hybrid top score range : "
        f"{min(hyb_top_scores):.4f} – {max(hyb_top_scores):.4f}"
    )
    print()
    # ── Negative (off-domain) probes for the confidence gate ──────────────────
    neg_cos: list[float] = []
    neg_rrf: list[float] = []
    for q in NEGATIVE_QUERIES:
        emb = await generate_embedding(q)
        vrows = await vector_search(embedding=emb, limit=TOP_K, user_id=None)
        neg_cos.append(float(vrows[0].get("similarity", 0.0)) if vrows else 0.0)
        hres = await hybrid_search_tool(HybridSearchInput(query=q, limit=TOP_K))
        neg_rrf.append(float(hres[0].score) if hres else 0.0)

    # Gate confusion: positives should PASS, negatives should be SUPPRESSED.
    pos_cos_suppressed = sum(1 for s in sims if s < COSINE_GATE)
    pos_rrf_suppressed = suppressed
    neg_cos_suppressed = sum(1 for s in neg_cos if s < COSINE_GATE)
    neg_rrf_suppressed = sum(1 for s in neg_rrf if s < LEGACY_GATE)
    nn = len(NEGATIVE_QUERIES)

    print(
        "── CONFIDENCE GATE: old RRF (<%.3f) vs new cosine (<%.2f) ──"
        % (LEGACY_GATE, COSINE_GATE)
    )
    print(
        f"  cosine range — relevant: {min(sims):.3f}-{max(sims):.3f} | "
        f"off-domain: {min(neg_cos):.3f}-{max(neg_cos):.3f}"
    )
    print(
        f"  RRF range    — relevant: {min(hyb_top_scores):.4f}-{max(hyb_top_scores):.4f} | "
        f"off-domain: {min(neg_rrf):.4f}-{max(neg_rrf):.4f}"
    )
    print()
    print("  RELEVANT wrongly suppressed (lower=better):")
    print(f"    old RRF gate    : {pos_rrf_suppressed}/{n}")
    print(f"    new cosine gate : {pos_cos_suppressed}/{n}")
    print("  OFF-DOMAIN correctly suppressed (higher=better):")
    print(f"    old RRF gate    : {neg_rrf_suppressed}/{nn}")
    print(f"    new cosine gate : {neg_cos_suppressed}/{nn}")
    print()

    # ── Gate threshold sweep — pick the cosine cutoff that maximizes accuracy ──
    # Optimal = keeps the most relevant answers while suppressing the most
    # off-domain noise. We score each candidate threshold by Youden's J
    # (TPR - FPR): pass-relevant rate minus pass-off-domain rate.
    print("── GATE THRESHOLD SWEEP (cosine) — maximize (keep relevant − keep noise) ──")
    print("  thresh | relevant kept | off-domain kept | J (higher=better)")
    best_t, best_j = COSINE_GATE, -1.0
    for t in [round(0.10 + 0.025 * i, 3) for i in range(21)]:  # 0.10 … 0.60
        rel_kept = sum(1 for s in sims if s >= t) / max(len(sims), 1)
        neg_kept = sum(1 for s in neg_cos if s >= t) / max(len(neg_cos), 1)
        j = rel_kept - neg_kept
        if j > best_j:
            best_j, best_t = j, t
        marker = "  <-- current" if abs(t - COSINE_GATE) < 1e-6 else ""
        print(f"   {t:.3f} |    {rel_kept:5.1%}     |     {neg_kept:5.1%}     | {j:+.3f}{marker}")
    print(
        f"\n  >> Recommended CONFIDENCE_MIN_SIMILARITY = {best_t:.3f} "
        f"(J={best_j:+.3f}); currently {COSINE_GATE:.3f}"
    )

    # ── Optional end-to-end anti-hallucination eval (EVAL_CITATIONS=1) ─────────
    # Runs the FULL agent turn (retrieve → rerank → generate → gate → citation
    # validator) and measures: positives should produce a cited, non-abstaining
    # answer; off-domain should ABSTAIN. This is the real "zero hallucination"
    # signal — does a fabricated/uncited answer ever slip through?
    if os.getenv("EVAL_CITATIONS", "0") in {"1", "true", "yes"}:
        print("\n── END-TO-END CITATION / ABSTENTION FAITHFULNESS ──")
        from unittest.mock import patch

        from agent.conversation import run_agent_turn
        from agent.citations import extract_citations, ABSTAIN_MESSAGE

        async def _eval_prompt() -> str:
            return EVAL_SYSTEM_PROMPT

        # Swap the domain-scoped production prompt for a corpus-agnostic one so
        # the agent retrieves+cites whatever is indexed (works for any corpus).
        with patch("agent.settings_store.get_system_prompt", _eval_prompt):
            cite_sample = rows[: min(len(rows), int(os.getenv("EVAL_CITE_SAMPLE", "10")))]
            pos_cited = pos_abstained = 0
            for i, r in enumerate(cite_sample):
                q = await _gen_question(client, LLM_CHOICE, r["content"])
                if not q:
                    continue
                ans, _ = await run_agent_turn(q, session_id=f"eval-pos-{i}")
                if ans.strip() == ABSTAIN_MESSAGE.strip() or "don't have" in ans.lower():
                    pos_abstained += 1
                elif extract_citations(ans):
                    pos_cited += 1
            neg_abstained = 0
            for i, q in enumerate(NEGATIVE_QUERIES):
                ans, _ = await run_agent_turn(q, session_id=f"eval-neg-{i}")
                low = ans.lower()
                if (
                    ans.strip() == ABSTAIN_MESSAGE.strip()
                    or "don't have" in low
                    or "scope" in low
                    or "couldn't find" in low
                ):
                    neg_abstained += 1
        ns = len(cite_sample)
        print(f"  RELEVANT  → cited answer     : {pos_cited}/{ns}")
        print(f"  RELEVANT  → abstained        : {pos_abstained}/{ns} (false negatives)")
        print(f"  OFF-DOMAIN → abstained       : {neg_abstained}/{len(NEGATIVE_QUERIES)} (should be all)")

    await db_utils.close_database()


if __name__ == "__main__":
    asyncio.run(main())
