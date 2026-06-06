"""
Knowledge-graph builder (Neo4j).

For every ingested chunk we build a small, queryable graph:

    (:Entity {name}) -[:HAS_FACT]-> (:Fact {uuid, content}) -[:FROM_CHUNK]-> (:Chunk)

* **:Chunk**  — one node per chunk (kept for provenance + a keyword fallback).
* **:Fact**   — concise standalone statements extracted from the chunk by the LLM.
* **:Entity** — the proper nouns / key concepts a fact is about (programs, fees,
  departments, dates …). Linking entities → facts → entities yields real 2-hop
  relationships, which is what ``search_knowledge_graph`` queries via the
  ``factIndex`` full-text index.

Extraction is gated by ``GRAPH_EXTRACTION_ENABLED`` (default true) and fails
open: if the LLM call or Neo4j write fails, ingestion is never blocked — at
worst the chunk node is still written so the keyword fallback keeps working.
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GRAPH_EXTRACTION_ENABLED = (
    os.getenv("GRAPH_EXTRACTION_ENABLED", "true").lower() == "true"
)
_MAX_FACTS_PER_CHUNK = int(os.getenv("GRAPH_MAX_FACTS_PER_CHUNK", "6"))

# Cached per-process resources (avoids a new Neo4j connection per chunk).
_driver = None
_indexes_ready = False
_openai_client = None


def _get_driver():
    global _driver
    if _driver is not None:
        return _driver
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning("neo4j package not installed — skipping graph building")
        return None
    uri = os.getenv("NEO4J_URI")
    if not uri:
        logger.warning("NEO4J_URI not set — skipping graph building")
        return None
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def _database() -> str:
    return os.getenv("NEO4J_DATABASE", "neo4j")


def _ensure_indexes(session) -> None:
    """Create the full-text + lookup indexes the search layer relies on (idempotent)."""
    global _indexes_ready
    if _indexes_ready:
        return
    session.run(
        "CREATE FULLTEXT INDEX factIndex IF NOT EXISTS FOR (f:Fact) ON EACH [f.content]"
    )
    session.run(
        "CREATE FULLTEXT INDEX chunkIndex IF NOT EXISTS "
        "FOR (c:Chunk) ON EACH [c.content]"
    )
    session.run("CREATE INDEX entityName IF NOT EXISTS FOR (e:Entity) ON (e.name)")
    _indexes_ready = True


def _get_openai():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    from openai import OpenAI

    key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or ""
    )
    if not key:
        return None
    _openai_client = OpenAI(api_key=key)
    return _openai_client


def _extract_facts(content: str) -> List[Dict[str, Any]]:
    """Ask the LLM for concise facts + their entities. Returns [] on any failure."""
    client = _get_openai()
    if client is None:
        return []
    model = os.getenv("LLM_CHOICE", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You build a knowledge graph from organizational documents. "
                        "Extract the key factual statements from the passage. Respond "
                        'ONLY as JSON: {"facts":[{"fact":"<concise standalone fact>",'
                        '"entities":["<proper noun or key concept>"]}]}. '
                        f"At most {_MAX_FACTS_PER_CHUNK} facts. Entities are products, "
                        "services, dates, teams, departments, policies, or named things. "
                        'If there are no clear facts, return {"facts":[]}.'
                    ),
                },
                {"role": "user", "content": content[:4000]},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        facts = data.get("facts") or []
    except Exception as exc:
        logger.warning("Fact extraction failed: %s", exc)
        return []

    cleaned: List[Dict[str, Any]] = []
    for f in facts[:_MAX_FACTS_PER_CHUNK]:
        text = (f.get("fact") or "").strip()
        if not text:
            continue
        ents = [
            e.strip()
            for e in (f.get("entities") or [])
            if isinstance(e, str) and e.strip()
        ]
        cleaned.append({"fact": text, "entities": ents[:8]})
    return cleaned


def build_knowledge_graph(
    chunk: str,
    embedding: List[float],
    doc_key: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one chunk (and its extracted facts/entities) into the graph.

    Fails open: never raises, so a graph hiccup cannot break ingestion.
    """
    driver = _get_driver()
    if driver is None:
        return

    meta = metadata or {}
    access_level = meta.get("access_level", "public")
    content_hash = _hash(chunk)

    try:
        with driver.session(database=_database()) as session:
            _ensure_indexes(session)

            # 1. The chunk node (provenance + keyword fallback).
            session.run(
                """
                MERGE (c:Chunk {doc_key: $doc_key, content_hash: $content_hash})
                SET c.content = $chunk, c.access_level = $access_level
                """,
                doc_key=doc_key,
                content_hash=content_hash,
                chunk=chunk[:2000],
                access_level=access_level,
            )

            # 2. Extracted facts + entities, linked back to the chunk.
            facts = _extract_facts(chunk) if GRAPH_EXTRACTION_ENABLED else []
            if not facts:
                return

            payload = [
                {
                    "uuid": _hash(doc_key + f["fact"]),
                    "fact": f["fact"],
                    "entities": f["entities"],
                }
                for f in facts
            ]
            session.run(
                """
                MATCH (c:Chunk {doc_key: $doc_key, content_hash: $content_hash})
                UNWIND $facts AS fct
                  MERGE (f:Fact {uuid: fct.uuid})
                    SET f.content = fct.fact,
                        f.doc_key = $doc_key,
                        f.access_level = $access_level
                  MERGE (f)-[:FROM_CHUNK]->(c)
                  FOREACH (ename IN fct.entities |
                    MERGE (e:Entity {name: ename})
                    MERGE (e)-[:HAS_FACT]->(f)
                  )
                """,
                doc_key=doc_key,
                content_hash=content_hash,
                access_level=access_level,
                facts=payload,
            )
    except Exception as exc:
        logger.error("Graph building failed for %s: %s", doc_key, exc)


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()
