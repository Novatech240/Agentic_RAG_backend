"""
Backfill / rebuild the Neo4j knowledge graph from the chunks already in Supabase.

Clears the existing graph, then re-runs entity/fact extraction for every stored
chunk so :Entity/:Fact nodes + relationships exist for data ingested before the
graph builder was reconciled. Also removes the stale-node drift.

Run (Docker, real .env mounted):
    docker run --rm -v "$PWD":/app -w /app uchenab-backend:latest \
        python scripts/backfill_graph.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


async def main() -> None:
    from agent.db_utils import initialize_database, close_database
    from agent import db_utils
    from ingestion.graph_builder import build_knowledge_graph, _get_driver, _database

    await initialize_database()

    rows = (
        await db_utils._client.table("chunks")
        .select("content, documents(source, access_level)")
        .execute()
    ).data or []
    print(f"Loaded {len(rows)} chunks from Supabase")

    driver = _get_driver()
    if driver is None:
        print("Neo4j not configured — aborting")
        return

    # Clean slate (removes stale drift + any old bare :Chunk nodes).
    with driver.session(database=_database()) as s:
        s.run("MATCH (n) DETACH DELETE n")
    print("Cleared existing graph")

    built = 0
    for r in rows:
        content = (r.get("content") or "").strip()
        doc = r.get("documents") or {}
        source = doc.get("source")
        if not content or not source:
            continue
        build_knowledge_graph(
            chunk=content,
            embedding=[],
            doc_key=source,
            metadata={"access_level": doc.get("access_level", "public")},
        )
        built += 1
    print(f"Rebuilt graph for {built} chunks")

    # Report.
    with driver.session(database=_database()) as s:
        for label in ("Chunk", "Fact", "Entity"):
            c = s.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
            print(f"  :{label} nodes -> {c}")
        rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        print(f"  relationships -> {rels}")
        sample = s.run(
            "MATCH (e:Entity)-[:HAS_FACT]->(f:Fact) "
            "RETURN e.name AS entity, f.content AS fact LIMIT 5"
        ).data()
        print("  sample entity→fact:")
        for row in sample:
            print(f"    [{row['entity']}] {row['fact'][:90]}")

    await close_database()


if __name__ == "__main__":
    asyncio.run(main())
