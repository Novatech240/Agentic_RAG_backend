"""
Neo4j knowledge graph utilities.
"""

import os
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_driver = None


async def initialize_graph() -> None:
    """Open the Neo4j driver for knowledge-graph search."""
    global _driver

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not uri:
        logger.warning("NEO4J_URI not set — graph database disabled")
        return

    try:
        from neo4j import AsyncGraphDatabase

        _driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4j graph database initialised (%s)", database)
    except ImportError:
        logger.warning("neo4j package not installed — graph database disabled")
    except Exception as exc:
        logger.error("Failed to initialise Neo4j: %s", exc)


async def close_graph() -> None:
    """Close the Neo4j driver."""
    global _driver
    if _driver:
        await _driver.close()
        _driver = None
        logger.info("Neo4j connection closed")


async def test_graph_connection() -> bool:
    """Return True if Neo4j is reachable."""
    if not _driver:
        return False
    try:
        async with _driver.session(
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        ) as session:
            await session.run("RETURN 1")
        return True
    except Exception as exc:
        logger.error("Neo4j connection test failed: %s", exc)
        return False


def _lucene_sanitize(query: str) -> str:
    """Neutralise Lucene operators so a user query can't break the full-text call."""
    import re

    cleaned = re.sub(r'[+\-!(){}\[\]^"~*?:\\/]', " ", query or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


_FACT_CYPHER = """
CALL db.index.fulltext.queryNodes('factIndex', $query)
YIELD node, score
WHERE node:Fact
RETURN node.content AS fact, node.uuid AS uuid,
       node.valid_at AS valid_at, node.invalid_at AS invalid_at,
       node.source_node_uuid AS source_node_uuid
ORDER BY score DESC
LIMIT 20
"""

# Fallback: if no extracted facts match, search the raw chunk text so the graph
# leg still contributes something rather than silently returning nothing.
_CHUNK_CYPHER = """
CALL db.index.fulltext.queryNodes('chunkIndex', $query)
YIELD node, score
WHERE node:Chunk
RETURN node.content AS fact, node.content_hash AS uuid,
       null AS valid_at, null AS invalid_at, null AS source_node_uuid
ORDER BY score DESC
LIMIT 10
"""


async def search_knowledge_graph(query: str) -> List[Dict[str, Any]]:
    """Full-text search over extracted facts, falling back to raw chunk text."""
    if not _driver:
        return []

    safe = _lucene_sanitize(query)
    if not safe:
        return []

    try:
        async with _driver.session(
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        ) as session:
            # Parameters go in a dict — `run()`'s first positional arg is itself
            # named `query`, so a `query=` kwarg would collide with the Cypher.
            result = await session.run(_FACT_CYPHER, {"query": safe})
            records = await result.data()
            if records:
                return records
            # No facts matched — try the raw-chunk fallback index.
            result = await session.run(_CHUNK_CYPHER, {"query": safe})
            return await result.data()
    except Exception as exc:
        logger.error("Knowledge graph search failed: %s", exc)
        return []
