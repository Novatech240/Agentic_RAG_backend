"""
Backfill Vespa from Postgres.

Feeds every chunk already stored in Postgres into Vespa — using the existing
embeddings (no re-embedding). Run once when first standing up Vespa, or any time
you need to rebuild the Vespa index from the Postgres source of truth.

Usage:
    python -m scripts.backfill_vespa
    python -m scripts.backfill_vespa --batch 500
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backfill_vespa")


async def backfill(batch: int) -> int:
    from agent import db_utils, vespa_client

    if not vespa_client.vespa_enabled():
        logger.error("VESPA_ENABLED is false — nothing to do.")
        return 0

    await db_utils.initialize_database()
    client = db_utils._client

    total_fed = 0
    offset = 0
    while True:
        result = await (
            client.table("chunks")
            .select(
                "id, document_id, content, embedding, chunk_index, metadata, "
                "documents(title, source, access_level)"
            )
            .order("id")
            .range(offset, offset + batch - 1)
            .execute()
        )
        rows = result.data or []
        if not rows:
            break

        records = []
        for r in rows:
            doc = r.pop("documents", None) or {}
            records.append(
                {
                    "id": r["id"],
                    "document_id": r["document_id"],
                    "content": r.get("content", ""),
                    "embedding": r.get("embedding"),
                    "chunk_index": r.get("chunk_index", 0),
                    "metadata": r.get("metadata", {}),
                    "document_title": doc.get("title", ""),
                    "document_source": doc.get("source", ""),
                    "access_level": doc.get("access_level", "public"),
                }
            )

        fed = await vespa_client.feed_chunks(records)
        total_fed += fed
        logger.info("Backfilled %d chunks (offset %d)", total_fed, offset)
        offset += batch

    await db_utils.close_database()
    logger.info("Done — %d chunks fed to Vespa.", total_fed)
    return total_fed


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Vespa from Postgres chunks.")
    parser.add_argument("--batch", type=int, default=500, help="rows per page")
    args = parser.parse_args()
    asyncio.run(backfill(args.batch))


if __name__ == "__main__":
    main()
