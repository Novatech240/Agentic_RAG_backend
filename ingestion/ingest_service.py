"""
IngestService — the single source of truth for document ingestion.

Key behaviours
--------------
* **Upsert / deduplication**: every document is keyed by its ``source``.
  A SHA-256 of the full text detects whether anything changed at all.
  If the document changed, **chunk-level diffing** is applied:

  ┌──────────────────┬─────────────────┬──────────────────────────────────────┐
  │ source exists?   │ doc hash chgd?  │ Action                               │
  ├──────────────────┼─────────────────┼──────────────────────────────────────┤
  │ No               │ —               │ Insert doc + embed all chunks        │
  │ Yes              │ No              │ Skip entirely                        │
  │ Yes              │ Yes             │ Chunk-level diff:                    │
  │                  │                 │  • keep chunks whose hash unchanged  │
  │                  │                 │  • delete chunks that disappeared    │
  │                  │                 │  • embed only new / changed chunks   │
  └──────────────────┴─────────────────┴──────────────────────────────────────┘

  Example: policy.pdf has 50 chunks, 3 sentences change → only 2-3 chunks
  are re-embedded.  The other 47 are left untouched.

* **Knowledge-graph upsert**: Neo4j MERGE on chunk nodes so re-ingestion
  updates existing nodes instead of creating duplicates.

* **Vespa feed**: after every successful ingest the newly saved chunks are
  fed to Vespa (the primary retrieval index); stale chunks are removed from
  Vespa too. Postgres remains the durable source of truth / fallback.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of a single document ingestion."""

    source: str
    status: str  # "inserted" | "updated" | "skipped" | "failed"
    document_id: Optional[str] = None
    chunks_created: int = 0
    error: Optional[str] = None


@dataclass
class IngestStats:
    """Aggregate statistics for a bulk ingestion run."""

    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    results: List[IngestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.inserted + self.updated + self.skipped + self.failed


class IngestService:
    """
    Orchestrates the full ingestion pipeline with upsert semantics.

    Dependencies (injected via constructor for testability):
    - ``chunker``  : SemanticChunker or SimpleChunker instance
    - ``embedder`` : EmbeddingGenerator instance
    """

    def __init__(self, chunker=None, embedder=None) -> None:
        self._chunker = chunker
        self._embedder = embedder

    # ── Public API ────────────────────────────────────────────────────────────

    async def ingest_document(
        self,
        content: str,
        source: str,
        title: str,
        metadata: Optional[Dict[str, Any]] = None,
        access_level: str = "public",
    ) -> IngestResult:
        """
        Ingest a single document with upsert semantics.

        Parameters
        ----------
        content : str
            Full text content of the document.
        source : str
            Unique document identifier (e.g. S3 key or file path).
        title : str
            Human-readable document title.
        metadata : dict, optional
            Arbitrary metadata stored alongside the document.
        access_level : str
            ``"public"`` or ``"private"``.

        Returns
        -------
        IngestResult
            Status and statistics for this document.
        """
        from agent.db_utils import (
            get_document_by_source,
            get_document_chunks,
            upsert_document,
            save_chunks,
            delete_chunks_by_ids,
        )
        from ingestion.graph_builder import build_knowledge_graph

        meta = metadata or {}
        content_hash = _sha256(content)

        try:
            existing = await get_document_by_source(source)
            is_update = existing is not None

            # ── Fast path: nothing changed ────────────────────────────────────
            if is_update and existing["content_hash"] == content_hash:
                logger.info("Skipping '%s' — content unchanged", source)
                return IngestResult(
                    source=source,
                    status="skipped",
                    document_id=existing["id"],
                )

            # ── Chunk the new content ─────────────────────────────────────────
            chunker = self._get_chunker()
            embedder = self._get_embedder()

            chunks = await chunker.chunk_document(
                content=content, title=title, source=source, metadata=meta
            )

            # Build a hash → chunk map for the new version
            new_chunk_hashes = {_sha256(c.content): c for c in chunks}

            if is_update:
                # ── Chunk-level diff ──────────────────────────────────────────
                doc_id = existing["id"]
                old_chunks = await get_document_chunks(doc_id)
                old_hashes = {
                    r.get("content_hash"): r
                    for r in old_chunks
                    if r.get("content_hash")
                }

                # Chunks to delete: exist in DB but not in new version
                stale_ids = [
                    r["id"]
                    for r in old_chunks
                    if r.get("content_hash") not in new_chunk_hashes
                ]
                # Chunks to embed: new or changed (not in old DB)
                chunks_to_embed = [
                    c for h, c in new_chunk_hashes.items() if h not in old_hashes
                ]

                logger.info(
                    "Updating '%s' — %d unchanged, %d to delete, %d to embed",
                    source,
                    len(chunks) - len(chunks_to_embed),
                    len(stale_ids),
                    len(chunks_to_embed),
                )

                # Delete only the stale chunks (Postgres + Vespa)
                if stale_ids:
                    await delete_chunks_by_ids(stale_ids)
                    try:
                        from agent.vespa_client import delete_chunks, vespa_enabled

                        if vespa_enabled():
                            await delete_chunks([str(i) for i in stale_ids])
                    except Exception as exc:
                        logger.warning("Vespa stale-chunk delete skipped: %s", exc)

                # Update document metadata / hash
                await upsert_document(
                    source=source,
                    title=title,
                    content=content,
                    content_hash=content_hash,
                    metadata=meta,
                    access_level=access_level,
                )

                if not chunks_to_embed:
                    # All chunks identical — only metadata changed
                    return IngestResult(
                        source=source,
                        status="updated",
                        document_id=doc_id,
                        chunks_created=0,
                    )

                embedded_new = await embedder.embed_chunks(chunks_to_embed)
                saved_rows = await save_chunks(doc_id, embedded_new)
                chunks_saved = embedded_new

            else:
                # ── Brand-new document ────────────────────────────────────────
                doc_id = await upsert_document(
                    source=source,
                    title=title,
                    content=content,
                    content_hash=content_hash,
                    metadata=meta,
                    access_level=access_level,
                )
                embedded_new = await embedder.embed_chunks(chunks)
                saved_rows = await save_chunks(doc_id, embedded_new)
                chunks_saved = embedded_new

            # Knowledge-graph upsert for newly embedded chunks
            for chunk in chunks_saved:
                build_knowledge_graph(
                    chunk=chunk.content,
                    embedding=getattr(chunk, "embedding", []),
                    doc_key=source,
                    metadata={**meta, "access_level": access_level, "doc_id": doc_id},
                )

            # Feed the newly saved chunks to Vespa (primary retrieval index).
            # Postgres already holds them as the durable source of truth.
            try:
                from agent.vespa_client import feed_chunks, vespa_enabled

                if vespa_enabled() and saved_rows:
                    for r in saved_rows:
                        r["document_title"] = title
                        r["document_source"] = source
                        r["access_level"] = access_level
                    await feed_chunks(saved_rows)
            except Exception as exc:
                logger.warning("Vespa feed skipped: %s", exc)

            # Corpus changed — invalidate the retrieval cache so newly indexed
            # chunks are never masked by a stale cached result (fail-open no-op).
            try:
                from agent.query_cache import bump_version

                bump_version()
            except Exception as exc:
                logger.debug("RAG cache invalidation skipped: %s", exc)

            status = "updated" if is_update else "inserted"
            logger.info(
                "%s '%s' → %d new chunks [%s]",
                status.capitalize(),
                source,
                len(chunks_saved),
                access_level,
            )
            return IngestResult(
                source=source,
                status=status,
                document_id=doc_id,
                chunks_created=len(chunks_saved),
            )

        except Exception as exc:
            logger.error("Ingestion failed for '%s': %s", source, exc)
            return IngestResult(source=source, status="failed", error=str(exc))

    async def ingest_all_s3_buckets(self) -> IngestStats:
        """Ingest both private and public S3 buckets and return combined stats."""
        stats = IngestStats()
        for bucket_type in ("private", "public"):
            bucket_stats = await self.ingest_from_s3(bucket_type=bucket_type)
            stats.inserted += bucket_stats.inserted
            stats.updated += bucket_stats.updated
            stats.skipped += bucket_stats.skipped
            stats.failed += bucket_stats.failed
            stats.results.extend(bucket_stats.results)
        return stats

    async def ingest_from_s3(
        self,
        bucket_type: str = "private",
        prefix: str = "",
    ) -> IngestStats:
        """Download and ingest all supported documents from an S3 bucket."""
        from ingestion.s3_utils import (
            list_documents_from_s3,
            download_document_from_s3,
            verify_s3_access,
        )
        from ingestion.file_parsers import parse_document
        from agent.access_control import assign_access_level

        stats = IngestStats()

        if not verify_s3_access():
            logger.error("S3 access verification failed")
            return stats

        doc_keys = list_documents_from_s3(bucket_type=bucket_type, prefix=prefix)
        if not doc_keys:
            logger.warning("No documents found in '%s' bucket", bucket_type)
            return stats

        access_level = assign_access_level(bucket_type)

        with tempfile.TemporaryDirectory() as tmp_dir:
            for doc_key in doc_keys:
                local_path = str(Path(tmp_dir) / Path(doc_key).name)
                content_bytes = download_document_from_s3(
                    doc_key, bucket_type=bucket_type, save_path=local_path
                )
                if content_bytes is None:
                    stats.failed += 1
                    stats.results.append(
                        IngestResult(
                            source=doc_key, status="failed", error="Download failed"
                        )
                    )
                    continue

                parsed_content, file_meta = parse_document(local_path)
                if not parsed_content:
                    stats.failed += 1
                    stats.results.append(
                        IngestResult(
                            source=doc_key, status="failed", error="Empty content"
                        )
                    )
                    continue

                result = await self.ingest_document(
                    content=parsed_content,
                    source=doc_key,
                    title=Path(doc_key).stem,
                    metadata={
                        **file_meta,
                        "bucket_type": bucket_type,
                        "s3_key": doc_key,
                    },
                    access_level=access_level,
                )
                stats.results.append(result)
                if result.status == "inserted":
                    stats.inserted += 1
                elif result.status == "updated":
                    stats.updated += 1
                elif result.status == "skipped":
                    stats.skipped += 1
                else:
                    stats.failed += 1

        logger.info(
            "S3 ingest '%s' complete — inserted=%d updated=%d skipped=%d failed=%d",
            bucket_type,
            stats.inserted,
            stats.updated,
            stats.skipped,
            stats.failed,
        )
        return stats

    async def ingest_single_s3_object(
        self, s3_key: str, bucket_name: str
    ) -> IngestResult:
        """Ingest a single S3 object (used by webhook handler)."""
        from ingestion.s3_utils import download_document_from_s3
        from ingestion.file_parsers import parse_document
        from agent.access_control import assign_access_level

        private_bucket = os.getenv("S3_PRIVATE_BUCKET", "")
        bucket_type = "private" if bucket_name == private_bucket else "public"
        access_level = assign_access_level(bucket_type)

        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = str(Path(tmp_dir) / Path(s3_key).name)
            content_bytes = download_document_from_s3(
                s3_key, bucket_type=bucket_type, save_path=local_path
            )
            if content_bytes is None:
                return IngestResult(
                    source=s3_key, status="failed", error="Download failed"
                )

            parsed_content, file_meta = parse_document(local_path)
            if not parsed_content:
                return IngestResult(
                    source=s3_key, status="failed", error="Empty content"
                )

            return await self.ingest_document(
                content=parsed_content,
                source=s3_key,
                title=Path(s3_key).stem,
                metadata={**file_meta, "bucket_type": bucket_type, "s3_key": s3_key},
                access_level=access_level,
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_chunker(self):
        if self._chunker:
            return self._chunker
        from ingestion.chunker import ChunkingConfig, create_chunker

        return create_chunker(ChunkingConfig())

    def _get_embedder(self):
        if self._embedder:
            return self._embedder
        from ingestion.embedder import create_embedder

        return create_embedder()


# ── Module-level singleton ────────────────────────────────────────────────────

ingest_service = IngestService()


# ── Utilities ─────────────────────────────────────────────────────────────────


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
