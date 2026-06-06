"""
Unit tests for IngestService (upsert / deduplication pipeline).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ingestion.ingest_service import IngestResult, IngestService, IngestStats, _sha256


# ── _sha256 utility ───────────────────────────────────────────────────────────


class TestSha256:
    def test_returns_64_char_hex(self):
        h = _sha256("hello world")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_input_same_hash(self):
        assert _sha256("test") == _sha256("test")

    def test_different_inputs_different_hashes(self):
        assert _sha256("foo") != _sha256("bar")

    def test_empty_string_hashed(self):
        h = _sha256("")
        assert len(h) == 64


# ── IngestResult / IngestStats dataclasses ────────────────────────────────────


class TestIngestResult:
    def test_failed_result_has_error(self):
        r = IngestResult(source="s3://bucket/doc.pdf", status="failed", error="timeout")
        assert r.error == "timeout"
        assert r.chunks_created == 0

    def test_success_result(self):
        r = IngestResult(
            source="doc.pdf", status="inserted", document_id="uuid-1", chunks_created=5
        )
        assert r.status == "inserted"
        assert r.chunks_created == 5


class TestIngestStats:
    def test_total_sums_all_buckets(self):
        stats = IngestStats(inserted=3, updated=2, skipped=1, failed=1)
        assert stats.total == 7

    def test_empty_stats_total_zero(self):
        assert IngestStats().total == 0


# ── IngestService.ingest_document ─────────────────────────────────────────────


class TestIngestDocument:
    def _service(self):
        chunker = MagicMock()
        embedder = MagicMock()
        return IngestService(chunker=chunker, embedder=embedder)

    def _mock_chunk(self, content: str = "chunk content"):
        c = MagicMock()
        c.content = content
        c.index = 0
        c.token_count = 10
        c.metadata = {}
        c.embedding = [0.1] * 1536
        return c

    @pytest.mark.asyncio
    async def test_new_document_inserted(self):
        svc = self._service()
        chunk = self._mock_chunk()
        svc._chunker.chunk_document = AsyncMock(return_value=[chunk])
        svc._embedder.embed_chunks = AsyncMock(return_value=[chunk])

        with (
            patch(
                "ingestion.ingest_service.IngestService._get_chunker",
                return_value=svc._chunker,
            ),
            patch(
                "ingestion.ingest_service.IngestService._get_embedder",
                return_value=svc._embedder,
            ),
            patch(
                "agent.db_utils.get_document_by_source", AsyncMock(return_value=None)
            ),
            patch(
                "agent.db_utils.upsert_document", AsyncMock(return_value="doc-uuid-new")
            ),
            patch("agent.db_utils.save_chunks", AsyncMock()),
            patch("agent.db_utils.delete_document_chunks", AsyncMock()),
            patch("ingestion.graph_builder.build_knowledge_graph"),
            patch("agent.retriever.hybrid_retriever.rebuild_index", AsyncMock()),
        ):
            result = await svc.ingest_document(
                content="New document content",
                source="s3://bucket/new.pdf",
                title="New Doc",
            )

        assert result.status == "inserted"
        assert result.document_id == "doc-uuid-new"
        assert result.chunks_created == 1

    @pytest.mark.asyncio
    async def test_unchanged_document_skipped(self):
        svc = self._service()
        content = "Unchanged document content"
        content_hash = _sha256(content)
        existing = {"id": "existing-uuid", "content_hash": content_hash}

        with patch(
            "agent.db_utils.get_document_by_source", AsyncMock(return_value=existing)
        ):
            result = await svc.ingest_document(
                content=content,
                source="s3://bucket/unchanged.pdf",
                title="Unchanged Doc",
            )

        assert result.status == "skipped"
        assert result.document_id == "existing-uuid"

    @pytest.mark.asyncio
    async def test_changed_document_updated(self):
        svc = self._service()
        old_hash = _sha256("old content")
        existing = {"id": "doc-uuid", "content_hash": old_hash}
        new_content = "New updated content for this document"
        chunk = self._mock_chunk(new_content)
        svc._chunker.chunk_document = AsyncMock(return_value=[chunk])
        svc._embedder.embed_chunks = AsyncMock(return_value=[chunk])

        # Old DB chunk has a different content_hash → it will be deleted
        old_db_chunk = {"id": "old-chunk-id", "content_hash": _sha256("old content")}

        with (
            patch(
                "ingestion.ingest_service.IngestService._get_chunker",
                return_value=svc._chunker,
            ),
            patch(
                "ingestion.ingest_service.IngestService._get_embedder",
                return_value=svc._embedder,
            ),
            patch(
                "agent.db_utils.get_document_by_source",
                AsyncMock(return_value=existing),
            ),
            patch(
                "agent.db_utils.get_document_chunks",
                AsyncMock(return_value=[old_db_chunk]),
            ),
            patch(
                "agent.db_utils.delete_chunks_by_ids", AsyncMock()
            ) as mock_delete_ids,
            patch("agent.db_utils.upsert_document", AsyncMock(return_value="doc-uuid")),
            patch("agent.db_utils.save_chunks", AsyncMock()),
            patch("ingestion.graph_builder.build_knowledge_graph"),
            patch("agent.retriever.hybrid_retriever.rebuild_index", AsyncMock()),
        ):
            result = await svc.ingest_document(
                content=new_content,
                source="s3://bucket/changed.pdf",
                title="Changed Doc",
            )
            # Only the stale chunk id should have been deleted
            mock_delete_ids.assert_awaited_once_with(["old-chunk-id"])

        assert result.status == "updated"
        assert result.chunks_created == 1

    @pytest.mark.asyncio
    async def test_ingestion_error_returns_failed_status(self):
        svc = self._service()

        with patch(
            "agent.db_utils.get_document_by_source",
            AsyncMock(side_effect=RuntimeError("DB connection lost")),
        ):
            result = await svc.ingest_document(
                content="content", source="s3://bucket/fail.pdf", title="Fail"
            )

        assert result.status == "failed"
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_access_level_passed_to_upsert(self):
        svc = self._service()
        chunk = self._mock_chunk()
        svc._chunker.chunk_document = AsyncMock(return_value=[chunk])
        svc._embedder.embed_chunks = AsyncMock(return_value=[chunk])
        captured_access: list = []

        async def _fake_upsert(**kwargs):
            captured_access.append(kwargs.get("access_level"))
            return "doc-id"

        with (
            patch(
                "ingestion.ingest_service.IngestService._get_chunker",
                return_value=svc._chunker,
            ),
            patch(
                "ingestion.ingest_service.IngestService._get_embedder",
                return_value=svc._embedder,
            ),
            patch(
                "agent.db_utils.get_document_by_source", AsyncMock(return_value=None)
            ),
            patch("agent.db_utils.upsert_document", _fake_upsert),
            patch("agent.db_utils.save_chunks", AsyncMock()),
            patch("ingestion.graph_builder.build_knowledge_graph"),
            patch("agent.retriever.hybrid_retriever.rebuild_index", AsyncMock()),
        ):
            await svc.ingest_document(
                content="private content",
                source="s3://private-bucket/doc.pdf",
                title="Private Doc",
                access_level="private",
            )

        assert captured_access[0] == "private"

    @pytest.mark.asyncio
    async def test_chunks_fed_to_vespa_after_ingest(self):
        svc = self._service()
        chunk = self._mock_chunk()
        svc._chunker.chunk_document = AsyncMock(return_value=[chunk])
        svc._embedder.embed_chunks = AsyncMock(return_value=[chunk])
        saved_rows = [
            {"id": "c1", "document_id": "doc-id", "content": "text", "metadata": {}}
        ]

        with (
            patch(
                "ingestion.ingest_service.IngestService._get_chunker",
                return_value=svc._chunker,
            ),
            patch(
                "ingestion.ingest_service.IngestService._get_embedder",
                return_value=svc._embedder,
            ),
            patch(
                "agent.db_utils.get_document_by_source", AsyncMock(return_value=None)
            ),
            patch("agent.db_utils.upsert_document", AsyncMock(return_value="doc-id")),
            patch("agent.db_utils.save_chunks", AsyncMock(return_value=saved_rows)),
            patch("ingestion.graph_builder.build_knowledge_graph"),
            patch("agent.vespa_client.vespa_enabled", return_value=True),
            patch("agent.vespa_client.feed_chunks", AsyncMock()) as mock_feed,
        ):
            await svc.ingest_document(content="text", source="src", title="t")
            mock_feed.assert_awaited_once()
            # fed rows are enriched with the document title/source/access level
            fed = mock_feed.await_args.args[0]
            assert fed[0]["document_source"] == "src"
            assert fed[0]["access_level"] == "public"


# ── IngestService.ingest_all_s3_buckets ──────────────────────────────────────


class TestIngestAllS3Buckets:
    @pytest.mark.asyncio
    async def test_aggregates_stats_from_both_buckets(self):
        svc = IngestService()
        private_stats = IngestStats(inserted=2, skipped=1)
        public_stats = IngestStats(inserted=1, failed=1)

        with (
            patch.object(
                svc,
                "ingest_from_s3",
                AsyncMock(side_effect=[private_stats, public_stats]),
            ),
        ):
            combined = await svc.ingest_all_s3_buckets()

        assert combined.inserted == 3
        assert combined.skipped == 1
        assert combined.failed == 1

    @pytest.mark.asyncio
    async def test_calls_both_bucket_types(self):
        svc = IngestService()
        called_with: list = []

        async def _fake_ingest(bucket_type="private", prefix=""):
            called_with.append(bucket_type)
            return IngestStats()

        with patch.object(svc, "ingest_from_s3", _fake_ingest):
            await svc.ingest_all_s3_buckets()

        assert "private" in called_with
        assert "public" in called_with


# ── IngestService.ingest_single_s3_object ────────────────────────────────────


class TestIngestSingleS3Object:
    @pytest.mark.asyncio
    async def test_returns_failed_when_download_fails(self):
        svc = IngestService()

        with patch("ingestion.s3_utils.download_document_from_s3", return_value=None):
            result = await svc.ingest_single_s3_object(
                "bad/key.pdf", "my-private-bucket"
            )

        assert result.status == "failed"

    @pytest.mark.asyncio
    async def test_detects_private_bucket_correctly(self, monkeypatch):
        svc = IngestService()
        monkeypatch.setenv("S3_PRIVATE_BUCKET", "my-private-bucket")

        chunk = MagicMock()
        chunk.content = "text"
        chunk.index = 0
        chunk.token_count = 5
        chunk.metadata = {}
        chunk.embedding = [0.1] * 1536

        mock_chunker = MagicMock()
        mock_chunker.chunk_document = AsyncMock(return_value=[chunk])
        mock_embedder = MagicMock()
        mock_embedder.embed_chunks = AsyncMock(return_value=[chunk])
        captured_level: list = []

        async def _fake_ingest(
            content, source, title, metadata=None, access_level="public"
        ):
            captured_level.append(access_level)
            return IngestResult(source=source, status="inserted", chunks_created=1)

        with (
            patch(
                "ingestion.s3_utils.download_document_from_s3", return_value=b"content"
            ),
            patch(
                "ingestion.file_parsers.parse_document",
                return_value=("document text", {}),
            ),
            patch.object(svc, "ingest_document", _fake_ingest),
        ):
            await svc.ingest_single_s3_object("docs/report.pdf", "my-private-bucket")

        assert captured_level[0] == "private"
