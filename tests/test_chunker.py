"""
Unit tests for document chunking (SimpleChunker and SemanticChunker).
"""

from __future__ import annotations


import pytest

from ingestion.chunker import (
    ChunkingConfig,
    DocumentChunk,
    SemanticChunker,
    SimpleChunker,
    create_chunker,
)


class TestChunkingConfig:
    def test_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.use_semantic_splitting is True

    def test_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValueError, match="overlap"):
            ChunkingConfig(chunk_size=500, chunk_overlap=500)

    def test_overlap_zero_is_valid(self):
        cfg = ChunkingConfig(chunk_overlap=0)
        assert cfg.chunk_overlap == 0

    def test_min_chunk_size_must_be_positive(self):
        with pytest.raises(ValueError, match="Minimum"):
            ChunkingConfig(min_chunk_size=0)


class TestDocumentChunk:
    def test_token_count_estimated_from_content(self):
        chunk = DocumentChunk(
            content="a" * 400,
            index=0,
            start_char=0,
            end_char=400,
            metadata={},
        )
        # Rough estimate: len(content) // 4
        assert chunk.token_count == 100

    def test_explicit_token_count_preserved(self):
        chunk = DocumentChunk(
            content="hello",
            index=0,
            start_char=0,
            end_char=5,
            metadata={},
            token_count=42,
        )
        assert chunk.token_count == 42


class TestCreateChunker:
    def test_semantic_splitting_returns_semantic_chunker(self):
        cfg = ChunkingConfig(use_semantic_splitting=True)
        chunker = create_chunker(cfg)
        assert isinstance(chunker, SemanticChunker)

    def test_no_semantic_splitting_returns_simple_chunker(self):
        cfg = ChunkingConfig(use_semantic_splitting=False)
        chunker = create_chunker(cfg)
        assert isinstance(chunker, SimpleChunker)


# ── SimpleChunker ─────────────────────────────────────────────────────────────


class TestSimpleChunker:
    @pytest.fixture()
    def chunker(self):
        return SimpleChunker(
            ChunkingConfig(
                chunk_size=200, chunk_overlap=20, use_semantic_splitting=False
            )
        )

    def test_empty_content_returns_no_chunks(self, chunker):
        chunks = chunker.chunk_document("", "title", "src")
        assert chunks == []

    def test_whitespace_content_returns_no_chunks(self, chunker):
        chunks = chunker.chunk_document("   \n\n  ", "title", "src")
        assert chunks == []

    def test_short_content_returns_single_chunk(self, chunker):
        chunks = chunker.chunk_document("Short text.", "title", "src")
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_chunk_metadata_includes_title_and_source(self, chunker):
        chunks = chunker.chunk_document("Some text.", "My Title", "my/source.txt")
        assert chunks[0].metadata["title"] == "My Title"
        assert chunks[0].metadata["source"] == "my/source.txt"
        assert chunks[0].metadata["chunk_method"] == "simple"

    def test_long_content_split_into_multiple_chunks(self, chunker):
        paragraph = "Word " * 50  # ~250 chars
        content = "\n\n".join([paragraph] * 5)
        chunks = chunker.chunk_document(content, "title", "src")
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self, chunker):
        paragraph = "Word " * 50
        content = "\n\n".join([paragraph] * 5)
        chunks = chunker.chunk_document(content, "title", "src")
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_total_chunks_in_metadata(self, chunker):
        paragraph = "Word " * 50
        content = "\n\n".join([paragraph] * 5)
        chunks = chunker.chunk_document(content, "title", "src")
        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_additional_metadata_merged(self, chunker):
        chunks = chunker.chunk_document("text", "title", "src", metadata={"lang": "en"})
        assert chunks[0].metadata["lang"] == "en"


class TestSimpleChunkerHelpers:
    """Tests for SimpleChunker internal helpers."""

    @pytest.fixture()
    def chunker(self):
        return SimpleChunker(
            ChunkingConfig(
                chunk_size=100, chunk_overlap=10, use_semantic_splitting=False
            )
        )

    def test_create_chunk_preserves_content(self, chunker):
        chunk = chunker._create_chunk(
            "short text", 0, 0, 10, {"chunk_method": "simple"}
        )
        assert chunk.content == "short text"

    def test_semantic_chunker_simple_split_creates_multiple_parts(self):
        """SemanticChunker._simple_split produces overlapping sub-chunks."""
        semantic = SemanticChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))
        text = "a" * 300
        parts = semantic._simple_split(text)
        assert len(parts) > 1


# ── SemanticChunker ───────────────────────────────────────────────────────────


class TestSemanticChunker:
    @pytest.fixture()
    def chunker(self):
        return SemanticChunker(ChunkingConfig(chunk_size=300, chunk_overlap=30))

    @pytest.mark.asyncio
    async def test_empty_content_returns_no_chunks(self, chunker):
        chunks = await chunker.chunk_document("", "t", "s")
        assert chunks == []

    @pytest.mark.asyncio
    async def test_tabular_data_uses_simple_chunking(self, chunker):
        tabular = "col1\tcol2\tcol3\n" + "a\tb\tc\n" * 20
        chunks = await chunker.chunk_document(tabular, "sheet", "data.csv")
        # tabular data skips semantic chunking; should still produce chunks
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_short_content_single_chunk(self, chunker):
        chunks = await chunker.chunk_document("Short paragraph.", "t", "s")
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_large_content_falls_back_to_simple(self, chunker):
        # >500 KB triggers the simple-chunking fast path
        large = "word " * 110_000  # ~550 KB
        chunks = await chunker.chunk_document(large, "big_doc", "big.txt")
        assert len(chunks) > 0

    def test_is_tabular_detects_tab_data(self, chunker):
        text = "\t".join(["col"] * 5) + "\n" + ("\t".join(["val"] * 5) + "\n") * 5
        assert chunker._is_tabular_data(text) is True

    def test_is_tabular_detects_csv(self, chunker):
        csv_text = "a,b,c,d\n" * 10
        assert chunker._is_tabular_data(csv_text) is True

    def test_is_tabular_rejects_prose(self, chunker):
        prose = "This is a normal sentence. No tabs here."
        assert chunker._is_tabular_data(prose) is False

    def test_split_on_structure_markdown(self, chunker):
        md = "# Header\n\nParagraph one.\n\n## Subheader\n\nParagraph two."
        sections = chunker._split_on_structure(md)
        assert len(sections) >= 2
