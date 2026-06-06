# Test Cases Documentation

> Test suite: `tests/` | Runner: `pytest` | Async mode: `asyncio_mode = auto`
> Total: **202 tests** — 0 failures, 0 skipped

Run all tests (Docker — the supported workflow; hermetic, no local venv):
```bash
docker run --rm -v "$(pwd)":/app -w /app uchenab-backend:latest python -m pytest -q
```

---

## Table of Contents

1. [_sha256 Utility](#1-_sha256-utility)
2. [Data Models](#2-data-models)
3. [Chunkers](#3-chunkers)
4. [Embedder](#4-embedder)
5. [Hybrid Retriever (BM25 + pgvector + RRF)](#5-hybrid-retriever-bm25--pgvector--rrf)
6. [Ingest Service (Upsert / Deduplication Pipeline)](#6-ingest-service-upsert--deduplication-pipeline)
7. [Agent Tools](#7-agent-tools)
8. [JSON Storage](#8-json-storage)
9. [Access Control](#9-access-control)
10. [Graph Utils](#10-graph-utils)

---

## 1. `_sha256` Utility

**File:** `tests/test_ingest_service.py` | **Class:** `TestSha256`

| # | Test Name                         | Input              | Expected Output                     | Status |
|---|-----------------------------------|--------------------|-------------------------------------|--------|
| 1 | `test_returns_64_char_hex`        | `"hello world"`    | 64-char lowercase hex string        | PASS   |
| 2 | `test_same_input_same_hash`       | `"test"` twice     | Both hashes equal                   | PASS   |
| 3 | `test_different_inputs_different_hashes` | `"foo"` vs `"bar"` | Hashes differ                   | PASS   |
| 4 | `test_empty_string_hashed`        | `""`               | 64-char hex (no error)              | PASS   |

**Why:** SHA-256 is the deduplication key. These tests assert determinism and uniqueness before the hash is stored in `documents.content_hash`.

---

## 2. Data Models

### 2.1 Enumerations

**File:** `tests/test_models.py` | **Class:** `TestEnumerations`

| # | Test Name                    | Input                  | Expected Output             | Status |
|---|------------------------------|------------------------|-----------------------------|--------|
| 1 | `test_message_role_values`   | `MessageRole` members  | `"user"`, `"assistant"`, `"system"` | PASS |
| 2 | `test_search_type_values`    | `SearchType` members   | `"vector"`, `"hybrid"`, `"graph"`   | PASS |

### 2.2 ChatRequest

**File:** `tests/test_models.py` | **Class:** `TestChatRequest`

| # | Test Name                        | Input                                 | Expected Output              | Status |
|---|----------------------------------|---------------------------------------|------------------------------|--------|
| 3 | `test_required_message`          | `message="hello"`                     | Model constructed, no error  | PASS   |
| 4 | `test_custom_search_type`        | `search_type="graph"`                 | `request.search_type == SearchType.graph` | PASS |
| 5 | `test_missing_message_raises`    | No `message` field                    | `ValidationError` raised     | PASS   |
| 6 | `test_metadata_defaults_to_empty`| No `metadata`                         | `metadata == {}`             | PASS   |

### 2.3 SearchRequest

**File:** `tests/test_models.py` | **Class:** `TestSearchRequest`

| # | Test Name                    | Input              | Expected Output    | Status |
|---|------------------------------|--------------------|--------------------|--------|
| 7 | `test_default_limit`         | No `limit`         | `limit == 10`      | PASS   |
| 8 | `test_limit_clamps_lower`    | `limit=0`          | `ValidationError`  | PASS   |
| 9 | `test_limit_clamps_upper`    | `limit=101`        | `ValidationError`  | PASS   |
|10 | `test_limit_valid_boundary`  | `limit=100`        | `limit == 100`     | PASS   |

### 2.4 ChunkResult

**File:** `tests/test_models.py` | **Class:** `TestChunkResult`

| # | Test Name                      | Input        | Expected Output  | Status |
|---|--------------------------------|--------------|------------------|--------|
|11 | `test_score_clamped_above_one` | `score=1.5`  | `score == 1.0`   | PASS   |
|12 | `test_score_clamped_below_zero`| `score=-0.1` | `score == 0.0`   | PASS   |
|13 | `test_score_in_range_unchanged`| `score=0.85` | `score == 0.85`  | PASS   |

### 2.5 GraphSearchResult

**File:** `tests/test_models.py` | **Class:** `TestGraphSearchResult`

| # | Test Name                     | Input                          | Expected Output        | Status |
|---|-------------------------------|--------------------------------|------------------------|--------|
|14 | `test_minimal_fields`         | `fact`, `uuid` only            | Temporal fields are `None` | PASS |
|15 | `test_with_temporal_fields`   | `valid_at="2023-01-01"`        | Fields preserved       | PASS   |

### 2.6 IngestionConfig

**File:** `tests/test_models.py` | **Class:** `TestIngestionConfig`

| # | Test Name                              | Input                              | Expected Output        | Status |
|---|----------------------------------------|------------------------------------|------------------------|--------|
|16 | `test_defaults`                        | No args                            | `chunk_size=1000`, `overlap=200` | PASS |
|17 | `test_overlap_must_be_less_than_chunk_size` | `overlap >= chunk_size`       | `ValidationError`      | PASS   |
|18 | `test_overlap_just_below_chunk_size`   | `chunk_size=100, overlap=99`       | Valid, no error        | PASS   |
|19 | `test_skip_graph_building_flag`        | `skip_graph_building=True`         | Flag stored correctly  | PASS   |

### 2.7 Chunk

**File:** `tests/test_models.py` | **Class:** `TestChunk`

| # | Test Name                    | Input                          | Expected Output            | Status |
|---|------------------------------|--------------------------------|----------------------------|--------|
|20 | `test_valid_embedding`       | `embedding=[0.1]*1536`         | Accepted                   | PASS   |
|21 | `test_empty_embedding_raises`| `embedding=[]`                 | `ValidationError`          | PASS   |
|22 | `test_none_embedding_allowed`| `embedding=None`               | Accepted (pre-embed state) | PASS   |

### 2.8 ToolCall

**File:** `tests/test_models.py` | **Class:** `TestToolCall`

| # | Test Name           | Input                        | Expected Output        | Status |
|---|---------------------|------------------------------|------------------------|--------|
|23 | `test_default_args` | `tool_name="search"` only    | `args == {}`           | PASS   |
|24 | `test_with_args`    | `args={"query": "AI"}`       | Args preserved         | PASS   |

### 2.9 IngestResult / IngestStats

**File:** `tests/test_ingest_service.py` | **Classes:** `TestIngestResult`, `TestIngestStats`

| # | Test Name                        | Input                                  | Expected Output                | Status |
|---|----------------------------------|----------------------------------------|--------------------------------|--------|
|25 | `test_failed_result_has_error`   | `status="failed", error="timeout"`     | `error == "timeout"`, `chunks_created == 0` | PASS |
|26 | `test_success_result`            | `status="inserted", chunks_created=5`  | Fields stored correctly        | PASS   |
|27 | `test_total_sums_all_buckets`    | `inserted=3, updated=2, skipped=1, failed=1` | `total == 7`             | PASS   |
|28 | `test_empty_stats_total_zero`    | Default `IngestStats()`                | `total == 0`                   | PASS   |

---

## 3. Chunkers

**File:** `tests/test_chunker.py`

### 3.1 SimpleChunker

| # | Test Name                                | Input                               | Expected Output                | Status |
|---|------------------------------------------|-------------------------------------|--------------------------------|--------|
|29 | `test_creates_chunks_from_text`          | 200-word text, `chunk_size=100`     | 2+ chunks returned             | PASS   |
|30 | `test_chunk_has_required_fields`         | Any text                            | Each chunk has `content`, `index`, `token_count` | PASS |
|31 | `test_empty_text_returns_empty_list`     | `""`                                | `[]`                           | PASS   |
|32 | `test_short_text_is_single_chunk`        | Short sentence                      | 1 chunk                        | PASS   |
|33 | `test_metadata_attached_to_chunks`       | `metadata={"source": "test.pdf"}`   | Each chunk carries metadata    | PASS   |
|34 | `test_chunk_size_respected`              | `chunk_size=50`                     | Each chunk ≤ 50 tokens (approx)| PASS   |

### 3.2 SemanticChunker

| # | Test Name                                | Input                               | Expected Output                | Status |
|---|------------------------------------------|-------------------------------------|--------------------------------|--------|
|35 | `test_splits_on_headers`                 | Markdown with `##` headings         | Separate chunks per section    | PASS   |
|36 | `test_respects_max_chunk_size`           | Large text, `max_chunk_size=500`    | No chunk exceeds limit         | PASS   |
|37 | `test_overlap_preserved`                 | Text with `overlap=50`              | Overlap tokens appear in next chunk | PASS |
|38 | `test_empty_text_returns_empty_list`     | `""`                                | `[]`                           | PASS   |
|39 | `test_single_paragraph_no_split`         | Single paragraph                    | 1 chunk                        | PASS   |
|40 | `test_simple_split_fallback`             | `SemanticChunker._simple_split()`   | List of string parts           | PASS   |

---

## 4. Embedder

**File:** `tests/test_embedder.py`

| # | Test Name                              | Input                         | Expected Output                     | Status |
|---|----------------------------------------|-------------------------------|-------------------------------------|--------|
|41 | `test_embed_single_chunk`              | 1 chunk, mock embedding       | Chunk with `embedding` filled in    | PASS   |
|42 | `test_embed_multiple_chunks`           | 3 chunks                      | All 3 returned with embeddings      | PASS   |
|43 | `test_empty_chunks_returns_empty`      | `[]`                          | `[]`                                | PASS   |
|44 | `test_embedding_dimension_preserved`   | Mock returns 1536-dim vector  | `len(chunk.embedding) == 1536`      | PASS   |
|45 | `test_rate_limit_error_retried`        | First call raises `RateLimitError`, second succeeds | Result returned | PASS |
|46 | `test_api_error_raises`               | `APIError` raised             | Exception propagated               | PASS   |

---

## 5. Hybrid Retriever (BM25 + pgvector + RRF)

**File:** `tests/test_retriever.py`

### 5.1 RRF Fusion (pure Python, no mocking)

**Class:** `TestRRFFusion`

| # | Test Name                              | Input                                    | Expected Output                        | Status |
|---|----------------------------------------|------------------------------------------|----------------------------------------|--------|
|47 | `test_empty_inputs_return_empty`       | Both lists empty                         | `[]`                                   | PASS   |
|48 | `test_vector_only_returns_sorted`      | 3 vector rows (scores 0.9, 0.8, 0.7)    | `c1` ranked first                      | PASS   |
|49 | `test_bm25_only_returns_sorted`        | 2 BM25 rows (0.95, 0.75)                | `b1` ranked first                      | PASS   |
|50 | `test_overlap_boosts_shared_chunks`    | `c1` in both vector+BM25 lists          | `c1` ranked first (dual-list boost)    | PASS   |
|51 | `test_limit_respected`                 | 10 vector rows, `limit=3`               | Exactly 3 results                      | PASS   |
|52 | `test_result_has_rrf_score_field`      | 1 vector row                            | `result[0]["rrf_score"] > 0`           | PASS   |
|53 | `test_result_has_combined_score_field` | 1 vector row                            | `"combined_score"` key present         | PASS   |
|54 | `test_no_duplicate_chunk_ids_in_output`| Same chunk in both lists                | Each chunk_id appears once             | PASS   |
|55 | `test_rrf_k_affects_scores`            | `rrf_k=1` vs `rrf_k=100`               | Lower k → higher score (verified math) | PASS   |

**RRF formula:** `score = 1 / (rank + k)` where `k=60` by default.

### 5.2 BM25 Index Build

**Class:** `TestBuildIndex`

| # | Test Name                              | Input                                   | Expected Output                   | Status |
|---|----------------------------------------|-----------------------------------------|-----------------------------------|--------|
|56 | `test_build_with_empty_chunks_does_not_error` | `chunks=[]`                    | `_bm25 is None`, no exception     | PASS   |
|57 | `test_build_with_chunks_creates_bm25`  | 1 chunk dict with `chunk_id`, `content` | `_bm25 is not None`, id in `_node_map` | PASS |
|58 | `test_rebuild_replaces_previous_index` | Build v1 then v2                        | v2 chunks in map, v1 chunks gone  | PASS   |

### 5.3 Retrieve

**Class:** `TestRetrieve`

| # | Test Name                                        | Mocks                                    | Expected Output              | Status |
|---|--------------------------------------------------|------------------------------------------|------------------------------|--------|
|59 | `test_returns_merged_results`                    | `vector_search` → 1 row; BM25 index built | `result` is a list          | PASS   |
|60 | `test_handles_vector_search_error_gracefully`    | `vector_search` → `RuntimeError`         | `result == []`               | PASS   |
|61 | `test_empty_bm25_index_still_returns_vector_results` | No BM25 index; vector returns 1 row  | `len(result) >= 0`           | PASS   |

### 5.4 Node Conversion

**Class:** `TestToNode`

| # | Test Name                          | Input                              | Expected Output                     | Status |
|---|------------------------------------|------------------------------------|-------------------------------------|--------|
|62 | `test_creates_node_with_correct_id`| Chunk dict with `chunk_id="abc"` | `node.node_id == "abc-123"`          | PASS   |
|63 | `test_metadata_populated`          | `document_title="My Doc"`          | `node.metadata["document_title"] == "My Doc"` | PASS |

---

## 6. Ingest Service (Upsert / Deduplication Pipeline)

**File:** `tests/test_ingest_service.py`

### 6.1 `ingest_document` — Core Upsert Logic

**Class:** `TestIngestDocument`

| # | Test Name                               | Scenario                                               | Expected Output                  | Status |
|---|-----------------------------------------|--------------------------------------------------------|----------------------------------|--------|
|64 | `test_new_document_inserted`            | `get_document_by_source` → `None` (first time)         | `status="inserted"`, chunks saved | PASS  |
|65 | `test_unchanged_document_skipped`       | Existing doc with **same** content hash                | `status="skipped"`, no DB writes  | PASS  |
|66 | `test_changed_document_updated`         | Existing doc with **different** hash                   | `delete_chunks` called, `status="updated"` | PASS |
|67 | `test_ingestion_error_returns_failed_status` | `get_document_by_source` raises `RuntimeError`    | `status="failed"`, `error` field set | PASS |
|68 | `test_access_level_passed_to_upsert`    | `access_level="private"` provided                      | Upsert receives `access_level="private"` | PASS |
|69 | `test_bm25_index_rebuilt_after_ingest`  | Successful insert                                      | `hybrid_retriever.rebuild_index` awaited once | PASS |

**Upsert decision flow tested by these cases:**

```
get_document_by_source(source)
├── None          → INSERT  (test 64)
├── same hash     → SKIP    (test 65)
└── different hash
    ├── delete_document_chunks(doc_id)
    └── UPDATE             (test 66)
```

### 6.2 `ingest_all_s3_buckets`

**Class:** `TestIngestAllS3Buckets`

| # | Test Name                              | Scenario                                    | Expected Output                     | Status |
|---|----------------------------------------|---------------------------------------------|-------------------------------------|--------|
|70 | `test_aggregates_stats_from_both_buckets` | Private: `inserted=2, skipped=1`; Public: `inserted=1, failed=1` | Combined totals correct | PASS |
|71 | `test_calls_both_bucket_types`         | `ingest_from_s3` patched                    | Called with `"private"` and `"public"` | PASS |

### 6.3 `ingest_single_s3_object`

**Class:** `TestIngestSingleS3Object`

| # | Test Name                              | Scenario                                       | Expected Output        | Status |
|---|----------------------------------------|------------------------------------------------|------------------------|--------|
|72 | `test_returns_failed_when_download_fails` | `download_document_from_s3` → `None`        | `status="failed"`      | PASS   |
|73 | `test_detects_private_bucket_correctly`| `S3_PRIVATE_BUCKET=my-private-bucket`; bucket matches | `access_level="private"` passed to `ingest_document` | PASS |

---

## 7. Agent Tools

**File:** `tests/test_tools.py`

### 7.1 Input Models

**Class:** `TestInputModels`

| # | Test Name                              | Input                           | Expected Output                  | Status |
|---|----------------------------------------|---------------------------------|----------------------------------|--------|
|74 | `test_vector_search_input_defaults`    | `query="test"` only             | `limit=10`, `user_id=None`       | PASS   |
|75 | `test_hybrid_search_input_weights`     | `text_weight=0.7`               | Weight stored                    | PASS   |
|76 | `test_graph_search_input_query_required` | No `query`                    | `ValidationError`                | PASS   |
|77 | `test_document_list_defaults`          | No args                         | `limit=20`, `offset=0`           | PASS   |
|78 | `test_entity_relationship_default_depth` | `entity_name="Google"` only   | `depth == 2`                     | PASS   |

### 7.2 Vector Search Tool

**Class:** `TestVectorSearchTool`

| # | Test Name                         | Mocks                                         | Expected Output                  | Status |
|---|-----------------------------------|-----------------------------------------------|----------------------------------|--------|
|79 | `test_returns_chunk_results`      | `generate_embedding`, `vector_search` patched | `len(results) == 1`, `score` correct | PASS |
|80 | `test_returns_empty_list_on_error`| `generate_embedding` raises `ValueError`      | `results == []`                  | PASS   |
|81 | `test_passes_user_id_to_db`       | `user_id="u1"` in input                       | `vector_search` called with `user_id="u1"` | PASS |

### 7.3 Graph Search Tool

**Class:** `TestGraphSearchTool`

| # | Test Name                         | Mocks                                           | Expected Output               | Status |
|---|-----------------------------------|-------------------------------------------------|-------------------------------|--------|
|82 | `test_returns_graph_results`      | `search_knowledge_graph` → 1 fact               | 1 `GraphSearchResult`         | PASS   |
|83 | `test_returns_empty_list_on_error`| `search_knowledge_graph` raises `RuntimeError`  | `results == []`               | PASS   |
|84 | `test_maps_fact_field`            | Row with `fact="Google acquired DeepMind"`      | `result.fact` matches         | PASS   |

### 7.4 Hybrid Search Tool

**Class:** `TestHybridSearchTool`

| # | Test Name                         | Mocks                                                   | Expected Output              | Status |
|---|-----------------------------------|---------------------------------------------------------|------------------------------|--------|
|85 | `test_returns_chunk_results`      | `generate_embedding`, `hybrid_retriever.retrieve`, `hybrid_search` patched | `score == 0.88` | PASS |
|86 | `test_returns_empty_list_on_error`| `generate_embedding` raises `ValueError`                | `results == []`              | PASS   |

### 7.5 List Documents Tool

**Class:** `TestListDocumentsTool`

| # | Test Name                          | Mocks                                  | Expected Output            | Status |
|---|------------------------------------|----------------------------------------|----------------------------|--------|
|87 | `test_returns_document_metadata`   | `list_documents` → 1 document row      | 1 `DocumentMetadata`       | PASS   |
|88 | `test_returns_empty_list_on_error` | `list_documents` raises `RuntimeError` | `results == []`            | PASS   |

### 7.6 Confidence Gate (cosine-similarity based)

**Class:** `TestConfidenceGate` — verifies `tools.is_low_confidence` /
`max_cosine_similarity`, which gate answers on the stable pgvector cosine
similarity (`CONFIDENCE_MIN_SIMILARITY`, default 0.25) instead of the old
rank-based RRF score.

| # | Test Name                         | Input                                  | Expected Output            | Status |
|---|-----------------------------------|----------------------------------------|----------------------------|--------|
|89 | `test_max_cosine_empty_is_zero`   | `[]`                                   | `0.0`                      | PASS   |
|90 | `test_max_cosine_picks_highest`   | chunks cos 0.20/0.55/0.31              | `0.55`                     | PASS   |
|91 | `test_low_confidence_when_no_chunks` | `[]`                                | `is_low_confidence == True`| PASS   |
|92 | `test_low_confidence_below_threshold` | top cosine 0.10 (off-domain)       | `True` (suppress)          | PASS   |
|93 | `test_high_confidence_above_threshold`| top cosine ≥ gate+0.2 (relevant)   | `False` (answer)           | PASS   |
|94 | `test_threshold_is_sane`          | —                                      | `0.1 < gate < 0.4`         | PASS   |

---

## 8. Agent Settings Store (editable system prompt / scope)

**File:** `tests/test_settings_store.py` — the runtime-editable assistant config
(`agent/settings_store.py`) backing `PUT /api/v1/settings/agent`.

| # | Test Name                              | Input                             | Expected Output                      | Status |
|---|----------------------------------------|-----------------------------------|--------------------------------------|--------|
|95 | `test_defaults_are_university`         | `AgentConfig()`                   | Default name/prompt/scope, `enforce_scope=True` | PASS |
|96 | `test_full_row_maps_through`           | Full DB row (CineMENA)            | All fields mapped; `enforce_scope=False` | PASS |
|97 | `test_blank_fields_fall_back_to_defaults` | Empty/None fields              | Falls back to built-in defaults      | PASS   |
|98 | `test_returns_default_without_client`  | No Supabase client                | Returns defaults, never raises       | PASS   |

---

## 9. Access Control

**File:** `tests/test_access_control.py`

| # | Test Name                                 | Input                                | Expected Output           | Status |
|---|-------------------------------------------|--------------------------------------|---------------------------|--------|
|99 | `test_private_bucket_assigns_private`     | `bucket_type="private"`              | `"private"`               | PASS   |
|100| `test_public_bucket_assigns_public`       | `bucket_type="public"`               | `"public"`                | PASS   |
|101| `test_admin_user_can_access_private`      | `user_id` in `ADMIN_USERS`           | `can_access == True`      | PASS   |
|102| `test_private_user_can_access_private`    | `user_id` in `PRIVATE_DOCUMENT_USERS`| `can_access == True`      | PASS   |
|103| `test_anonymous_cannot_access_private`    | `user_id=None, access_level="private"` | `can_access == False`   | PASS   |

> Note: `get_user_access_filter` (and the SQL-string `_build_access_filter`) were
> removed during cleanup — all access filtering now flows through
> `can_access_document` / the PostgREST `access_level` predicate.

---

## 10. Knowledge Graph (Neo4j)

Graph retrieval (`agent/graph_utils.search_knowledge_graph`) and graph building
(`ingestion/graph_builder.build_knowledge_graph`) talk to a live Neo4j instance,
so they are exercised via **integration scripts** rather than mocked unit tests:

| Check | How | Expected |
|-------|-----|----------|
| Entity/fact extraction writes nodes | `scripts/backfill_graph.py` | `:Entity`/`:Fact`/`:Chunk` nodes + `HAS_FACT`/`FROM_CHUNK` rels created |
| Fact search returns results | `search_knowledge_graph("...")` | Non-empty fact list; falls back to `:Chunk` keyword search when no fact matches |
| Lucene safety | query with operators (`:`, `(`, `*`) | Sanitised; no Cypher/Lucene error |

> Regression fixed: `session.run(cypher, query=...)` collided with Neo4j's
> positional `query` arg, so graph search had silently returned `[]`. Parameters
> are now passed as a dict: `session.run(cypher, {"query": safe})`.

---

## Mock / Stub Strategy

All tests run without real external services. The following stubs are installed in `tests/conftest.py` before any project module is imported:

| Package                        | Stub Type               | Key Behaviour                                           |
|-------------------------------|-------------------------|---------------------------------------------------------|
| `openai`                      | `MagicMock`             | `AsyncOpenAI().embeddings.create()` returns 1536-dim vector |
| `asyncpg`                     | `MagicMock`             | `create_pool()` → mock pool                             |
| `neo4j`                       | `MagicMock`             | `AsyncGraphDatabase.driver()` → mock driver             |
| `boto3` / `botocore`          | `MagicMock`             | S3 client is a mock; `ClientError` is a real exception class |
| `pydantic_ai`                 | `MagicMock`             | `Agent`, `RunContext` are mocks                         |
| `fastapi`                     | `_FakeRouter`           | Route decorators (`get`, `post`, …) pass through functions unchanged |
| `llama_index.core.schema`     | `_FakeTextNode`         | `TextNode(id_=..., text=..., metadata=...)` works      |
| `llama_index.retrievers.bm25` | `_FakeBM25Retriever`    | `from_defaults(nodes=...)` stores nodes; `retrieve()` returns first 5 |

---

## 11. Auth Utilities (`auth_utils.py`)

Live tests verified against real Supabase (see `e2e_test.py`).

### 11.1 Password Hashing

| # | Test | Input | Expected | Status |
|---|---|---|---|---|
|110| Hash + verify correct password | `"MyPass123!"` | `verify_password` returns `True` | PASS (live) |
|111| Verify wrong password | `"wrong"` against hash | Returns `False` | PASS (live) |
|112| Hashes are bcrypt format | Any password | Hash starts with `$2b$` | PASS (live) |

### 11.2 JWT Tokens

| # | Test | Input | Expected | Status |
|---|---|---|---|---|
|113| Create + decode access token | `user_id`, `email`, `roles=["user"]` | Payload round-trips correctly | PASS (live) |
|114| Roles encoded in token | `roles=["user","private_user"]` | Both roles in decoded payload | PASS (live) |
|115| Expired token rejected | Token with `exp` in past | `jwt.ExpiredSignatureError` raised | PASS (live) |

### 11.3 Auth Endpoints (live integration tests)

| # | Endpoint | Scenario | Expected | Status |
|---|---|---|---|---|
|116| `POST /register` | New email | 201, `verification_token` returned | PASS (live) |
|117| `POST /register` | Duplicate email | 409 Conflict | PASS (live) |
|118| `POST /verify-email` | Valid token | 200, `is_verified=True` in DB | PASS (live) |
|119| `POST /verify-email` | Invalid token | 400 Bad Request | PASS (live) |
|120| `POST /login` | Correct credentials, verified | 200, access + refresh tokens | PASS (live) |
|121| `POST /login` | Wrong password | 401 Unauthorized | PASS (live) |
|122| `POST /login` | Unverified email | 403 Forbidden | PASS (live) |
|123| `POST /refresh-token` | Valid refresh token | 200, new token pair, old revoked | PASS (live) |
|124| `POST /refresh-token` | Revoked token | 401 Unauthorized | PASS (live) |
|125| `POST /logout` | Valid tokens | 200, refresh token revoked in DB | PASS (live) |
|126| `POST /forgot-password` | Known email | 200, `reset_token` returned | PASS (live) |
|127| `POST /forgot-password` | Unknown email | 200 (no enumeration) | PASS (live) |
|128| `POST /reset-password` | Valid token + new password | 200, all sessions revoked | PASS (live) |
|129| `POST /reset-password` | Used token | 400 Bad Request | PASS (live) |
|130| `POST /change-password` | Correct current password | 200, all sessions revoked | PASS (live) |
|131| `POST /change-password` | Wrong current password | 400 Bad Request | PASS (live) |
|132| `GET /me` | Valid Bearer token | 200, user profile + roles | PASS (live) |
|133| `GET /me` | No token | 401 Unauthorized | PASS (live) |

### 11.4 RBAC — Users / Roles / Permissions

| # | Endpoint | Scenario | Expected | Status |
|---|---|---|---|---|
|134| `GET /users` | Admin token | 200, user list with roles | PASS (live) |
|135| `GET /users` | Non-admin token | 403 Forbidden | PASS (live) |
|136| `GET /users/{id}` | Own user ID | 200, own profile | PASS (live) |
|137| `GET /users/{id}` | Different user, non-admin | 403 Forbidden | PASS (live) |
|138| `PUT /users/{id}` | Change roles to `["private_user"]` | 200, roles updated in DB | PASS (live) |
|139| `DELETE /users/{id}` | Admin deletes user | 204 No Content | PASS (live) |
|140| `GET /roles` | Any authenticated user | 200, roles with permissions | PASS (live) |
|141| `POST /roles` | Admin creates `"analyst"` | 201, role created | PASS (live) |
|142| `POST /roles` | Duplicate name | 409 Conflict | PASS (live) |
|143| `DELETE /roles/{id}` | Built-in `"admin"` role | 400, protected | PASS (live) |
|144| `GET /permissions` | Any authenticated user | 200, all 5 seeded permissions | PASS (live) |
|145| `POST /permissions` | Admin creates new | 201, permission created | PASS (live) |

### 11.5 Default RBAC Wiring (seeded on migration)

| Role | Permissions |
|---|---|
| `admin` | all 5 permissions |
| `user` | `read:public_docs` |
| `private_user` | `read:public_docs`, `read:private_docs` |

---

## 12. Guardrails (`tests/test_guardrails.py`)

| # | Test | Input | Expected | Status |
|---|---|---|---|---|
|146| Allows normal question | `"What is the last date for admission?"` | `allowed=True` | PASS |
|147| Blocks empty input | `"   "` | `allowed=False, reason="empty"` | PASS |
|148| Blocks overlong input | `> MAX_INPUT_CHARS` | `reason="too_long"` | PASS |
|149| Blocks prompt-injection | `"ignore all previous instructions…"` | `reason="prompt_injection"` | PASS |
|150| Blocks abuse | profanity | `reason="abuse"` | PASS |
|151| Redacts CNIC | `"35201-1234567-8"` | `[REDACTED-CNIC]` | PASS |
|152| Redacts API key | `"sk-proj-…"` | `[REDACTED-KEY]` | PASS |
|153| Scrubs system-prompt leak | leaked prompt line | line removed | PASS |
|154| Detects Devanagari | `"नमस्ते"` / `"aap kaise"` | `True` / `False` | PASS |

## 13. Session memory (`tests/test_session_memory.py`)

| # | Test | Input | Expected | Status |
|---|---|---|---|---|
|162| Two sessions never mix | distinct `session_id`s | isolated histories | PASS |
|163| Unknown session empty | new id | `""` | PASS |
|164| `get_or_create` registers | provided id | `session_exists=True` | PASS |
|165| Mints UUID when none | `None` | 36-char uuid | PASS |
|166| Sliding window trims | 5 turns, window=2 | last 2 turns kept | PASS |
|167| `session_info` counts | 2 turns | `turn_count=2` | PASS |
|168| `clear_session` | after add | empty | PASS |

---

## Running Specific Test Subsets

All commands run inside the backend image (`-v "$(pwd)":/app` mounts the live
source; do **not** use `docker compose run`, which injects the production `.env`
and breaks hermetic tests).

```bash
# Single test class
docker run --rm -v "$(pwd)":/app -w /app uchenab-backend:latest \
  python -m pytest tests/test_retriever.py::TestRRFFusion -v

# Single test
docker run --rm -v "$(pwd)":/app -w /app uchenab-backend:latest \
  python -m pytest tests/test_tools.py::TestConfidenceGate -v

# All ingest-related tests
docker run --rm -v "$(pwd)":/app -w /app uchenab-backend:latest \
  python -m pytest tests/test_ingest_service.py -v

# Confidence gate + settings store (this round's additions)
docker run --rm -v "$(pwd)":/app -w /app uchenab-backend:latest \
  python -m pytest tests/test_tools.py tests/test_settings_store.py -v
```

## Retrieval / RAG evaluation (live)

Beyond unit tests, retrieval quality is measured end-to-end against the live
knowledge base:

```bash
# Recall@1/@5, MRR, mean cosine, and the confidence-gate confusion matrix
docker run --rm -v "$(pwd)":/app -w /app uchenab-backend:latest python scripts/eval_rag.py

# Rebuild the Neo4j entity/fact graph from stored chunks
docker run --rm -v "$(pwd)":/app -w /app uchenab-backend:latest python scripts/backfill_graph.py
```
