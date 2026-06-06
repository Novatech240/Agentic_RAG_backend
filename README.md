# Builtpulse Real Estate RAG — Enterprise Backend Services

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DD0031?style=for-the-badge&logo=redis&logoColor=white)
![Celery](https://img.shields.io/badge/Celery-37814A?style=for-the-badge&logo=celery&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![AWS S3](https://img.shields.io/badge/Amazon_S3-569A31?style=for-the-badge&logo=amazons3&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

The **Builtpulse Real Estate Assistant** backend is a state-of-the-art, enterprise-grade AI engine designed to assist property buyers, sellers, and agents. Powered by **Pydantic AI** and **FastAPI**, it runs a **three-stage retrieval pipeline**: a **Vespa** hybrid engine that fuses true **BM25** lexical search and **dense HNSW** vector search with **Reciprocal Rank Fusion (RRF)** in a single query, a **cross-encoder reranker** (Voyage) precision stage, and a **Neo4j Knowledge Graph** for structured entity/fact lookups. **Supabase pgvector** is the durable source of truth and an automatic fallback if Vespa is unreachable.

Answers pass through **three safety tiers**—a scope classifier, a cosine-similarity confidence gate, and post-generation citation enforcement—so the assistant abstains rather than hallucinates. The system features real-time asynchronous background pipelines running on **Celery + Redis**, strict guardrails (prompt-injection filters and CNIC/PII redaction), and robust connection caching. Users and agents test the agent via the web **Assistant** chat UI.

---

## 🏗️ Architectural Blueprints

### 0. End-to-End Architecture (Ingestion → Answer)
The complete pipeline in one view: documents are ingested, deduplicated, embedded
and indexed into three stores; queries then rewrite, cache, retrieve, fuse, and
pass through four anti-hallucination tiers before answering. The numbered sections
below break each half down in detail.

```mermaid
graph TD
    %% ───────────── INGESTION ─────────────
    subgraph ING["Ingestion — Celery workers"]
        direction TB
        SRC[(S3 buckets<br/>public / private)] --> ITASK["Celery ingest task"]
        UP["API upload / S3 webhook"] --> ITASK
        ITASK --> PARSE["Parse PDF / DOCX / XLSX / PPTX / CSV / HTML"]
        PARSE --> DHASH{"Doc SHA-256<br/>changed?"}
        DHASH -->|No| SKIP["Skip — no re-embed"]
        DHASH -->|Yes| CHUNK["Semantic chunking<br/>LLM-assisted; simple/tabular fallback"]
        CHUNK --> CDIFF{"Per-chunk<br/>hash diff"}
        CDIFF -->|removed| DELC["Delete stale chunks"]
        CDIFF -->|"new / changed"| EMB["OpenAI embeddings<br/>text-embedding-3-large, 3072-d"]
        EMB --> FACTS["LLM entity / fact extraction"]
    end

    %% ───────────── STORES ─────────────
    subgraph STORE["Stores"]
        direction LR
        PG[(Supabase pgvector<br/>source of truth)]
        VES[(Vespa index<br/>BM25 + dense HNSW + RRF)]
        NEO[(Neo4j graph<br/>Entity-Fact-Chunk)]
    end

    EMB --> PG
    PG --> FEED["Feed chunks (shared ids)"] --> VES
    FACTS --> NEO
    DELC --> PG
    DELC -.->|delete| VES
    EMB --> BUMP["Bump cache version<br/>(invalidate answer + retrieval caches)"]

    %% ───────────── QUERY / ANSWER ─────────────
    subgraph QRY["Query → Answer"]
        direction TB
        UI([Assistant UI]) --> API["POST /api/v1/chat<br/>(/chat/stream SSE)"]
        API --> REW["Rewrite → canonical query<br/>(history-resolved, fail-open)"]
        REW --> AC{"Turn answer cache<br/>(canonical query + scope)"}
        AC -->|Hit| OUT
        AC -->|Miss| SCOPE{"Tier 1 — scope<br/>(LLM, history-aware)"}
        SCOPE -->|out of scope| DECL["Decline"]
        SCOPE -->|in scope| GIN{"Input guardrails<br/>injection / abuse (fail-closed)"}
        GIN -->|blocked| REFUSE["Refusal"]
        GIN -->|allowed| AGENT["Pydantic AI agent<br/>(tool selection)"]
        AGENT --> RET["Retrieval:<br/>cache 0 → embed → Vespa hybrid (RRF)<br/>/ Postgres fallback → cross-encoder rerank"]
        AGENT --> GS["Graph fact lookup"]
        RET --> GEN["LLM generates cited answer"]
        GS --> GEN
        GEN --> T2{"Tier 2 — confidence<br/>top cosine >= 0.25?"}
        T2 -->|low| ESC["Escalate + alert admin"]
        T2 -->|ok| T3{"Tier 3 — citations valid?"}
        T3 -->|"fabricated / uncited"| ABS["Abstain"]
        T3 -->|ok| T4{"Tier 4 — groundedness<br/>chunks entail claims?"}
        T4 -->|grounded| GOUT
        T4 -->|"ungrounded → remediate"| REM{"Strip unsupported claims;<br/>grounded remainder survives?"}
        REM -->|yes| GOUT["Store answer cache<br/>(verified-good only)"]
        REM -->|no| ABS
        GOUT --> OG["Output guardrails<br/>(PII redaction, leak scrub)"]
        ABS --> OG
        ESC --> OG
        OG --> OUT["Final answer + provenance"]
        OUT --> PERSIST["Persist turn<br/>(Supabase messages + Redis memory)"]
    end

    %% Query reads from the shared stores
    RET -.->|primary| VES
    RET -.->|fallback| PG
    GS -.->|reads| NEO
```

---

### 1. Web Chat RAG Flow
Users and agents test the agent via the **Assistant** UI (`/dashboard/chat`), which calls `POST /api/v1/chat` (or `/chat/stream` for SSE). The pipeline retrieves and fuses context from hybrid document search and the Neo4j knowledge graph before generating a grounded answer.

```mermaid
graph TD
    UI([Assistant UI<br/>/dashboard/chat]) --> API["POST /api/v1/chat<br/>(or /chat/stream SSE)"]
    API --> Session["Resolve session_id<br/>(Redis / in-process memory)"]

    Session --> Canon["Conversational query rewrite<br/>condense history + turn → canonical query<br/>(fail-open)"]
    Canon --> ACache{"Turn-level answer cache<br/>(Redis; canonical query + scope)"}
    ACache -->|Hit| CacheOut["Return cached answer + tools<br/>(zero model calls; admin bypass)"]
    ACache -->|Miss| Scope{"Tier 1 — Scope classifier<br/>(LLM, history-aware)"}
    Scope -->|Out of scope| Refuse["Return out_of_scope_message<br/>(from app_settings)"]
    Scope -->|In scope| InputG{"Input guardrails<br/>(length / injection / abuse)<br/>fail-closed"}

    InputG -->|Blocked| Blocked["Return refusal<br/>(no tools called)"]
    InputG -->|Allowed| Memory[(Redis session memory<br/>sliding-window history)]

    Memory --> Prompt["Build prompt<br/>history + canonical question"]
    Prompt --> Agent["Pydantic AI RAG Agent<br/>(OpenAI GPT-4o-mini / GPT-4o)"]

    Agent --> Tools{"Agent selects tools"}
    Tools --> DocTool["search_documents"]
    Tools --> GraphTool["search_knowledge_graph_facts"]
    Tools --> ListTool["list_available_documents"]

    subgraph Retrieval["Three-stage retrieval (search_documents)"]
        direction TB
        DocTool --> Cache{"Stage 0 — retrieval cache<br/>(Redis; normalized query + scope)"}
        Cache -->|Hit| Context
        Cache -->|Miss| Embed["1 - Embed query<br/>(text-embedding-3-large, 3072-d)"]
        Embed --> Vespa["2 - Vespa hybrid recall<br/>BM25 + dense HNSW → RRF<br/>(widened candidate pool)"]
        Vespa -.->|Vespa down / empty| PGFallback["Postgres hybrid_search RPC<br/>(pgvector + tsvector fallback)"]
        Vespa --> Rerank["3 - Cross-encoder rerank<br/>(Voyage rerank-2-lite, fail-open)<br/>→ top-k by true relevance"]
        PGFallback --> Rerank
        Rerank --> Store["cache result (TTL; ingest-invalidated)"]
    end

    GraphTool --> Neo4j["Neo4j full-text factIndex<br/>(entity / fact lookup)"]

    Store --> Context["Retrieved chunks + graph facts<br/>(stored in AgentDependencies)"]
    Neo4j --> Context

    Context --> LLM["LLM generates grounded answer<br/>(inline [n] citation contract)"]

    LLM --> Conf{"Tier 2 — Confidence gate<br/>top cosine >= 0.25?"}
    Conf -->|Low confidence| Alert["Celery: notify_admin_weak_context_task"]
    Alert --> Escalate["Return support-escalation message"]
    Conf -->|High confidence| Cite{"Tier 3 — Citation enforcement<br/>fabricated / uncited claim?"}
    Cite -->|Fails| Abstain["Swap for ABSTAIN message"]
    Cite -->|Passes| Ground{"Tier 4 — Groundedness judge<br/>do cited chunks ENTAIL every claim?"}
    Ground -->|Grounded| Store2["Cache verified answer<br/>(only good answers)"]
    Ground -->|"Ungrounded → remediate"| Remed{"Strip unsupported claims;<br/>grounded remainder survives?"}
    Remed -->|Yes| Store2
    Remed -->|No| Abstain
    Store2 --> OutG["Output guardrails<br/>(prompt-leak scrub, PII redaction)"]
    Abstain --> OutG

    OutG --> Answer["Final answer + provenance"]

    Answer --> Log["Persist turn + provenance<br/>(Supabase messages + Redis memory)"]
    CacheOut --> Log
    Escalate --> Log
    Refuse --> Log
    Blocked --> Log

    Log --> UI
```

| Stage | What it does |
|--------|----------------|
| **Query rewrite** | Condenses multi-turn follow-ups into one canonical, self-contained query — reused for the cache key, scope check, and retrieval (fail-open) |
| **Answer cache** | Turn-level Redis cache keyed on the canonical query + scope; a hit returns the cached answer with **zero model calls**. Only verified-good answers stored; admin requests bypass; ingest-invalidated |
| **Scope (Tier 1)** | LLM classifier (history-aware) blocks off-topic questions before retrieval (admin-toggleable) |
| **Input guardrails** | Fail-closed screen for injection, abuse, empty/oversized input |
| **Retrieval** | **(0)** retrieval cache check → **(1)** embed query → **(2)** Vespa BM25+dense+RRF recall (Postgres fallback) → **(3)** cross-encoder rerank to top-k → cache store |
| **Graph** | Optional Neo4j `factIndex` lookup for structured entity/fact relationships |
| **Confidence (Tier 2)** | If top pgvector cosine `< 0.25`, escalate to an admin instead of trusting the answer |
| **Citation (Tier 3)** | Verifies every claim cites a real retrieved chunk; abstains on fabricated/uncited answers |
| **Groundedness (Tier 4)** | LLM judge confirms the cited chunks actually *entail* each claim; **remediates** by stripping unsupported claims (keeping the grounded remainder), abstaining only when nothing survives |
| **Output guardrails** | Scrubs prompt leaks, redacts PII, optional script normalization |
| **Logging** | Saves user/assistant turns and debug provenance to Supabase + Redis |

---

### 2. Session Memory & Cache Lifecycle (Redis)
Redis backs three things per turn: the sliding-window conversation **history**
(shared across stateless API/worker processes, in-process fallback if offline),
the turn-level **answer cache**, and the **retrieval cache**. The sequence below
shows how a turn uses them — a cache hit returns with zero model calls.

```mermaid
sequenceDiagram
    autonumber
    participant UI as Assistant UI
    participant API as FastAPI /chat
    participant Redis as Redis (history + caches)
    participant Agent as RAG Agent

    UI->>API: message + session_id
    API->>Redis: load sliding-window history (builtpulse:session:*)
    Redis-->>API: prior turns (or empty)
    API->>API: rewrite → canonical query (history-resolved)
    API->>Redis: GET answer cache (canonical query + scope)
    alt Cache hit
        Redis-->>API: cached answer + tools
        API-->>UI: answer (zero model calls)
    else Cache miss
        API->>Agent: history + canonical question + tools
        Note over Agent: scope → guardrails → retrieval<br/>→ 4 anti-hallucination tiers
        Agent-->>API: grounded, gated answer
        API->>Redis: SETEX answer cache (verified-good only, TTL)
        API-->>UI: JSON / SSE response
    end
    API->>Redis: append turn + trim window + refresh TTL
```

---

### 3. Document Ingestion Pipeline
When files are uploaded via the API or synced periodically from S3, Celery workers parse, chunk, embed, and index them into Supabase pgvector and Neo4j.

```mermaid
graph LR
    S3[(AWS S3 buckets)] --> Ingest["Celery ingest tasks"]
    Upload["API upload / S3 webhook"] --> Ingest
    Ingest --> Parse["Parse PDF/DOCX/XLSX/PPTX/HTML"]
    Parse --> Chunk["Semantic chunk text"]
    Chunk --> Embed["OpenAI embeddings"]
    Embed --> PG[(Supabase pgvector<br/>source of truth)]
    Embed --> Vespa[(Vespa<br/>primary retrieval index)]
    Chunk --> Graph["Neo4j graph builder"]
    Graph --> Neo4j[(Neo4j)]
```

Detailed ingest steps (with content-hash dedup + chunk-level diffing):

```mermaid
graph TD
    Doc([PDF, DOCX, XLSX, PPTX, CSV, HTML]) --> Upload["API Upload / S3 Bucket Webhook"]
    Upload --> Temp["Local Temp Directory"]
    Temp --> Parse["Parsers (pypdf, python-docx, openpyxl, beautifulsoup4)"]
    Parse --> CeleryTask["Celery Task (ingest_document_task)"]

    CeleryTask --> DocHash{"Source exists &<br/>doc SHA-256 unchanged?"}
    DocHash -->|Yes| Skip["Skip entirely<br/>(no re-embed)"]
    DocHash -->|"No (new / changed)"| Chunk["Semantic chunking<br/>(LLM-assisted; simple/tabular fallback)"]

    Chunk --> Diff{"Per-chunk hash diff<br/>vs stored chunks"}
    Diff -->|Unchanged chunks| Keep["Keep as-is"]
    Diff -->|Disappeared chunks| Del["Delete from Postgres + Vespa"]
    Diff -->|New / changed chunks| Embed["Generate OpenAI vector embeddings<br/>(text-embedding-3-large, 3072-d)"]

    Embed --> Postgres[("PostgreSQL<br/>(pgvector chunks upsert — source of truth)")]
    Postgres --> Feed["Feed chunks to Vespa<br/>(shared chunk ids; BM25 + HNSW indexes)"]
    Feed --> VespaIdx[("Vespa primary index<br/>backfillable from Postgres")]
    Embed --> GraphBuild["graph_builder: LLM entity/fact extraction<br/>(fail-open)"]
    GraphBuild --> Neo4jOut[("Neo4j: Entity - HAS_FACT → Fact - FROM_CHUNK → Chunk<br/>full-text factIndex + chunkIndex")]
    Feed --> Bump["Bump cache version<br/>(invalidate answer + retrieval caches)"]
```

---

## 🧭 Hardening Roadmap — making retrieval stronger

The pipeline above is solid (hybrid recall, cross-encoder rerank, fail-open
everywhere). The first wave of hardening — **query rewriting, a retrieval cache,
and a groundedness judge** — is now **implemented** (✅ below). The remaining
items are where accuracy, cost, and robustness improve most next.

```mermaid
graph TD
    Q([User turn]) --> R1["✅ Conversational query rewrite<br/>condense history → standalone query"]
    R1 --> R2["✅ Retrieval cache (Redis)<br/>skip embed+retrieve on repeat/near-dup"]
    R2 --> R3["③ Contextual Retrieval<br/>prepend doc/section context to each chunk<br/>before embedding"]
    R3 --> R4["Vespa hybrid + rerank<br/>(existing)"]
    R4 --> R5["✅ Groundedness / NLI check (Tier 4)<br/>(semantic faithfulness, not just regex)"]
    R5 --> R6["⑤ GraphRAG fusion<br/>merge graph facts + chunks into one ranked context"]
    R6 --> A([Answer])

    subgraph Cross-cutting
        E["⑥ Continuous eval in CI<br/>(golden set: recall@k, MRR, faithfulness)"]
        S["⑦ Model-based injection guard<br/>(prompt-guard atop regex)"]
        M["⑧ Self-query metadata filters<br/>(date / type / department)"]
    end
```

| # | Enhancement | Status | Why it makes the RAG stronger | Where it lives |
|---|-------------|--------|-------------------------------|----------------|
| 1 | **Conversational query rewriting** | ✅ Done | Follow-ups like *"and for masters?"* retrieved poorly; now condensed once to a canonical query reused for cache/scope/retrieval. | `agent/query_rewriter.py` → hoisted in `api.chat`/`chat_stream` + `conversation.py` |
| 2 | **Turn-level answer cache** | ✅ Done | Memoizes the **whole turn** (answer + tools) keyed on the canonical query, so repeats skip scope+agent+retrieval+judge (live: ~15s → ~1.5s). Only good answers cached; admin-bypass; ingest-invalidated. Retrieval cache (`query_cache`) remains as a second tier. | `agent/answer_cache.py` (+ `agent/query_cache.py`) → wired in both chat paths + worker |
| 4 | **Groundedness check + remediation (Tier 4)** | ✅ Done | LLM judge confirms the cited chunk *entails* each claim; **strips only the unsupported claims** (e.g. a fabricated "Google Wallet") and keeps the grounded remainder, abstaining only when nothing survives. | `agent/groundedness.py` (`assess`/`enforce`) → wired in both gates |
| 3 | **Contextual Retrieval** | Planned | Chunks lose surrounding context once split, hurting recall on terse chunks. Prepend a short doc/section summary to each chunk *before* embedding (Anthropic's contextual-retrieval pattern). | `ingestion/chunker.py` + `embedder.py` |
| 5 | **GraphRAG fusion** | Planned | `search_documents` and `search_knowledge_graph_facts` are independent tools the model may not combine. Fuse graph facts + vector chunks into one ranked context for multi-hop questions. | `tools.py` / `retriever.py` |
| 6 | **Continuous eval in CI** | Planned | `scripts/eval_rag.py` exists but runs ad-hoc. Promote a fixed golden Q→chunk set into CI tracking recall@k, MRR, and faithfulness to catch regressions on every change. | CI workflow + `scripts/eval_rag.py` |
| 7 | **Model-based injection guard** | Planned | `guardrails._INJECTION_RE` is regex — trivially bypassed by paraphrase. Layer a small prompt-injection classifier (e.g. prompt-guard) over the regex fast-path. | `agent/guardrails.py` |
| 8 | **Self-query metadata filters** | Planned | Retrieval only filters on `access_level`. Let the agent emit structured filters (date / doc-type / department) into the Vespa YQL for precise scoping. | `vespa_client.search` YQL + tool schema |
| 9 | **Expanded PII + rate limiting** | Planned | PII redaction covers CNIC/card/key/JWT only; add email/phone. Add per-session/user rate limiting to blunt abuse and cost spikes. | `guardrails.redact_pii` + API middleware |
| 10 | **Vespa-native reranking** | Planned | The Voyage rerank adds a network hop per query. Move to a Vespa global-phase ONNX cross-encoder to rerank in-engine and drop the external dependency. | `vespa/schemas/chunk.sd` global-phase |

> None of these change the public API or break the fail-open guarantees — each is
> an additive, independently shippable layer.

### Scaling to ~100M documents

Scale is a **retrieval-engine** concern, independent of the agent/orchestration
layer (LangGraph, Pydantic AI, etc. have no bearing on it):

* **Vespa** is the component that scales — it's content-addressable and shards
  horizontally. 100M chunks × 3072-d `bfloat16` HNSW ≈ a multi-node content
  cluster; grow `<nodes>` in `services.xml` and Vespa redistributes. Keep the
  reranker's candidate pool bounded (≤100) so per-query cost stays flat as the
  corpus grows.
* **Postgres/pgvector** stays the durable source of truth and the rebuild source
  for Vespa (`scripts/backfill_vespa.py`) — it is *not* on the hot query path at
  scale, so its index size doesn't gate query latency.
* **Cost at scale** is dominated by embeddings (ingest, one-time per chunk) and
  per-turn LLM calls (scope + rewrite + agent + groundedness). The retrieval
  cache blunts repeat-query cost; set `GROUNDEDNESS_CHECK_ENABLED` per your
  hallucination-vs-cost tolerance — it is the largest per-turn add.

---

## 📚 References & Developer Documentation
To deep-dive into endpoints testing strategies or to check the QA test matrices, refer to our comprehensive internal documentation:
* 📄 **[API Testing Guide](apitesting.md)**: Detailed step-by-step specifications of all REST operations, Celery hooks, auth handshakes, and postman testing guidelines.
* 📋 **[QA Test Cases Log](testcases.md)**: Fully granular testing matrices, expected assertions, input/output boundary cases, and performance criteria log.

---

## 🛠️ Ingestion & Setup

### Service Map
* `api`: FastAPI application, serving all routes under `/api/v1/*` (port `8058`).
* `worker`: Celery worker performing document ingestion and background alerts.
* `beat`: Celery scheduler driving periodic 15-minute S3 synchronizations.
* `redis`: High-speed message broker, Celery backend, session memory, and endpoint/avatar cache store.

---

### 🚀 Running with Docker (Recommended)

1. Set up configurations:
   ```bash
   cp .env.example .env
   ```
2. Build and run all services:
   ```bash
   docker compose up --build
   ```
3. Access API Documentation at [http://localhost:8058/api/docs](http://localhost:8058/api/docs).

---

### 💻 Local (Non-Containerized) Setup

Ensure **PostgreSQL**, **Neo4j**, and **Redis** servers are running locally.

1. Initialize a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the services across separate terminals:
   * **Terminal 1 (API)**:
     ```bash
     python -m uvicorn agent.api:app --reload --port 8058
     ```
   * **Terminal 2 (Celery Worker)**:
     ```bash
     celery -A worker.celery_app:celery_app worker -Q ingestion --loglevel=info
     ```
   * **Terminal 3 (Celery Scheduler)**:
     ```bash
     celery -A worker.celery_app:celery_app beat --loglevel=info
     ```

---

## 🧪 Testing Suite
Execute the testing framework using a clean container build:
```bash
docker run --rm -v "$(pwd)":/app -w /app builtpulse-backend:latest python -m pytest -q
```
