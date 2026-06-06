# Vespa application package

Retrieval engine for the RAG pipeline: true BM25 + dense HNSW + RRF fusion in a
single query. Canonical document rows stay in Postgres; only chunk retrieval
lives here. See the project memory `rag-vespa-migration` for the full design.

```
vespa/
├── services.xml        # cluster topology (1 node; scale via <nodes>)
├── hosts.xml           # local single-node host mapping (ignored on Cloud)
└── schemas/
    └── chunk.sd        # chunk document + semantic/bm25/hybrid rank-profiles
```

## Local dev / EC2 (Docker) — automated, one command

The stack self-deploys this package. `docker compose up` starts the `vespa`
engine, then a one-shot **`vespa-deploy`** init service (`scripts/deploy_vespa.py`)
waits for the config server, zips this directory, and activates the schema. The
`api` and `worker` services gate on its successful completion, so:

```bash
docker compose up -d                 # vespa → deploy schema → api/worker start
```

No Vespa CLI needed — the deploy is a plain HTTP POST to the config server and
is **idempotent** (safe to re-run on every boot). Schema edits redeploy on the
next `up` without an image rebuild (the package is bind-mounted into the deploy
service). To redeploy manually after editing a schema:

```bash
docker compose run --rm --no-deps vespa-deploy
```

Query/feed endpoint: `http://localhost:8080` · config/deploy: `http://localhost:19071`.

> First boot leaves Vespa empty. New documents populate it via the normal
> ingestion path; to migrate chunks already in Postgres, run the backfill once:
> `docker compose run --rm --no-deps api python -m scripts.backfill_vespa`.

### Manual deploy with the Vespa CLI (optional)

```bash
docker compose up -d vespa
vespa config set target local && vespa deploy --wait 300 ./vespa && vespa status
```

## Production (Vespa Cloud)

```bash
vespa config set target cloud
vespa config set application <tenant>.<app>.default
vespa auth login
vespa deploy --wait 600 ./vespa
```

Same package; Cloud provisions nodes itself. `hosts.xml` is ignored there.

## Hybrid query shape

Both legs in one request (vector OR text → fused by the `hybrid` profile):

```
yql:  select * from chunk
      where ({targetHits:100}nearestNeighbor(embedding, q))
            or userInput(@query)
ranking.profile: hybrid
input.query(q): <3072-d float embedding>
query: <raw user text for BM25>
```

Private-doc filtering: add `and access_level contains "<user_id|public>"`.
