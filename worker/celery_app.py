"""
Celery application — the enterprise async task queue.

Why Celery + Redis?
-------------------
Document ingestion (download → parse → chunk → embed → upsert → graph) is
slow, bursty, and must not block the API request/response cycle.  Running
them as Celery tasks gives us:

* **Durability** — tasks survive an API restart (they live in Redis).
* **Retries with backoff** — transient S3 / OpenAI / Graph errors auto-retry.
* **Horizontal scale** — add more `worker` containers to ingest faster.
* **Scheduling** — Celery Beat replaces the old asyncio polling loop.
* **Observability** — every task has an id you can poll for status/result.

Topology
--------
    FastAPI (api)  ──enqueue──▶  Redis (broker)  ──▶  Celery worker(s)
                                      ▲
    Celery Beat  ──periodic enqueue───┘

Configuration is entirely env-driven so the same image runs as api, worker,
or beat depending on the container command.
"""

from __future__ import annotations

import os
from pathlib import Path

from celery import Celery
from celery.schedules import crontab  # noqa: F401  (available for custom schedules)
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Broker / backend ──────────────────────────────────────────────────────────
# A single REDIS_URL drives both broker and result backend by default, but each
# may be overridden independently for advanced deployments.
# `… or REDIS_URL` (not getenv defaults) so a present-but-blank env var — e.g.
# `CELERY_BROKER_URL=` in .env — still falls back to Redis instead of Celery's
# default RabbitMQ broker.
REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL") or REDIS_URL
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND") or REDIS_URL

# How often Beat fires the full S3 ingest sweep (minutes).
INGEST_INTERVAL_MINUTES = int(os.getenv("INGEST_INTERVAL_MINUTES", "15"))

celery_app = Celery(
    "uchenab",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["worker.tasks"],
)

celery_app.conf.update(
    # ── Serialization (JSON only — never pickle untrusted payloads) ───────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone=os.getenv("CELERY_TIMEZONE", "UTC"),
    enable_utc=True,
    # ── Reliability ───────────────────────────────────────────────────────────
    task_acks_late=True,  # re-deliver if a worker dies mid-task
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # fair dispatch for long-running tasks
    task_track_started=True,  # expose STARTED state for polling
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "86400")),  # 24h
    # ── Safety limits ─────────────────────────────────────────────────────────
    task_soft_time_limit=int(os.getenv("CELERY_SOFT_TIME_LIMIT", "1500")),  # 25m
    task_time_limit=int(os.getenv("CELERY_TIME_LIMIT", "1800")),  # 30m
    broker_connection_retry_on_startup=True,
)

# ── Periodic schedule (Celery Beat) ───────────────────────────────────────────
# Replaces the old in-process asyncio auto-ingest loop. Disable by setting
# AUTO_INGEST_ENABLED=false (e.g. when driving ingestion purely via webhooks).
if os.getenv("AUTO_INGEST_ENABLED", "true").lower() == "true":
    celery_app.conf.beat_schedule = {
        "auto-ingest-s3-sweep": {
            "task": "worker.tasks.ingest_all_task",
            "schedule": INGEST_INTERVAL_MINUTES * 60.0,
            "options": {"queue": "ingestion", "expires": INGEST_INTERVAL_MINUTES * 60},
        }
    }

# ── Queue routing ─────────────────────────────────────────────────────────────
celery_app.conf.task_routes = {
    "worker.tasks.ingest_*": {"queue": "ingestion"},
}
