"""
Celery tasks — the actual units of background work.

* **Ingestion** (`ingest_*`) — wrap `IngestService` so document indexing runs
  off the request path, with automatic retry/backoff on transient failures.

Every task delegates its async body through `run_async` (see bootstrap.py),
which guarantees the data layer is initialized on the worker's event loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .bootstrap import run_async
from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Retry policy shared by ingestion tasks — transient S3/OpenAI/graph hiccups.
_INGEST_RETRY = dict(
    autoretry_for=(Exception,),
    retry_backoff=True,  # 1s, 2s, 4s, …
    retry_backoff_max=300,  # cap at 5 minutes
    retry_jitter=True,
    max_retries=5,
)


# ── Ingestion tasks ───────────────────────────────────────────────────────────


@celery_app.task(name="worker.tasks.ingest_all_task", bind=True, **_INGEST_RETRY)
def ingest_all_task(self) -> Dict[str, int]:
    """Full sweep of every configured S3 bucket (private + public)."""
    from ingestion.ingest_service import ingest_service

    logger.info("Task ingest_all_task starting (id=%s)", self.request.id)
    stats = run_async(ingest_service.ingest_all_s3_buckets())
    result = {
        "inserted": stats.inserted,
        "updated": stats.updated,
        "skipped": stats.skipped,
        "failed": stats.failed,
        "total": stats.total,
    }
    logger.info("Task ingest_all_task complete: %s", result)
    return result


@celery_app.task(name="worker.tasks.ingest_bucket_task", bind=True, **_INGEST_RETRY)
def ingest_bucket_task(
    self, bucket_type: str = "private", prefix: str = ""
) -> Dict[str, int]:
    """Ingest a single bucket / prefix."""
    from ingestion.ingest_service import ingest_service

    logger.info(
        "Task ingest_bucket_task '%s/%s' (id=%s)", bucket_type, prefix, self.request.id
    )
    stats = run_async(
        ingest_service.ingest_from_s3(bucket_type=bucket_type, prefix=prefix)
    )
    return {
        "bucket_type": bucket_type,
        "prefix": prefix,
        "inserted": stats.inserted,
        "updated": stats.updated,
        "skipped": stats.skipped,
        "failed": stats.failed,
    }


@celery_app.task(name="worker.tasks.ingest_single_s3_task", bind=True, **_INGEST_RETRY)
def ingest_single_s3_task(self, s3_key: str, bucket_name: str) -> Dict[str, Any]:
    """Ingest one S3 object (used by the S3/SNS webhook)."""
    from ingestion.ingest_service import ingest_service

    logger.info("Task ingest_single_s3_task '%s' (id=%s)", s3_key, self.request.id)
    result = run_async(ingest_service.ingest_single_s3_object(s3_key, bucket_name))
    return {
        "source": result.source,
        "status": result.status,
        "document_id": result.document_id,
        "chunks_created": result.chunks_created,
        "error": result.error,
    }


@celery_app.task(name="worker.tasks.ingest_document_task", bind=True, **_INGEST_RETRY)
def ingest_document_task(
    self,
    content: str,
    source: str,
    title: str,
    metadata: Optional[Dict[str, Any]] = None,
    access_level: str = "public",
) -> Dict[str, Any]:
    """Ingest an already-parsed document (used by the upload endpoint)."""
    from ingestion.ingest_service import ingest_service

    logger.info("Task ingest_document_task '%s' (id=%s)", source, self.request.id)
    result = run_async(
        ingest_service.ingest_document(
            content=content,
            source=source,
            title=title,
            metadata=metadata or {},
            access_level=access_level,
        )
    )
    return {
        "source": result.source,
        "status": result.status,
        "document_id": result.document_id,
        "chunks_created": result.chunks_created,
        "error": result.error,
    }


@celery_app.task(name="worker.tasks.notify_admin_weak_context_task")
def notify_admin_weak_context_task(session_id: str, user_query: str) -> Dict[str, Any]:
    """Background task to alert admins about a low-confidence chat query."""
    logger.info(
        "ADMIN ALERT [Celery Task]: Session %s asked low-confidence question: '%s'",
        session_id,
        user_query,
    )
    return {"status": "notified", "session_id": session_id, "query": user_query}
