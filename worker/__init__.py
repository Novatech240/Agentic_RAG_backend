"""Celery worker package — async task queue for ingestion and messaging.

Note: we intentionally do NOT re-export `celery_app` here, to avoid shadowing
the `worker.celery_app` submodule (Celery is launched via
`-A worker.celery_app:celery_app`, which needs that dotted path to resolve to
the module, not an attribute).
"""
