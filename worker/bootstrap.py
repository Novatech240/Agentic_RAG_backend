"""
Worker bootstrap — bridges Celery's synchronous task model to our async stack.

The whole data layer (Supabase async client, Neo4j async driver) is async, but
Celery executes tasks synchronously in prefork child processes.  We therefore:

1. Keep **one persistent event loop per worker process** (creating a fresh loop
   per task would orphan the Supabase client, which is bound to the loop that
   created it).
2. Lazily run **one-time async initialization** (DB + graph) on that loop the
   first time a task runs in the process.

`run_async(coro)` is the single entry point every task uses.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None
_initialized: bool = False


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return the process-wide event loop, creating it on first use."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


async def _ensure_initialized() -> None:
    """Initialize DB + graph connections once per worker process (idempotent)."""
    global _initialized
    import agent.db_utils as db

    if _initialized and db._client is not None:
        return

    from agent.db_utils import initialize_database
    from agent.graph_utils import initialize_graph

    await initialize_database()
    logger.info("Worker: Supabase client ready")
    try:
        await initialize_graph()
        logger.info("Worker: Neo4j ready")
    except Exception as exc:  # graph is optional for ingestion to proceed
        logger.warning("Worker: graph init failed (continuing): %s", exc)

    _initialized = True


def run_async(coro):
    """Run an async coroutine to completion on the persistent worker loop.

    Ensures connections are initialized before the coroutine executes so every
    task can assume a ready data layer.
    """
    loop = _get_loop()

    async def _runner():
        await _ensure_initialized()
        return await coro

    return loop.run_until_complete(_runner())
