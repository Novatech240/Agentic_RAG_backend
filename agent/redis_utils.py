import os
import logging
import redis

logger = logging.getLogger(__name__)

_redis_client = None
_redis_checked = False


def get_redis_client() -> redis.Redis | None:
    """Return a shared Redis client, or None if Redis is unavailable.

    Uses lazy initialization so that Celery children get their own connection pool.
    Auto-degrades to None if Redis is unreachable on first use.
    """
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client

    _redis_checked = True
    backend = os.getenv("SESSION_MEMORY_BACKEND", "redis").lower()
    if backend == "memory":
        logger.info("Redis disabled via SESSION_MEMORY_BACKEND=memory.")
        _redis_client = None
        return None

    try:
        url = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
        # We set decode_responses=False so we can safely store raw binary files (avatars)
        client = redis.Redis.from_url(url, decode_responses=False)
        client.ping()
        _redis_client = client
        logger.info("Shared Redis client connected successfully to %s", url)
    except Exception as exc:
        logger.warning(
            "Redis is unavailable (%s) — caching will fall back to direct database reads.",
            exc,
        )
        _redis_client = None

    return _redis_client
