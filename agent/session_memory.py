"""
Session memory with a sliding window — Redis-backed, in-process fallback.

Why Redis?
----------
The HTTP API may run as multiple worker processes / replicas, and Celery
ingestion workers run in separate processes too. An in-process dict would split a
single user's history across processes. Storing history in Redis (keyed by
``session_id``) means **every** process reads/writes the same per-session history,
so conversation continuity is consistent no matter which worker handles a turn.

Isolation & privacy
--------------------
History is keyed strictly by ``session_id`` (web chat → UUID or custom id). Two
users can never share a key, so no cross-user mixing.

Not permanent storage
---------------------
Only the last ``window_size`` turn-pairs are kept (older turns are dropped via
``LTRIM``), and each session key carries a rolling **TTL** (``SESSION_TTL_SECONDS``)
that is refreshed on every turn — active chats stay warm, idle ones expire. We do
not persist a durable chat transcript.

Fallback
--------
If Redis is unreachable (e.g. unit tests, or ``SESSION_MEMORY_BACKEND=memory``),
the manager transparently falls back to an in-process dict with identical
behaviour. The public API is unchanged either way.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Dict, List, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

# Default: 10 turns = 20 messages (user + assistant). Override via env.
_DEFAULT_WINDOW = int(os.getenv("CONTEXT_WINDOW_TURNS", "10"))


class SessionMemoryManager:
    """Per-session sliding-window history backed by Redis (in-process fallback).

    The public surface is intentionally identical to the previous in-process
    implementation so no call sites change.
    """

    def __init__(self, window_size: int = _DEFAULT_WINDOW) -> None:
        self._window_size = window_size
        self._ttl = int(os.getenv("SESSION_TTL_SECONDS", "86400"))  # 24h rolling
        self._prefix = os.getenv("SESSION_KEY_PREFIX", "uchenab:session:")
        # "redis" (default) or "memory". Auto-degrades to "memory" if Redis is
        # unreachable on first use.
        self._backend = os.getenv("SESSION_MEMORY_BACKEND", "redis").lower()
        self._redis = None
        self._redis_checked = False
        # In-process fallback store: session_id -> InMemoryChatMessageHistory
        self._histories: Dict[str, InMemoryChatMessageHistory] = {}

    # ── Backend resolution ────────────────────────────────────────────────────

    def _client(self):
        """Return a live Redis client, or None if Redis is unavailable.

        Created lazily (and per-process, which keeps Celery prefork children on
        their own connection pools). A failed ping permanently degrades this
        manager to the in-process backend.
        """
        if self._backend == "memory":
            return None
        if self._redis_checked:
            return self._redis
        self._redis_checked = True
        try:
            import redis  # redis-py (already a dependency)

            url = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
            client = redis.Redis.from_url(url, decode_responses=True)
            client.ping()
            self._redis = client
            logger.info("Session memory using Redis backend (%s)", url)
        except Exception as exc:
            logger.warning(
                "Session memory: Redis unavailable (%s) — falling back to in-process memory.",
                exc,
            )
            self._backend = "memory"
            self._redis = None
        return self._redis

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    # ── Public API ────────────────────────────────────────────────────────────

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session and return its ID.

        With Redis, an empty session has no key yet — it materialises on the
        first ``add_turn``. The returned id is stable and usable immediately.
        """
        sid = session_id or str(uuid.uuid4())
        if self._client() is None:
            self._histories.setdefault(sid, InMemoryChatMessageHistory())
        return sid

    def get_or_create(self, session_id: Optional[str]) -> str:
        """Return the given session_id (registering it), or mint a new UUID.

        ``create_session`` is idempotent: for the in-process backend it registers
        an empty history (so ``session_exists`` is true immediately) without
        clobbering an existing one; for Redis it is a no-op and the session
        materialises on the first ``add_turn``.
        """
        return self.create_session(session_id)

    def add_turn(self, session_id: str, user_message: str, ai_message: str) -> None:
        """Append a user + assistant turn and trim to the window (rolling TTL)."""
        client = self._client()
        if client is None:
            history = self._get_history(session_id)
            history.add_user_message(user_message)
            history.add_ai_message(ai_message)
            self._trim_memory(session_id)
            return

        key = self._key(session_id)
        pipe = client.pipeline()
        pipe.rpush(key, json.dumps({"role": "user", "content": user_message}))
        pipe.rpush(key, json.dumps({"role": "assistant", "content": ai_message}))
        pipe.ltrim(key, -self._window_size * 2, -1)  # keep last N turn-pairs
        pipe.expire(key, self._ttl)  # refresh rolling TTL
        pipe.execute()

    def get_messages(self, session_id: str) -> List:
        """Return LangChain message objects for this session (oldest → newest)."""
        client = self._client()
        if client is None:
            return list(self._get_history(session_id).messages)

        raw = client.lrange(self._key(session_id), 0, -1)
        messages: List = []
        for item in raw:
            try:
                obj = json.loads(item)
            except (ValueError, TypeError):
                continue
            if obj.get("role") == "user":
                messages.append(HumanMessage(content=obj.get("content", "")))
            else:
                messages.append(AIMessage(content=obj.get("content", "")))
        return messages

    def get_context_string(self, session_id: str) -> str:
        """Return the conversation history as a prompt-ready string ('' if new)."""
        messages = self.get_messages(session_id)
        if not messages:
            return ""
        lines = []
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def session_exists(self, session_id: str) -> bool:
        client = self._client()
        if client is None:
            return session_id in self._histories
        return client.exists(self._key(session_id)) == 1

    def clear_session(self, session_id: str) -> None:
        """Remove a session's history."""
        client = self._client()
        if client is None:
            self._histories.pop(session_id, None)
            return
        client.delete(self._key(session_id))

    def session_info(self, session_id: str) -> dict:
        """Return lightweight metadata about a session (for the /sessions endpoint)."""
        messages = self.get_messages(session_id)
        return {
            "session_id": session_id,
            "turn_count": len(messages) // 2,
            "message_count": len(messages),
            "window_size": self._window_size,
            "backend": self._backend,
        }

    # ── Private (in-process fallback helpers) ─────────────────────────────────

    def _get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._histories:
            self._histories[session_id] = InMemoryChatMessageHistory()
        return self._histories[session_id]

    def _trim_memory(self, session_id: str) -> None:
        history = self._histories[session_id]
        max_msgs = self._window_size * 2
        if len(history.messages) > max_msgs:
            history.messages = history.messages[-max_msgs:]


# Module-level singleton shared across the whole process
memory_manager = SessionMemoryManager()
