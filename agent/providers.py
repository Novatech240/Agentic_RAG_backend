"""
OpenAI LLM + Embedding providers.

The system uses OpenAI exclusively for both chat completions and embeddings.
A single ``OPENAI_API_KEY`` drives both (legacy ``LLM_API_KEY`` /
``EMBEDDING_API_KEY`` aliases are still accepted for backward compatibility).
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

LLM_CHOICE = os.getenv("LLM_CHOICE", "gpt-4o-mini")
INGESTION_LLM_CHOICE = os.getenv("INGESTION_LLM_CHOICE", LLM_CHOICE)
_EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Single API key — OPENAI_API_KEY is canonical; legacy aliases still accepted.
_OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("LLM_API_KEY")
    or os.getenv("EMBEDDING_API_KEY")
    or ""
)


def get_embedding_model() -> str:
    """Return the embedding model name string."""
    return _EMBEDDING_MODEL_NAME


def get_ingestion_model() -> str:
    """Return the Pydantic AI model string for the ingestion LLM."""
    return f"openai:{INGESTION_LLM_CHOICE}"


def get_embedding_client():
    """Return an AsyncOpenAI client configured for embedding requests."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=_OPENAI_API_KEY)


def get_llm_client():
    """Return an AsyncOpenAI client configured for LLM requests."""
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=_OPENAI_API_KEY)


_llm_client: Optional[object] = None
_embedding_client: Optional[object] = None


def get_cached_llm():
    """Return the cached async LLM client, initialising if needed."""
    global _llm_client
    if _llm_client is None:
        _llm_client = get_llm_client()
    return _llm_client


def get_cached_embedding():
    """Return the cached async embedding client, initialising if needed."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = get_embedding_client()
    return _embedding_client
