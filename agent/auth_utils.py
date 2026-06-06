"""
JWT + password utilities for the auth system.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import bcrypt
import jwt

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULT_JWT_SECRET = "change-me-in-production"
JWT_SECRET = os.getenv("JWT_SECRET_KEY", _DEFAULT_JWT_SECRET)
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Refuse to run with the insecure default secret outside development — a leaked
# default would let anyone forge admin tokens.
if JWT_SECRET == _DEFAULT_JWT_SECRET or not JWT_SECRET:
    if os.getenv("APP_ENV", "development").lower() == "development":
        logger.warning(
            "JWT_SECRET_KEY is unset/default — set a strong secret before production."
        )
    else:
        raise RuntimeError(
            "JWT_SECRET_KEY must be set to a strong, unique value in non-development "
            "environments."
        )


# ── Password helpers ──────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ── JWT helpers ───────────────────────────────────────────────────────────────


def create_access_token(user_id: str, email: str, roles: list[str]) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "email": email,
        "roles": roles,
        "type": "access",
        "exp": expire,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token() -> tuple[str, str]:
    """Return (raw_token, hashed_token). Store the hash; send the raw."""
    raw = secrets.token_urlsafe(48)
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    return raw, hashed


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and verify an access token. Raises jwt.PyJWTError on failure."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def make_otp() -> str:
    """Generate a short-lived one-time token (email verification / password reset)."""
    return secrets.token_urlsafe(32)


def refresh_token_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)


def verification_token_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=24)


def reset_token_expiry() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=1)
