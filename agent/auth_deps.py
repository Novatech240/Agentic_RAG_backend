"""
FastAPI dependency for authenticated routes.
"""

from __future__ import annotations

from typing import Dict, List

import jwt
from fastapi import Depends, HTTPException, Request

from .auth_utils import decode_access_token
from . import db_utils


def _bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Missing or invalid Authorization header"
        )
    return auth[7:]


async def get_current_user(token: str = Depends(_bearer_token)) -> Dict:
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    result = (
        await db_utils._client.table("users")
        .select("*")
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=401, detail="User not found")

    user = rows[0]
    user["roles"] = payload.get("roles", [])
    return user


async def get_current_active_user(user: Dict = Depends(get_current_user)) -> Dict:
    if not user.get("is_active"):
        raise HTTPException(status_code=403, detail="Account is deactivated")
    return user


def require_roles(*roles: str):
    """Factory: returns a dependency that checks the user has at least one of the given roles."""

    async def _check(user: Dict = Depends(get_current_active_user)) -> Dict:
        user_roles: List[str] = user.get("roles", [])
        if not any(r in user_roles for r in roles):
            raise HTTPException(
                status_code=403,
                detail=f"Required role(s): {', '.join(roles)}",
            )
        return user

    return _check
