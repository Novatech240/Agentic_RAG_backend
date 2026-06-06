"""
User management + RBAC endpoints — /api/v1/users, /api/v1/roles, /api/v1/permissions
All write operations require the 'admin' role.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from .auth_deps import get_current_active_user, require_roles
from .auth_utils import hash_password


class DBClientProxy:
    def __getattr__(self, name):
        from . import db_utils

        return getattr(db_utils._client, name)


_client = DBClientProxy()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["users", "roles", "permissions"])

_admin = require_roles("admin")


class UserCreateRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (min 8 chars)")
    username: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True
    is_verified: bool = True
    role_names: List[str] = Field(default_factory=lambda: ["user"])


class UserUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    username: Optional[str] = None
    is_active: Optional[bool] = None
    role_names: Optional[List[str]] = None  # admin: replace user's roles


class RoleCreateRequest(BaseModel):
    name: str = Field(..., min_length=2)
    description: Optional[str] = None


class RoleUpdateRequest(BaseModel):
    description: Optional[str] = None
    permission_ids: Optional[List[str]] = None


class PermissionCreateRequest(BaseModel):
    name: str = Field(..., min_length=2)
    description: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None


# ── /api/v1/users ─────────────────────────────────────────────────────────────


@router.get("/users")
async def list_users(
    limit: int = 20,
    offset: int = 0,
    _admin_user=Depends(_admin),
):
    """List all users (admin only)."""
    result = await (
        _client.table("users")
        .select(
            "id, email, username, full_name, is_active, is_verified, created_at, last_login_at"
        )
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    users = result.data or []

    # Attach roles to each user
    for user in users:
        roles_res = await (
            _client.table("user_roles")
            .select("roles(name)")
            .eq("user_id", user["id"])
            .execute()
        )
        user["roles"] = [
            r["roles"]["name"] for r in (roles_res.data or []) if r.get("roles")
        ]

    return {"users": users, "total": len(users), "limit": limit, "offset": offset}


@router.post("/users", status_code=201)
async def create_user(req: UserCreateRequest, _admin_user=Depends(_admin)):
    """Create a new user directly (admin only)."""
    # Check if exists
    existing = (
        await _client.table("users")
        .select("id")
        .eq("email", req.email)
        .limit(1)
        .execute()
    )
    if existing.data:
        raise HTTPException(status_code=409, detail="Email already registered")

    user_id = str(uuid.uuid4())
    await (
        _client.table("users")
        .insert(
            {
                "id": user_id,
                "email": req.email,
                "username": req.username,
                "full_name": req.full_name,
                "hashed_password": hash_password(req.password),
                "is_active": req.is_active,
                "is_verified": req.is_verified,
            }
        )
        .execute()
    )

    # Assign roles
    for role_name in req.role_names:
        role_res = (
            await _client.table("roles")
            .select("id")
            .eq("name", role_name)
            .limit(1)
            .execute()
        )
        if role_res.data:
            await (
                _client.table("user_roles")
                .insert(
                    {
                        "user_id": user_id,
                        "role_id": role_res.data[0]["id"],
                    }
                )
                .execute()
            )

    return {"id": user_id, "email": req.email, "message": "User created successfully"}


@router.get("/users/{user_id}")
async def get_user(user_id: str, current_user=Depends(get_current_active_user)):
    """Get a user by ID. Users can fetch their own profile; admins can fetch any."""
    if current_user["id"] != user_id and "admin" not in current_user.get("roles", []):
        raise HTTPException(status_code=403, detail="Access denied")

    result = await (
        _client.table("users")
        .select(
            "id, email, username, full_name, is_active, is_verified, created_at, last_login_at"
        )
        .eq("id", user_id)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")

    user = rows[0]
    roles_res = (
        await _client.table("user_roles")
        .select("roles(name)")
        .eq("user_id", user_id)
        .execute()
    )
    user["roles"] = [
        r["roles"]["name"] for r in (roles_res.data or []) if r.get("roles")
    ]
    return user


@router.put("/users/{user_id}")
async def update_user(
    user_id: str, req: UserUpdateRequest, _admin_user=Depends(_admin)
):
    """Update a user's profile or roles (admin only)."""
    updates: dict = {}
    if req.full_name is not None:
        updates["full_name"] = req.full_name
    if req.username is not None:
        updates["username"] = req.username
    if req.is_active is not None:
        updates["is_active"] = req.is_active

    if updates:
        await _client.table("users").update(updates).eq("id", user_id).execute()

    if req.role_names is not None:
        # Replace roles
        await _client.table("user_roles").delete().eq("user_id", user_id).execute()
        for role_name in req.role_names:
            role_res = (
                await _client.table("roles")
                .select("id")
                .eq("name", role_name)
                .limit(1)
                .execute()
            )
            if role_res.data:
                await (
                    _client.table("user_roles")
                    .insert(
                        {
                            "user_id": user_id,
                            "role_id": role_res.data[0]["id"],
                        }
                    )
                    .execute()
                )

    return {"message": "User updated successfully"}


@router.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str, _admin_user=Depends(_admin)):
    """Delete a user account (admin only)."""
    await _client.table("users").delete().eq("id", user_id).execute()
    return None


# ── /api/v1/roles ─────────────────────────────────────────────────────────────


@router.get("/roles")
async def list_roles(_user=Depends(get_current_active_user)):
    """List all roles with their permissions."""
    result = await _client.table("roles").select("*").order("name").execute()
    roles = result.data or []

    for role in roles:
        perms_res = await (
            _client.table("role_permissions")
            .select("permissions(id, name, description, resource, action)")
            .eq("role_id", role["id"])
            .execute()
        )
        role["permissions"] = [
            r["permissions"] for r in (perms_res.data or []) if r.get("permissions")
        ]

    return {"roles": roles}


@router.post("/roles", status_code=201)
async def create_role(req: RoleCreateRequest, _admin_user=Depends(_admin)):
    """Create a new role (admin only)."""
    existing = (
        await _client.table("roles")
        .select("id")
        .eq("name", req.name)
        .limit(1)
        .execute()
    )
    if existing.data:
        raise HTTPException(status_code=409, detail=f"Role '{req.name}' already exists")

    role_id = str(uuid.uuid4())
    await (
        _client.table("roles")
        .insert(
            {
                "id": role_id,
                "name": req.name,
                "description": req.description,
            }
        )
        .execute()
    )
    return {"id": role_id, "name": req.name, "message": "Role created"}


@router.put("/roles/{role_id}")
async def update_role(
    role_id: str, req: RoleUpdateRequest, _admin_user=Depends(_admin)
):
    """Update role description and/or replace its permissions (admin only)."""
    if req.description is not None:
        await (
            _client.table("roles")
            .update({"description": req.description})
            .eq("id", role_id)
            .execute()
        )

    if req.permission_ids is not None:
        await (
            _client.table("role_permissions").delete().eq("role_id", role_id).execute()
        )
        for perm_id in req.permission_ids:
            await (
                _client.table("role_permissions")
                .insert(
                    {
                        "role_id": role_id,
                        "permission_id": perm_id,
                    }
                )
                .execute()
            )

    return {"message": "Role updated"}


@router.delete("/roles/{role_id}", status_code=204)
async def delete_role(role_id: str, _admin_user=Depends(_admin)):
    """Delete a role (admin only). Protected names 'admin'/'user' cannot be deleted."""
    result = (
        await _client.table("roles").select("name").eq("id", role_id).limit(1).execute()
    )
    if result.data and result.data[0]["name"] in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Cannot delete a built-in role")
    await _client.table("roles").delete().eq("id", role_id).execute()
    return None


# ── /api/v1/permissions ───────────────────────────────────────────────────────


@router.get("/permissions")
async def list_permissions(_user=Depends(get_current_active_user)):
    """List all permissions."""
    result = await _client.table("permissions").select("*").order("name").execute()
    return {"permissions": result.data or []}


@router.post("/permissions", status_code=201)
async def create_permission(req: PermissionCreateRequest, _admin_user=Depends(_admin)):
    """Create a new permission (admin only)."""
    existing = (
        await _client.table("permissions")
        .select("id")
        .eq("name", req.name)
        .limit(1)
        .execute()
    )
    if existing.data:
        raise HTTPException(
            status_code=409, detail=f"Permission '{req.name}' already exists"
        )

    perm_id = str(uuid.uuid4())
    await (
        _client.table("permissions")
        .insert(
            {
                "id": perm_id,
                "name": req.name,
                "description": req.description,
                "resource": req.resource,
                "action": req.action,
            }
        )
        .execute()
    )
    return {"id": perm_id, "name": req.name, "message": "Permission created"}


@router.delete("/permissions/{permission_id}", status_code=204)
async def delete_permission(permission_id: str, _admin_user=Depends(_admin)):
    """Delete a permission (admin only)."""
    await _client.table("permissions").delete().eq("id", permission_id).execute()


def _get_s3_client():
    import boto3
    import os

    supabase_url = os.getenv("SUPABASE_URL")
    endpoint_url = None
    if supabase_url:
        endpoint_url = f"{supabase_url.rstrip('/')}/storage/v1/s3"

    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-2"),
        endpoint_url=endpoint_url,
    )


@router.post("/users/{user_id}/avatar")
async def upload_user_avatar(
    user_id: str,
    file: UploadFile = File(...),
    current_user=Depends(get_current_active_user),
):
    """Upload user avatar (saved locally + uploaded to S3).

    A user may only set their own avatar; admins may set anyone's.
    """
    import os
    from pathlib import Path

    if current_user["id"] != user_id and "admin" not in current_user.get("roles", []):
        raise HTTPException(status_code=403, detail="Access denied")

    # Reject non-image uploads (defense-in-depth: avatars are images only).
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Avatar must be an image file")

    try:
        # Verify user exists
        user_res = (
            await _client.table("users")
            .select("id")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        if not user_res.data:
            raise HTTPException(status_code=404, detail="User not found")

        contents = await file.read()

        # Save locally (robust developer fallback)
        local_dir = Path("data/avatars")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_path = local_dir / f"{user_id}.jpg"
        with open(local_path, "wb") as fh:
            fh.write(contents)

        # Upload to S3
        try:
            s3 = _get_s3_client()
            bucket = os.getenv("S3_PRIVATE_BUCKET", "uchenab")
            s3_key = f"avatars/{user_id}.jpg"
            s3.put_object(
                Bucket=bucket,
                Key=s3_key,
                Body=contents,
                ContentType=file.content_type or "image/jpeg",
            )
            logger.info("Uploaded avatar for %s to S3 bucket %s", user_id, bucket)
        except Exception as s3_err:
            logger.warning("S3 avatar upload failed (using local fallback): %s", s3_err)

        # Update user's avatar_url in the Supabase database
        try:
            avatar_url = f"/api/v1/users/{user_id}/avatar"
            await (
                _client.table("users")
                .update({"avatar_url": avatar_url})
                .eq("id", user_id)
                .execute()
            )
            logger.info(
                "Successfully updated avatar_url in Supabase users table for %s",
                user_id,
            )
        except Exception as db_err:
            logger.warning("Failed to update users table in Supabase: %s", db_err)

        # Invalidate avatar cache in Redis
        try:
            from .redis_utils import get_redis_client

            redis_client = get_redis_client()
            if redis_client:
                cache_key = f"uchenab:avatar:{user_id}"
                redis_client.delete(cache_key)
                logger.info("Invalidated Redis avatar cache for user %s", user_id)
        except Exception as cache_err:
            logger.warning("Failed to invalidate Redis avatar cache: %s", cache_err)

        return {
            "message": "Avatar uploaded successfully",
            "avatar_url": f"/api/v1/users/{user_id}/avatar",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Avatar upload failed: %s", e)
        raise HTTPException(status_code=500, detail="Avatar upload failed")


@router.get("/users/{user_id}/avatar")
async def get_user_avatar(user_id: str):
    """Retrieve user avatar (Redis first, local fallback, S3 fallback, otherwise 404)."""
    import os
    from pathlib import Path
    from fastapi.responses import Response
    from .redis_utils import get_redis_client

    cache_key = f"uchenab:avatar:{user_id}"

    # 0. Check Redis cache first
    try:
        redis_client = get_redis_client()
        if redis_client:
            cached_avatar = redis_client.get(cache_key)
            if cached_avatar:
                logger.info("Serving avatar for %s from Redis cache", user_id)
                return Response(content=cached_avatar, media_type="image/jpeg")
    except Exception as cache_err:
        logger.warning("Failed to read from Redis avatar cache: %s", cache_err)

    avatar_content = None

    # 1. Local check
    local_path = Path("data/avatars") / f"{user_id}.jpg"
    if local_path.exists():
        try:
            with open(local_path, "rb") as fh:
                avatar_content = fh.read()
        except Exception as e:
            logger.error("Failed to read local avatar file: %s", e)

    # 2. S3 check
    if not avatar_content:
        try:
            s3 = _get_s3_client()
            bucket = os.getenv("S3_PRIVATE_BUCKET", "uchenab")
            s3_key = f"avatars/{user_id}.jpg"
            try:
                response = s3.get_object(Bucket=bucket, Key=s3_key)
                avatar_content = response["Body"].read()
            except s3.exceptions.NoSuchKey:
                pass
            except Exception as s3_err:
                logger.warning("S3 avatar download failed: %s", s3_err)
        except Exception as e:
            logger.error("Failed to configure S3 client: %s", e)

    # 3. Cache in Redis and return
    if avatar_content:
        try:
            redis_client = get_redis_client()
            if redis_client:
                # Cache for 24 hours (86400 seconds)
                redis_client.setex(cache_key, 86400, avatar_content)
                logger.info("Cached avatar for %s in Redis", user_id)
        except Exception as cache_err:
            logger.warning("Failed to cache avatar in Redis: %s", cache_err)

        return Response(content=avatar_content, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Avatar not found")
