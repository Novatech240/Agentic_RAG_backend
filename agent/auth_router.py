"""
Authentication endpoints — /api/v1/auth/*
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .auth_deps import get_current_active_user
from .auth_utils import (
    create_access_token,
    create_refresh_token,
    hash_password,
    hash_token,
    make_otp,
    refresh_token_expiry,
    reset_token_expiry,
    verification_token_expiry,
    verify_password,
)


class DBClientProxy:
    def __getattr__(self, name):
        from . import db_utils

        return getattr(db_utils._client, name)


_client = DBClientProxy()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ── Request / Response models ─────────────────────────────────────────────────


class RegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (min 8 chars)")
    username: Optional[str] = None
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class VerifyEmailRequest(BaseModel):
    token: str


class ResendVerificationRequest(BaseModel):
    email: str


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _get_user_roles(user_id: str) -> list[str]:
    result = await (
        _client.table("user_roles")
        .select("roles(name)")
        .eq("user_id", user_id)
        .execute()
    )
    return [r["roles"]["name"] for r in (result.data or []) if r.get("roles")]


async def _build_token_response(user: dict) -> TokenResponse:
    roles = await _get_user_roles(user["id"])
    access_token = create_access_token(
        user_id=user["id"], email=user["email"], roles=roles
    )
    raw_refresh, hashed_refresh = create_refresh_token()
    await (
        _client.table("refresh_tokens")
        .insert(
            {
                "user_id": user["id"],
                "token_hash": hashed_refresh,
                "expires_at": refresh_token_expiry().isoformat(),
            }
        )
        .execute()
    )

    from .auth_utils import ACCESS_TOKEN_EXPIRE_MINUTES

    return TokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── POST /register ────────────────────────────────────────────────────────────


@router.post("/register", status_code=201)
async def register(req: RegisterRequest):
    """Register a new user account. Sends an email verification token."""
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
                "is_active": True,
                "is_verified": False,
            }
        )
        .execute()
    )

    # Assign default 'user' role
    role_res = (
        await _client.table("roles").select("id").eq("name", "user").limit(1).execute()
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

    # Create email verification token
    token = make_otp()
    await (
        _client.table("email_verifications")
        .insert(
            {
                "user_id": user_id,
                "token": token,
                "expires_at": verification_token_expiry().isoformat(),
            }
        )
        .execute()
    )

    logger.info("User registered: %s", req.email)
    return {
        "message": "Registration successful. Please verify your email.",
        "user_id": user_id,
        "verification_token": token,  # In production: send via email, don't return here
    }


# ── POST /login ───────────────────────────────────────────────────────────────


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Login with email and password. Returns access + refresh tokens."""
    result = (
        await _client.table("users")
        .select("*")
        .eq("email", req.email)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = rows[0]
    if not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account deactivated")
    if not user["is_verified"]:
        raise HTTPException(
            status_code=403, detail="Email not verified. Check your inbox."
        )

    await (
        _client.table("users")
        .update({"last_login_at": datetime.now(timezone.utc).isoformat()})
        .eq("id", user["id"])
        .execute()
    )

    return await _build_token_response(user)


# ── POST /logout ──────────────────────────────────────────────────────────────


@router.post("/logout")
async def logout(req: RefreshRequest, _user=Depends(get_current_active_user)):
    """Revoke the supplied refresh token."""
    token_hash = hash_token(req.refresh_token)
    await (
        _client.table("refresh_tokens")
        .update({"revoked": True})
        .eq("token_hash", token_hash)
        .execute()
    )
    return {"message": "Logged out successfully"}


# ── POST /refresh-token ───────────────────────────────────────────────────────


@router.post("/refresh-token", response_model=TokenResponse)
async def refresh_token(req: RefreshRequest):
    """Exchange a valid refresh token for a new access + refresh token pair."""
    token_hash = hash_token(req.refresh_token)
    result = (
        await _client.table("refresh_tokens")
        .select("*")
        .eq("token_hash", token_hash)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    stored = rows[0]
    if stored["revoked"]:
        raise HTTPException(status_code=401, detail="Refresh token revoked")

    expires_at = datetime.fromisoformat(stored["expires_at"].replace("Z", "+00:00"))
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expired")

    # Rotate: revoke old, issue new
    await (
        _client.table("refresh_tokens")
        .update({"revoked": True})
        .eq("id", stored["id"])
        .execute()
    )

    user_result = (
        await _client.table("users")
        .select("*")
        .eq("id", stored["user_id"])
        .limit(1)
        .execute()
    )
    user_rows = user_result.data or []
    if not user_rows or not user_rows[0]["is_active"]:
        raise HTTPException(status_code=401, detail="User not found or deactivated")

    return await _build_token_response(user_rows[0])


# ── POST /forgot-password ─────────────────────────────────────────────────────


@router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    """Request a password reset token. Always returns 200 to prevent user enumeration."""
    result = (
        await _client.table("users")
        .select("id")
        .eq("email", req.email)
        .limit(1)
        .execute()
    )
    if result.data:
        user_id = result.data[0]["id"]
        token = make_otp()
        await (
            _client.table("password_resets")
            .insert(
                {
                    "user_id": user_id,
                    "token": token,
                    "expires_at": reset_token_expiry().isoformat(),
                }
            )
            .execute()
        )
        logger.info("Password reset requested for %s", req.email)
        # In production: send token via email. Here we return it for testing.
        return {
            "message": "If that email exists, a reset link has been sent.",
            "reset_token": token,
        }

    return {"message": "If that email exists, a reset link has been sent."}


# ── POST /reset-password ──────────────────────────────────────────────────────


@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest):
    """Reset password using a valid reset token."""
    result = (
        await _client.table("password_resets")
        .select("*")
        .eq("token", req.token)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    stored = rows[0]
    if stored["used"]:
        raise HTTPException(status_code=400, detail="Reset token already used")

    expires_at = datetime.fromisoformat(stored["expires_at"].replace("Z", "+00:00"))
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Reset token expired")

    new_hash = hash_password(req.new_password)
    await (
        _client.table("users")
        .update({"hashed_password": new_hash})
        .eq("id", stored["user_id"])
        .execute()
    )
    await (
        _client.table("password_resets")
        .update({"used": True})
        .eq("id", stored["id"])
        .execute()
    )
    # Revoke all refresh tokens for security
    await (
        _client.table("refresh_tokens")
        .update({"revoked": True})
        .eq("user_id", stored["user_id"])
        .execute()
    )

    return {"message": "Password reset successfully. Please log in again."}


# ── POST /change-password ─────────────────────────────────────────────────────


@router.post("/change-password")
async def change_password(
    req: ChangePasswordRequest, user=Depends(get_current_active_user)
):
    """Change password for the currently authenticated user."""
    if not verify_password(req.current_password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    new_hash = hash_password(req.new_password)
    await (
        _client.table("users")
        .update({"hashed_password": new_hash})
        .eq("id", user["id"])
        .execute()
    )
    await (
        _client.table("refresh_tokens")
        .update({"revoked": True})
        .eq("user_id", user["id"])
        .execute()
    )

    return {"message": "Password changed successfully. Please log in again."}


# ── POST /verify-email ────────────────────────────────────────────────────────


@router.post("/verify-email")
async def verify_email(req: VerifyEmailRequest):
    """Verify email address using the token sent on registration."""
    result = (
        await _client.table("email_verifications")
        .select("*")
        .eq("token", req.token)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        raise HTTPException(status_code=400, detail="Invalid verification token")

    stored = rows[0]
    if stored["used"]:
        raise HTTPException(status_code=400, detail="Token already used")

    expires_at = datetime.fromisoformat(stored["expires_at"].replace("Z", "+00:00"))
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Verification token expired")

    await (
        _client.table("users")
        .update({"is_verified": True})
        .eq("id", stored["user_id"])
        .execute()
    )
    await (
        _client.table("email_verifications")
        .update({"used": True})
        .eq("id", stored["id"])
        .execute()
    )

    return {"message": "Email verified successfully. You can now log in."}


# ── POST /resend-verification ─────────────────────────────────────────────────


@router.post("/resend-verification")
async def resend_verification(req: ResendVerificationRequest):
    """Re-issue an email verification token."""
    result = (
        await _client.table("users")
        .select("id, is_verified")
        .eq("email", req.email)
        .limit(1)
        .execute()
    )
    rows = result.data or []
    if not rows:
        return {
            "message": "If that email exists and is unverified, a new token has been sent."
        }

    user = rows[0]
    if user["is_verified"]:
        return {"message": "Email is already verified."}

    token = make_otp()
    await (
        _client.table("email_verifications")
        .insert(
            {
                "user_id": user["id"],
                "token": token,
                "expires_at": verification_token_expiry().isoformat(),
            }
        )
        .execute()
    )

    return {
        "message": "If that email exists and is unverified, a new token has been sent.",
        "verification_token": token,  # In production: send via email only
    }


# ── GET /me ───────────────────────────────────────────────────────────────────


@router.get("/me")
async def me(user=Depends(get_current_active_user)):
    """Return the currently authenticated user's profile."""
    return {
        "id": user["id"],
        "email": user["email"],
        "username": user.get("username"),
        "full_name": user.get("full_name"),
        "is_verified": user["is_verified"],
        "roles": user.get("roles", []),
        "created_at": user["created_at"],
        "last_login_at": user.get("last_login_at"),
    }
