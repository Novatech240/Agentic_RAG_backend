"""
Agent settings endpoints — /api/v1/settings/agent

Lets admins edit the assistant's identity, system prompt, and scope from the
dashboard without a redeploy. Reading requires an active user; writing requires
the 'admin' role.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from . import settings_store
from .auth_deps import get_current_active_user, require_roles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

_admin = require_roles("admin")


class AgentSettingsUpdate(BaseModel):
    assistant_name: Optional[str] = Field(None, max_length=120)
    system_prompt: Optional[str] = Field(None, max_length=20000)
    scope_description: Optional[str] = Field(None, max_length=4000)
    enforce_scope: Optional[bool] = None
    out_of_scope_message: Optional[str] = Field(None, max_length=2000)


@router.get("/agent")
async def get_agent_settings(_user=Depends(get_current_active_user)):
    """Return the current (admin-editable) agent configuration."""
    config = await settings_store.get_config(force=True)
    return {"settings": asdict(config)}


@router.put("/agent")
async def update_agent_settings(body: AgentSettingsUpdate, _admin_user=Depends(_admin)):
    """Update the agent configuration (admin only)."""
    patch = body.model_dump(exclude_none=True)
    if not patch:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        config = await settings_store.update_config(**patch)
    except Exception as exc:
        logger.error("Failed to update agent settings: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to update settings")
    return {"status": "ok", "settings": asdict(config)}
