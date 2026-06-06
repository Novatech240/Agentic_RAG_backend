"""
Unit tests for the editable agent settings store (agent/settings_store.py).

No live Supabase needed: get_config falls back to defaults when the client is
uninitialised, and _row_to_config is pure.
"""

from __future__ import annotations

import pytest

from agent import settings_store
from agent.settings_store import (
    AgentConfig,
    DEFAULT_SCOPE_DESCRIPTION,
    DEFAULT_SYSTEM_PROMPT,
    _row_to_config,
)


class TestAgentConfigDefaults:
    def test_defaults_are_generic(self):
        c = AgentConfig()
        assert c.assistant_name == "Knowledge Assistant"
        assert c.system_prompt == DEFAULT_SYSTEM_PROMPT
        assert c.scope_description == DEFAULT_SCOPE_DESCRIPTION
        assert c.enforce_scope is True


class TestRowToConfig:
    def test_full_row_maps_through(self):
        c = _row_to_config(
            {
                "assistant_name": "CineMENA Assistant",
                "system_prompt": "You are the CineMENA helpdesk.",
                "scope_description": "CineMENA plans and pricing",
                "enforce_scope": False,
                "out_of_scope_message": "Sorry, off topic.",
            }
        )
        assert c.assistant_name == "CineMENA Assistant"
        assert c.enforce_scope is False
        assert c.scope_description == "CineMENA plans and pricing"

    def test_blank_fields_fall_back_to_defaults(self):
        c = _row_to_config({"assistant_name": "", "system_prompt": None})
        assert c.assistant_name == AgentConfig().assistant_name
        assert c.system_prompt == DEFAULT_SYSTEM_PROMPT


class TestGetConfigFallback:
    @pytest.mark.asyncio
    async def test_returns_default_without_client(self, monkeypatch):
        # No initialised Supabase client → defaults, never raises.
        monkeypatch.setattr(settings_store, "_client", lambda: None)
        settings_store._cache = None
        c = await settings_store.get_config(force=True)
        assert isinstance(c, AgentConfig)
        assert c.enforce_scope is True
