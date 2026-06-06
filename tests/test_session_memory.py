"""Tests for SessionMemoryManager — isolation, windowing, fallback.

conftest forces SESSION_MEMORY_BACKEND=memory, so these exercise the
in-process backend (identical public behaviour to the Redis backend).
"""

from agent.session_memory import SessionMemoryManager


class TestIsolation:
    def test_two_sessions_never_mix(self):
        mgr = SessionMemoryManager()
        a = mgr.get_or_create("chat:111")
        b = mgr.get_or_create("chat:222")
        mgr.add_turn(a, "I am Ali", "Salam Ali")
        mgr.add_turn(b, "I am Sara", "Salam Sara")

        ctx_a = mgr.get_context_string(a)
        ctx_b = mgr.get_context_string(b)

        assert "Ali" in ctx_a and "Sara" not in ctx_a
        assert "Sara" in ctx_b and "Ali" not in ctx_b

    def test_unknown_session_starts_empty(self):
        mgr = SessionMemoryManager()
        assert mgr.get_context_string("chat:999") == ""

    def test_get_or_create_registers_session(self):
        mgr = SessionMemoryManager()
        sid = mgr.get_or_create("sess-x")
        assert sid == "sess-x"
        assert mgr.session_exists("sess-x")

    def test_get_or_create_mints_uuid_when_none(self):
        mgr = SessionMemoryManager()
        sid = mgr.get_or_create(None)
        assert len(sid) == 36  # uuid4


class TestWindow:
    def test_sliding_window_trims_old_turns(self):
        mgr = SessionMemoryManager(window_size=2)  # keep last 2 turns = 4 messages
        sid = mgr.get_or_create("w")
        for i in range(5):
            mgr.add_turn(sid, f"q{i}", f"a{i}")
        msgs = mgr.get_messages(sid)
        assert len(msgs) == 4  # only last 2 turns retained
        assert "q4" in msgs[-2].content  # newest turn present
        assert all("q0" not in m.content for m in msgs)  # oldest dropped


class TestSessionInfo:
    def test_counts_turns(self):
        mgr = SessionMemoryManager()
        sid = mgr.get_or_create("i")
        mgr.add_turn(sid, "hi", "hello")
        mgr.add_turn(sid, "bye", "ok")
        info = mgr.session_info(sid)
        assert info["turn_count"] == 2
        assert info["message_count"] == 4

    def test_clear_session(self):
        mgr = SessionMemoryManager()
        sid = mgr.get_or_create("c")
        mgr.add_turn(sid, "x", "y")
        mgr.clear_session(sid)
        assert mgr.get_context_string(sid) == ""
