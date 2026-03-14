"""Tests for the in-memory ChatStore."""

import threading
import time

import pytest

from uniinfer.api.chat_store import ChatMessage, ChatSession, ChatStore


# ---------------------------------------------------------------------------
# ChatMessage dataclass
# ---------------------------------------------------------------------------


class TestChatMessage:
    def test_defaults(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tokens == 0
        assert msg.tokens_per_second == 0.0
        assert isinstance(msg.timestamp, float)

    def test_custom_fields(self):
        msg = ChatMessage(
            role="assistant",
            content="hi",
            timestamp=1000.0,
            tokens=42,
            tokens_per_second=12.5,
        )
        assert msg.tokens == 42
        assert msg.tokens_per_second == 12.5
        assert msg.timestamp == 1000.0


# ---------------------------------------------------------------------------
# ChatSession dataclass
# ---------------------------------------------------------------------------


class TestChatSession:
    def test_message_count(self):
        s = ChatSession(session_id="s1", model="m", source="api")
        assert s.message_count == 0
        s.messages.append(ChatMessage(role="user", content="a"))
        assert s.message_count == 1

    def test_last_message_preview_empty(self):
        s = ChatSession(session_id="s1", model="m", source="api")
        assert s.last_message_preview == ""

    def test_last_message_preview_short(self):
        s = ChatSession(session_id="s1", model="m", source="api")
        s.messages.append(ChatMessage(role="user", content="short text"))
        assert s.last_message_preview == "short text"

    def test_last_message_preview_truncation(self):
        s = ChatSession(session_id="s1", model="m", source="api")
        long_text = "x" * 120
        s.messages.append(ChatMessage(role="user", content=long_text))
        preview = s.last_message_preview
        assert len(preview) == 83  # 80 chars + "..."
        assert preview.endswith("...")

    def test_to_summary(self):
        s = ChatSession(session_id="s1", model="m", source="cli", created_at=500.0)
        s.messages.append(ChatMessage(role="user", content="hello"))
        summary = s.to_summary()
        assert summary["session_id"] == "s1"
        assert summary["model"] == "m"
        assert summary["source"] == "cli"
        assert summary["created_at"] == 500.0
        assert summary["message_count"] == 1
        assert summary["last_message_preview"] == "hello"


# ---------------------------------------------------------------------------
# ChatStore — session management
# ---------------------------------------------------------------------------


class TestChatStoreSessionManagement:
    def test_create_session_returns_id(self):
        store = ChatStore()
        sid = store.create_session("model-a", "api")
        assert sid.startswith("sess-")
        assert store.total_sessions == 1

    def test_create_session_custom_id(self):
        store = ChatStore()
        sid = store.create_session("model-a", "cli", session_id="my-id")
        assert sid == "my-id"

    def test_session_eviction(self):
        store = ChatStore(max_sessions=3)
        ids = [store.create_session("m", "api") for _ in range(5)]
        assert store.total_sessions == 3
        # First two should be evicted
        assert store.get_session(ids[0]) is None
        assert store.get_session(ids[1]) is None
        # Last three should exist
        assert store.get_session(ids[2]) is not None
        assert store.get_session(ids[3]) is not None
        assert store.get_session(ids[4]) is not None

    def test_get_or_create_existing(self):
        store = ChatStore()
        sid = store.create_session("m", "api")
        returned = store.get_or_create_session(sid, "m", "api")
        assert returned == sid
        assert store.total_sessions == 1

    def test_get_or_create_new(self):
        store = ChatStore()
        sid = store.get_or_create_session(None, "m", "cli")
        assert store.total_sessions == 1
        assert store.get_session(sid) is not None

    def test_get_or_create_missing_session(self):
        store = ChatStore()
        sid = store.get_or_create_session("nonexistent", "m", "api")
        assert sid == "nonexistent"
        assert store.get_session(sid) is not None

    def test_list_sessions_order(self):
        store = ChatStore()
        store.create_session("m", "api", session_id="a")
        store.create_session("m", "cli", session_id="b")
        store.create_session("m", "api", session_id="c")
        sessions = store.list_sessions()
        # Most recent first
        assert [s["session_id"] for s in sessions] == ["c", "b", "a"]

    def test_get_session_not_found(self):
        store = ChatStore()
        assert store.get_session("nope") is None


# ---------------------------------------------------------------------------
# ChatStore — message management
# ---------------------------------------------------------------------------


class TestChatStoreMessages:
    def test_add_message(self):
        store = ChatStore()
        sid = store.create_session("m", "api")
        store.add_message(sid, ChatMessage(role="user", content="hi"))
        session = store.get_session(sid)
        assert session is not None
        assert session.message_count == 1
        assert session.messages[0].content == "hi"

    def test_add_message_unknown_session(self):
        store = ChatStore()
        # Should not raise
        store.add_message("nope", ChatMessage(role="user", content="hi"))
        assert store.total_messages == 0

    def test_message_eviction(self):
        store = ChatStore(max_messages_per_session=5)
        sid = store.create_session("m", "api")
        for i in range(10):
            store.add_message(sid, ChatMessage(role="user", content=f"msg-{i}"))
        session = store.get_session(sid)
        assert session is not None
        assert session.message_count == 5
        # Oldest messages evicted, newest remain
        assert session.messages[0].content == "msg-5"
        assert session.messages[-1].content == "msg-9"

    def test_add_message_moves_session_to_end(self):
        store = ChatStore()
        store.create_session("m", "api", session_id="old")
        store.create_session("m", "api", session_id="new")
        # "old" was created first, should be last in list
        sessions = store.list_sessions()
        assert sessions[0]["session_id"] == "new"
        # Add message to "old" — should move it to most recent
        store.add_message("old", ChatMessage(role="user", content="bump"))
        sessions = store.list_sessions()
        assert sessions[0]["session_id"] == "old"


# ---------------------------------------------------------------------------
# ChatStore — recent messages
# ---------------------------------------------------------------------------


class TestChatStoreRecentMessages:
    def test_recent_messages_empty(self):
        store = ChatStore()
        assert store.get_recent_messages() == []

    def test_recent_messages_sorted(self):
        store = ChatStore()
        sid = store.create_session("m", "api")
        store.add_message(sid, ChatMessage(role="user", content="first", timestamp=100.0))
        store.add_message(sid, ChatMessage(role="assistant", content="second", timestamp=200.0))
        store.add_message(sid, ChatMessage(role="user", content="third", timestamp=300.0))
        msgs = store.get_recent_messages()
        # Newest first
        assert msgs[0]["content"] == "third"
        assert msgs[-1]["content"] == "first"

    def test_recent_messages_limit(self):
        store = ChatStore()
        sid = store.create_session("m", "api")
        for i in range(20):
            store.add_message(
                sid,
                ChatMessage(role="user", content=f"m{i}", timestamp=float(i)),
            )
        msgs = store.get_recent_messages(limit=5)
        assert len(msgs) == 5
        assert msgs[0]["content"] == "m19"

    def test_recent_messages_cross_session(self):
        store = ChatStore()
        s1 = store.create_session("m", "api")
        s2 = store.create_session("m", "cli")
        store.add_message(s1, ChatMessage(role="user", content="a", timestamp=100.0))
        store.add_message(s2, ChatMessage(role="user", content="b", timestamp=200.0))
        store.add_message(s1, ChatMessage(role="user", content="c", timestamp=300.0))
        msgs = store.get_recent_messages()
        assert msgs[0]["content"] == "c"
        assert msgs[0]["session_id"] == s1
        assert msgs[1]["content"] == "b"
        assert msgs[1]["session_id"] == s2

    def test_recent_messages_include_session_info(self):
        store = ChatStore()
        sid = store.create_session("test-model", "cli")
        store.add_message(
            sid,
            ChatMessage(role="assistant", content="hi", tokens=10, tokens_per_second=5.0),
        )
        msgs = store.get_recent_messages()
        assert len(msgs) == 1
        assert msgs[0]["model"] == "test-model"
        assert msgs[0]["source"] == "cli"
        assert msgs[0]["tokens"] == 10
        assert msgs[0]["tokens_per_second"] == 5.0


# ---------------------------------------------------------------------------
# ChatStore — summary and properties
# ---------------------------------------------------------------------------


class TestChatStoreSummary:
    def test_total_sessions(self):
        store = ChatStore()
        assert store.total_sessions == 0
        store.create_session("m", "api")
        store.create_session("m", "cli")
        assert store.total_sessions == 2

    def test_total_messages(self):
        store = ChatStore()
        sid = store.create_session("m", "api")
        assert store.total_messages == 0
        store.add_message(sid, ChatMessage(role="user", content="a"))
        store.add_message(sid, ChatMessage(role="assistant", content="b"))
        assert store.total_messages == 2

    def test_summary_empty(self):
        store = ChatStore()
        s = store.summary()
        assert s["active_sessions"] == 0
        assert s["total_messages"] == 0
        assert s["last_message_preview"] == ""

    def test_summary_with_data(self):
        store = ChatStore()
        sid = store.create_session("m", "api")
        store.add_message(sid, ChatMessage(role="user", content="hello world"))
        s = store.summary()
        assert s["active_sessions"] == 1
        assert s["total_messages"] == 1
        assert s["last_message_preview"] == "hello world"

    def test_summary_truncates_preview(self):
        store = ChatStore()
        sid = store.create_session("m", "api")
        long_text = "z" * 100
        store.add_message(sid, ChatMessage(role="user", content=long_text))
        s = store.summary()
        assert len(s["last_message_preview"]) == 63  # 60 + "..."
        assert s["last_message_preview"].endswith("...")


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestChatStoreThreadSafety:
    def test_concurrent_session_creation(self):
        store = ChatStore(max_sessions=100)
        errors: list[Exception] = []

        def create_sessions(n: int):
            try:
                for _ in range(n):
                    store.create_session("m", "api")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_sessions, args=(20,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert store.total_sessions == 100

    def test_concurrent_message_addition(self):
        store = ChatStore(max_messages_per_session=1000)
        sid = store.create_session("m", "api")
        errors: list[Exception] = []

        def add_messages(n: int):
            try:
                for i in range(n):
                    store.add_message(
                        sid, ChatMessage(role="user", content=f"msg-{i}")
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_messages, args=(50,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        session = store.get_session(sid)
        assert session is not None
        assert session.message_count == 250
