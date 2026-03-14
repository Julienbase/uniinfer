"""In-memory chat session and message store for dashboard tracking."""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    """A single message in a chat session."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0
    tokens_per_second: float = 0.0


@dataclass
class ChatSession:
    """A chat session containing messages."""

    session_id: str
    model: str
    source: str  # "cli", "api", "dashboard"
    created_at: float = field(default_factory=time.time)
    messages: list[ChatMessage] = field(default_factory=list)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_message_preview(self) -> str:
        if not self.messages:
            return ""
        last = self.messages[-1]
        preview = last.content[:80]
        if len(last.content) > 80:
            preview += "..."
        return preview

    def to_summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "model": self.model,
            "source": self.source,
            "created_at": self.created_at,
            "message_count": self.message_count,
            "last_message_preview": self.last_message_preview,
        }


class ChatStore:
    """Thread-safe in-memory store for chat sessions and messages.

    Bounded to prevent unbounded memory growth:
    - Max sessions: oldest evicted when limit reached
    - Max messages per session: oldest evicted when limit reached
    """

    def __init__(
        self,
        max_sessions: int = 50,
        max_messages_per_session: int = 200,
    ) -> None:
        self._max_sessions = max_sessions
        self._max_messages = max_messages_per_session
        self._sessions: OrderedDict[str, ChatSession] = OrderedDict()
        self._lock = threading.Lock()

    def create_session(
        self,
        model: str,
        source: str = "api",
        session_id: str | None = None,
    ) -> str:
        """Create a new chat session.

        Args:
            model: Model identifier.
            source: Origin of the session (cli, api, dashboard).
            session_id: Optional pre-generated session ID.

        Returns:
            The session ID.
        """
        sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"

        with self._lock:
            # Evict oldest if at capacity
            while len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)

            self._sessions[sid] = ChatSession(
                session_id=sid,
                model=model,
                source=source,
            )

        return sid

    def get_or_create_session(
        self,
        session_id: str | None,
        model: str,
        source: str = "api",
    ) -> str:
        """Get an existing session or create a new one.

        Args:
            session_id: Session ID to look up, or None to create.
            model: Model identifier (used if creating).
            source: Origin (used if creating).

        Returns:
            The session ID.
        """
        if session_id:
            with self._lock:
                if session_id in self._sessions:
                    return session_id
        return self.create_session(model, source, session_id)

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        """Add a message to a session.

        Args:
            session_id: Target session.
            message: The message to add.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return

            session.messages.append(message)

            # Evict oldest messages if over limit
            while len(session.messages) > self._max_messages:
                session.messages.pop(0)

            # Move session to end (most recent)
            self._sessions.move_to_end(session_id)

    def get_session(self, session_id: str) -> ChatSession | None:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        """List all sessions as summary dicts, most recent first."""
        with self._lock:
            return [
                s.to_summary()
                for s in reversed(self._sessions.values())
            ]

    def get_recent_messages(self, limit: int = 50) -> list[dict]:
        """Get the most recent messages across all sessions.

        Args:
            limit: Maximum number of messages to return.

        Returns:
            List of message dicts with session info, newest first.
        """
        all_msgs: list[dict] = []

        with self._lock:
            for session in self._sessions.values():
                for msg in session.messages:
                    all_msgs.append({
                        "session_id": session.session_id,
                        "model": session.model,
                        "source": session.source,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "tokens": msg.tokens,
                        "tokens_per_second": msg.tokens_per_second,
                    })

        # Sort by timestamp descending, take limit
        all_msgs.sort(key=lambda m: m["timestamp"], reverse=True)
        return all_msgs[:limit]

    @property
    def total_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    @property
    def total_messages(self) -> int:
        with self._lock:
            return sum(len(s.messages) for s in self._sessions.values())

    def summary(self) -> dict:
        """Get a summary for SSE events."""
        with self._lock:
            sessions = list(self._sessions.values())
            total_msgs = sum(len(s.messages) for s in sessions)
            last_preview = ""
            if sessions:
                last = sessions[-1]
                if last.messages:
                    last_preview = last.messages[-1].content[:60]
                    if len(last.messages[-1].content) > 60:
                        last_preview += "..."

            return {
                "active_sessions": len(sessions),
                "total_messages": total_msgs,
                "last_message_preview": last_preview,
            }
