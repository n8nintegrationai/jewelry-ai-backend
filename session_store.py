import time
from typing import Dict, List


class SessionStore:
    """In-memory session store for chat history. No external dependencies."""

    def __init__(self, max_idle_seconds: int = 1800):  # 30 minutes
        self._store: Dict[str, Dict] = {}
        self.max_idle_seconds = max_idle_seconds

    def _is_expired(self, timestamp: float) -> bool:
        return time.time() - timestamp > self.max_idle_seconds

    def _cleanup_expired(self):
        expired_keys = [
            session_id for session_id, data in self._store.items()
            if self._is_expired(data["timestamp"])
        ]
        for key in expired_keys:
            del self._store[key]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get the last 10 message turns for a session."""
        self._cleanup_expired()
        session = self._store.get(session_id)
        if session:
            return session["history"][-10:]  # Last 10 turns
        return []

    def add_turn(self, session_id: str, role: str, content: str):
        """Add a new turn to the session history."""
        if not self._store.get(session_id):
            self._store[session_id] = {"history": [], "timestamp": time.time()}

        self._store[session_id]["history"].append(
            {"role": role, "content": content})
        self._store[session_id]["timestamp"] = time.time()

        # Keep only last 10 turns
        self._store[session_id]["history"] = self._store[session_id]["history"][-10:]


# Global instance
session_store = SessionStore()
