"""
Short-term memory: holds the current conversation context in-memory.
Supports a configurable window size to avoid unbounded growth.
"""
from typing import List, Dict, Any
from datetime import datetime


class ShortTermMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._history: List[Dict[str, Any]] = []
        self._session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_turn(self, role: str, content: str, metadata: Dict = None):
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._history.append(entry)
        # Trim to max window (keep system messages)
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def get_messages(self) -> List[Dict[str, str]]:
        """Return history in Anthropic message format (role + content only)."""
        return [{"role": h["role"], "content": h["content"]} for h in self._history]

    def get_last_n(self, n: int) -> List[Dict[str, Any]]:
        return self._history[-n:]

    def clear(self):
        self._history = []
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    @property
    def session_id(self) -> str:
        return self._session_id

    def summary(self) -> str:
        return f"Session {self._session_id}: {len(self._history)} messages"
