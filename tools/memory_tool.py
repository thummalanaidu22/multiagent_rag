"""
MemoryTool exposes long-term memory read/write as Claude tool-use schemas.
"""
from typing import List, Dict, Any
from memory.long_term import LongTermMemory


class MemoryTool:
    SAVE_SCHEMA = {
        "name": "save_to_memory",
        "description": "Persist a key fact or Q&A interaction to long-term memory for future recall.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "A unique identifier for this memory entry."},
                "value": {"type": "string", "description": "The content to store."},
            },
            "required": ["key", "value"],
        },
    }

    RETRIEVE_SCHEMA = {
        "name": "retrieve_from_memory",
        "description": "Look up past interactions or stored facts from long-term memory using a keyword.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Keyword to search past Q&A interactions.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results to return.",
                    "default": 3,
                },
            },
            "required": ["keyword"],
        },
    }

    def __init__(self):
        self._ltm = LongTermMemory()

    def save(self, key: str, value: str) -> Dict[str, str]:
        self._ltm.set(key, value)
        return {"status": "saved", "key": key}

    def retrieve(self, keyword: str, limit: int = 3) -> List[Dict[str, Any]]:
        results = self._ltm.search_interactions(keyword, limit=limit)
        if not results:
            kv = self._ltm.get(keyword)
            if kv:
                return [{"key": keyword, "value": kv}]
        return results
