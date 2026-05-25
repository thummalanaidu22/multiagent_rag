"""
Long-term memory: persists Q&A interactions, document metadata, and user
preferences across sessions using SQLite.
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import LONG_TERM_DB_PATH


class LongTermMemory:
    def __init__(self, db_path: str = LONG_TERM_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT,
                    query       TEXT,
                    answer      TEXT,
                    context     TEXT,
                    sources     TEXT,
                    timestamp   TEXT,
                    feedback    TEXT
                );

                CREATE TABLE IF NOT EXISTS documents (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename    TEXT UNIQUE,
                    title       TEXT,
                    num_chunks  INTEGER,
                    ingested_at TEXT
                );

                CREATE TABLE IF NOT EXISTS key_value (
                    key         TEXT PRIMARY KEY,
                    value       TEXT,
                    updated_at  TEXT
                );
            """)

    # ---- Interactions ----

    def save_interaction(
        self,
        session_id: str,
        query: str,
        answer: str,
        context: List[str] = None,
        sources: List[str] = None,
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO interactions
                   (session_id, query, answer, context, sources, timestamp)
                   VALUES (?,?,?,?,?,?)""",
                (
                    session_id,
                    query,
                    answer,
                    json.dumps(context or []),
                    json.dumps(sources or []),
                    datetime.now().isoformat(),
                ),
            )
            return cur.lastrowid

    def get_recent_interactions(self, limit: int = 20) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM interactions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def search_interactions(self, keyword: str, limit: int = 5) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM interactions
                   WHERE query LIKE ? OR answer LIKE ?
                   ORDER BY id DESC LIMIT ?""",
                (f"%{keyword}%", f"%{keyword}%", limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- Documents ----

    def register_document(self, filename: str, title: str, num_chunks: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO documents (filename, title, num_chunks, ingested_at)
                   VALUES (?,?,?,?)""",
                (filename, title, num_chunks, datetime.now().isoformat()),
            )

    def get_documents(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM documents ORDER BY ingested_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ---- Key-Value store ----

    def set(self, key: str, value: Any):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO key_value (key, value, updated_at) VALUES (?,?,?)",
                (key, json.dumps(value), datetime.now().isoformat()),
            )

    def get(self, key: str, default=None) -> Any:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT value FROM key_value WHERE key=?", (key,)
            ).fetchone()
        if row is None:
            return default
        return json.loads(row[0])
