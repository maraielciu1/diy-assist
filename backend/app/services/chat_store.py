"""SQLite persistence for chat sessions and messages (Stage 4)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ChatStore:
    db_path: str

    def __post_init__(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    appliance_category TEXT,
                    appliance_type TEXT,
                    brand TEXT,
                    model TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    structured_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(chat_sessions)").fetchall()
            }
            for column in ("appliance_category", "appliance_type", "brand", "model"):
                if column not in columns:
                    conn.execute(f"ALTER TABLE chat_sessions ADD COLUMN {column} TEXT")
            conn.commit()

    def create_session(
        self,
        *,
        appliance_category: str | None = None,
        appliance_type: str | None = None,
        brand: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
    ) -> str:
        sid = session_id or str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions (id, created_at, appliance_category,
                    appliance_type, brand, model)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    _utc_now(),
                    appliance_category,
                    appliance_type,
                    brand,
                    model,
                ),
            )
            conn.execute(
                """
                UPDATE chat_sessions
                SET appliance_category = COALESCE(NULLIF(?, ''), appliance_category),
                    appliance_type = COALESCE(NULLIF(?, ''), appliance_type),
                    brand = COALESCE(NULLIF(?, ''), brand),
                    model = COALESCE(NULLIF(?, ''), model)
                WHERE id = ?
                """,
                (appliance_category, appliance_type, brand, model, sid),
            )
            conn.commit()
        return sid

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_session_context(
        self,
        session_id: str,
        *,
        appliance_category: str | None = None,
        appliance_type: str | None = None,
        brand: str | None = None,
        model: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET appliance_category = COALESCE(NULLIF(?, ''), appliance_category),
                    appliance_type = COALESCE(NULLIF(?, ''), appliance_type),
                    brand = COALESCE(NULLIF(?, ''), brand),
                    model = COALESCE(NULLIF(?, ''), model)
                WHERE id = ?
                """,
                (appliance_category, appliance_type, brand, model, session_id),
            )
            conn.commit()

    def get_recent_messages(self, session_id: str, limit: int = 8) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def append_message(
        self,
        session_id: str,
        *,
        role: str,
        content: str,
        structured: dict[str, Any] | None = None,
    ) -> None:
        structured_json = json.dumps(structured, ensure_ascii=True) if structured else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, structured_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, structured_json, _utc_now()),
            )
            conn.commit()


def get_chat_store() -> ChatStore:
    path = getattr(settings, "chat_db_path", "./data/chat.sqlite")
    return ChatStore(db_path=str(path))
