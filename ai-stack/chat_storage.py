from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class ChatStorage:
    DEFAULT_MAX_MESSAGES = 100

    def __init__(self, db_path: str | Path | None = None, max_messages: int = DEFAULT_MAX_MESSAGES):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent
            db_path = base_dir / "chat_history.db"
        self.db_path = Path(db_path)
        self.max_messages = max_messages
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'Nuova chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                active_branch_id TEXT,
                branch_seq INTEGER NOT NULL DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                model TEXT,
                intent TEXT,
                at TEXT NOT NULL,
                created_at TEXT,
                checkpoint_id TEXT,
                parent_message_id TEXT,
                branch_id TEXT,
                thinking TEXT,
                think_requested INTEGER,
                think_applied TEXT,
                think_status TEXT,
                warnings TEXT,
                streaming INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_at ON messages(session_id, at)
        """)
        conn.commit()

    def _now_iso(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def create_session(self, title: str = "Nuova chat") -> dict[str, Any]:
        conn = self._get_conn()
        session_id = str(uuid.uuid4())
        now = self._now_iso()
        cursor = conn.execute(
            """INSERT INTO sessions (id, title, created_at, updated_at, active_branch_id, branch_seq)
               VALUES (?, ?, ?, ?, NULL, 0)""",
            (session_id, title, now, now),
        )
        conn.commit()
        return {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "active_branch_id": None,
            "branch_seq": 0,
            "messages": [],
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            return None
        session = dict(row)
        if session.get("metadata"):
            session["metadata"] = json.loads(session["metadata"])
        else:
            session["metadata"] = {}
        messages = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY at ASC",
            (session_id,),
        ).fetchall()
        session["messages"] = [self._row_to_message(m) for m in messages]
        return session

    def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        result = []
        for row in rows:
            session = dict(row)
            if session.get("metadata"):
                session["metadata"] = json.loads(session["metadata"])
            else:
                session["metadata"] = {}
            msg_count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session["id"],),
            ).fetchone()[0]
            session["message_count"] = msg_count
            result.append(session)
        return result

    def update_session(self, session_id: str, title: str | None = None, active_branch_id: str | None = None, branch_seq: int | None = None) -> dict[str, Any] | None:
        conn = self._get_conn()
        now = self._now_iso()
        updates = ["updated_at = ?"]
        params: list[Any] = [now]
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if active_branch_id is not None:
            updates.append("active_branch_id = ?")
            params.append(active_branch_id)
        if branch_seq is not None:
            updates.append("branch_seq = ?")
            params.append(branch_seq)
        params.append(session_id)
        conn.execute(
            f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()
        return self.get_session(session_id)

    def add_message(self, session_id: str, message: dict[str, Any]) -> dict[str, Any]:
        conn = self._get_conn()
        now = self._now_iso()
        msg_id = message.get("id") or str(uuid.uuid4())
        at = message.get("at") or now
        conn.execute(
            """INSERT INTO messages 
               (id, session_id, role, text, model, intent, at, created_at, checkpoint_id, parent_message_id, branch_id, thinking, think_requested, think_applied, think_status, warnings, streaming)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_id,
                session_id,
                message.get("role", "user"),
                message.get("text", ""),
                message.get("model"),
                message.get("intent"),
                at,
                now,
                message.get("checkpoint_id"),
                message.get("parent_message_id"),
                message.get("branch_id"),
                message.get("thinking"),
                1 if message.get("think_requested") else 0 if "think_requested" in message else None,
                message.get("think_applied"),
                message.get("think_status"),
                json.dumps(message.get("warnings")) if message.get("warnings") else None,
                1 if message.get("streaming") else 0,
            ),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        self._prune_session(session_id)
        conn.commit()
        return self.get_message(msg_id)

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_message(row)

    def update_message(self, message_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        conn = self._get_conn()
        fields = []
        params: list[Any] = []
        for key in ["text", "model", "intent", "thinking", "think_status", "warnings", "streaming"]:
            if key in updates:
                fields.append(f"{key} = ?")
                val = updates[key]
                if key == "warnings":
                    val = json.dumps(val) if val else None
                elif key == "streaming":
                    val = 1 if val else 0
                params.append(val)
        if not fields:
            return self.get_message(message_id)
        params.append(message_id)
        conn.execute(
            f"UPDATE messages SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        now = self._now_iso()
        session_id = conn.execute(
            "SELECT session_id FROM messages WHERE id = ?", (message_id,)
        ).fetchone()
        if session_id:
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id[0]),
            )
        conn.commit()
        return self.get_message(message_id)

    def delete_session(self, session_id: str) -> bool:
        conn = self._get_conn()
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return True

    def _prune_session(self, session_id: str) -> None:
        conn = self._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        if count > self.max_messages:
            excess = count - self.max_messages
            conn.execute(
                """DELETE FROM messages WHERE id IN 
                   (SELECT id FROM messages WHERE session_id = ? ORDER BY at ASC LIMIT ?)""",
                (session_id, excess),
            )

    def _row_to_message(self, row: sqlite3.Row) -> dict[str, Any]:
        msg = dict(row)
        msg.pop("session_id", None)
        if msg.get("warnings"):
            try:
                msg["warnings"] = json.loads(msg["warnings"])
            except json.JSONDecodeError:
                msg["warnings"] = []
        else:
            msg["warnings"] = []
        if msg.get("think_requested") is not None:
            msg["think_requested"] = bool(msg["think_requested"])
        if msg.get("streaming") is not None:
            msg["streaming"] = bool(msg["streaming"])
        return msg

    def import_from_localstorage(self, local_sessions: list[dict[str, Any]]) -> dict[str, int]:
        imported = 0
        for local_session in local_sessions:
            session_id = local_session.get("id") or str(uuid.uuid4())
            now = self._now_iso()
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO sessions (id, title, created_at, updated_at, active_branch_id, branch_seq, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    local_session.get("title", "Nuova chat"),
                    local_session.get("createdAt") or now,
                    local_session.get("updatedAt") or now,
                    local_session.get("active_branch_id"),
                    local_session.get("branch_seq", 0),
                    json.dumps(local_session.get("metadata", {})),
                ),
            )
            for msg in local_session.get("messages", []):
                msg_id = msg.get("id") or str(uuid.uuid4())
                conn.execute(
                    """INSERT OR REPLACE INTO messages 
                       (id, session_id, role, text, model, intent, at, created_at, checkpoint_id, parent_message_id, branch_id, thinking, think_requested, think_applied, think_status, warnings, streaming)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        msg_id,
                        session_id,
                        msg.get("role", "user"),
                        msg.get("text", ""),
                        msg.get("model"),
                        msg.get("intent"),
                        msg.get("at") or now,
                        msg.get("created_at") or now,
                        msg.get("checkpoint_id"),
                        msg.get("parent_message_id"),
                        msg.get("branch_id"),
                        msg.get("thinking"),
                        1 if msg.get("think_requested") else 0 if "think_requested" in msg else None,
                        msg.get("think_applied"),
                        msg.get("think_status"),
                        json.dumps(msg.get("warnings")) if msg.get("warnings") else None,
                        1 if msg.get("streaming") else 0,
                    ),
                )
            imported += 1
        conn.commit()
        return {"sessions_imported": imported}

    def export_all(self) -> list[dict[str, Any]]:
        sessions = self.list_sessions(limit=1000)
        for session in sessions:
            session.pop("message_count", None)
        return sessions


def default_storage() -> ChatStorage:
    return ChatStorage()
