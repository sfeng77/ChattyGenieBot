from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


def _to_iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return _utc_now_iso()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(ISO_FORMAT)


class ChatStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(
            str(self._db_path),
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._fts_enabled = False
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def add_message(
        self,
        *,
        external_conversation_id: str,
        role: str,
        content: str,
        created_at: Optional[datetime] = None,
        token_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sender_id: Optional[str] = None,
    ) -> int:
        conversation_id = self._ensure_conversation(external_conversation_id)
        created_iso = _to_iso(created_at)
        metadata_json = json.dumps(metadata or {}) if metadata else None
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO messages(
                    conversation_id,
                    role,
                    content,
                    created_at,
                    token_count,
                    metadata,
                    sender_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, created_iso, token_count, metadata_json, sender_id),
            )
            message_id = int(cursor.lastrowid)
            self._conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (created_iso, conversation_id),
            )
        return message_id

    def search_messages(
        self,
        query: str,
        *,
        external_conversation_id: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        if not query:
            return []
        params: List[Any] = []
        where: List[str] = []
        join = ""
        order = ""
        select_snippet = ""
        if self._fts_enabled:
            select_snippet = ", snippet(messages_fts, 0, '[', ']', ' … ', 24) AS snippet, bm25(messages_fts) AS score"
            sql = [
                "SELECT m.*" + select_snippet,
                "FROM messages_fts",
                "JOIN messages m ON m.id = messages_fts.rowid",
            ]
            if external_conversation_id is not None:
                sql.append("JOIN conversations c ON c.id = m.conversation_id")
                where.append("c.external_id = ?")
                params.append(external_conversation_id)
            where.append("messages_fts MATCH ?")
            params.append(query)
            order = "ORDER BY score ASC, m.created_at ASC"
        else:
            sql = [
                "SELECT m.*",
                "FROM messages m",
            ]
            if external_conversation_id is not None:
                join = "JOIN conversations c ON c.id = m.conversation_id"
                sql.append(join)
                where.append("c.external_id = ?")
                params.append(external_conversation_id)
            where.append("LOWER(m.content) LIKE LOWER(?)")
            params.append(f"%{query}%")
            order = "ORDER BY m.created_at ASC, m.id ASC"
        if start is not None:
            where.append("m.created_at >= ?")
            params.append(_to_iso(start))
        if end is not None:
            where.append("m.created_at <= ?")
            params.append(_to_iso(end))
        if where:
            sql.append("WHERE " + " AND ".join(where))
        sql.append(order)
        sql.append("LIMIT ?")
        params.append(max(1, limit))
        statement = "\n".join(part for part in sql if part)
        cursor = self._conn.execute(statement, params)
        rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_messages_in_range(
        self,
        *,
        external_conversation_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        conversation_id = self._get_conversation_id(external_conversation_id)
        if conversation_id is None:
            return []
        params: List[Any] = [conversation_id]
        where = ["conversation_id = ?"]
        if start is not None:
            where.append("created_at >= ?")
            params.append(_to_iso(start))
        if end is not None:
            where.append("created_at <= ?")
            params.append(_to_iso(end))
        sql = [
            "SELECT * FROM messages",
            "WHERE " + " AND ".join(where),
            "ORDER BY created_at ASC, id ASC",
        ]
        if limit is not None:
            sql.append("LIMIT ?")
            params.append(max(1, limit))
        cursor = self._conn.execute("\n".join(sql), params)
        rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def iter_messages(
        self,
        *,
        external_conversation_id: str,
    ) -> Iterable[Dict[str, Any]]:
        for row in self.get_messages_in_range(external_conversation_id=external_conversation_id):
            yield row

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    external_id TEXT NOT NULL UNIQUE,
                    title TEXT,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    token_count INTEGER,
                    metadata TEXT,
                    sender_id TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS messages_conversation_created_idx ON messages(conversation_id, created_at, id)"
            )
        self._maybe_add_column("messages", "sender_id", "TEXT")
        self._setup_fts()

    def _maybe_add_column(self, table: str, column: str, column_type: str) -> None:
        cursor = self._conn.execute(f"PRAGMA table_info({table})")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if column in existing_columns:
            return
        with self._conn:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def _setup_fts(self) -> None:
        try:
            if not self._fts_available():
                return
            with self._conn:
                self._conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                        content,
                        tokenize = 'unicode61 remove_diacritics 2',
                        content = 'messages',
                        content_rowid = 'id'
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                        INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
                    END;
                    """
                )
                self._conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                        DELETE FROM messages_fts WHERE rowid = old.id;
                    END;
                    """
                )
                self._conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE OF content ON messages BEGIN
                        INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
                        INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
                    END;
                    """
                )
        except sqlite3.OperationalError:
            self._fts_enabled = False
        else:
            self._fts_enabled = True

    def _fts_available(self) -> bool:
        cursor = self._conn.execute("SELECT 1 FROM pragma_module_list WHERE name = 'fts5'")
        return cursor.fetchone() is not None

    def _ensure_conversation(self, external_id: str) -> int:
        existing = self._get_conversation_id(external_id)
        if existing is not None:
            return existing
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO conversations(external_id) VALUES (?)",
                (external_id,),
            )
        return int(cursor.lastrowid)

    def _get_conversation_id(self, external_id: str) -> Optional[int]:
        cursor = self._conn.execute(
            "SELECT id FROM conversations WHERE external_id = ?",
            (external_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return int(row["id"])

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        metadata = data.get("metadata")
        if isinstance(metadata, str) and metadata:
            try:
                data["metadata"] = json.loads(metadata)
            except json.JSONDecodeError:
                data["metadata"] = {"raw": metadata}
        snippet = data.get("snippet")
        if snippet is not None and isinstance(snippet, str):
            data["snippet"] = snippet
        return data


__all__ = ["ChatStore"]
