from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(ISO_FORMAT)


def parse_remind_at(s: str) -> datetime:
    """Parse an ISO datetime string from the DB into an aware UTC datetime."""
    try:
        dt = datetime.strptime(s, ISO_FORMAT)
    except ValueError:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ReminderStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    remind_at TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    done INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS reminders_chat_remind_idx "
                "ON reminders(chat_id, remind_at)"
            )

    def add_reminder(self, *, chat_id: int, remind_at: datetime, message: str) -> int:
        remind_iso = _to_iso(remind_at)
        created_iso = _utc_now_iso()
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO reminders(chat_id, remind_at, message, created_at, done)
                VALUES (?, ?, ?, ?, 0)
                """,
                (chat_id, remind_iso, message, created_iso),
            )
        return int(cursor.lastrowid)

    def get_pending_reminders(self) -> List[Dict[str, Any]]:
        cursor = self._conn.execute(
            "SELECT * FROM reminders WHERE done = 0 ORDER BY remind_at ASC"
        )
        return [dict(row) for row in cursor.fetchall()]

    def mark_done(self, reminder_id: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE reminders SET done = 1 WHERE id = ?",
                (reminder_id,),
            )


__all__ = ["ReminderStore", "parse_remind_at"]
