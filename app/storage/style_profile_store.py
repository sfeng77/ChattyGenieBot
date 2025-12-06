from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
LOGGER = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


class StyleProfileStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        """Use an existing SQLite connection (shared with chat history)."""
        self._conn = conn
        LOGGER.info("Initializing StyleProfileStore with shared connection %r", conn)
        self._ensure_schema()

    def upsert_profile(
        self,
        *,
        chat_id: int,
        sender_id: str,
        label: str,
        style_prompt: str,
        analysis: Optional[Dict[str, Any]] = None,
        sample_messages: Optional[List[str]] = None,
        message_count: int = 0,
    ) -> Dict[str, Any]:
        now = _utc_now_iso()
        analysis_json = json.dumps(analysis or {}, ensure_ascii=False)
        samples_json = json.dumps(sample_messages or [], ensure_ascii=False)
        chat_id_str = str(chat_id)
        LOGGER.info(
            "Upserting style profile chat_id=%s sender_id=%s label=%s message_count=%s",
            chat_id_str,
            sender_id,
            label,
            message_count,
        )
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO style_profiles (
                        chat_id,
                        sender_id,
                        label,
                        style_prompt,
                        analysis_json,
                        sample_messages_json,
                        message_count,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id, sender_id) DO UPDATE SET
                        label = excluded.label,
                        style_prompt = excluded.style_prompt,
                        analysis_json = excluded.analysis_json,
                        sample_messages_json = excluded.sample_messages_json,
                        message_count = excluded.message_count,
                        updated_at = excluded.updated_at
                    """,
                    (
                        chat_id_str,
                        sender_id,
                        label,
                        style_prompt,
                        analysis_json,
                        samples_json,
                        int(message_count),
                        now,
                        now,
                    ),
                )
        except Exception:  # noqa: BLE001
            LOGGER.exception(
                "Failed to upsert style profile chat_id=%s sender_id=%s", chat_id_str, sender_id
            )
            raise
        profile = self.get_profile(chat_id=chat_id, sender_id=sender_id) or {}
        if not profile:
            LOGGER.warning(
                "Upsert of style profile appeared to succeed but get_profile returned nothing (chat_id=%s sender_id=%s)",
                chat_id_str,
                sender_id,
            )
        else:
            LOGGER.info(
                "Upserted style profile id=%s chat_id=%s sender_id=%s label=%s message_count=%s",
                profile.get("id"),
                profile.get("chat_id"),
                profile.get("sender_id"),
                profile.get("label"),
                profile.get("message_count"),
            )
        return profile

    def get_profile(self, *, chat_id: int, sender_id: str) -> Optional[Dict[str, Any]]:
        chat_id_str = str(chat_id)
        LOGGER.debug("Fetching style profile chat_id=%s sender_id=%s", chat_id_str, sender_id)
        cursor = self._conn.execute(
            """
            SELECT *
            FROM style_profiles
            WHERE chat_id = ? AND sender_id = ?
            """,
            (chat_id_str, sender_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_profiles_for_chat(self, *, chat_id: int) -> List[Dict[str, Any]]:
        chat_id_str = str(chat_id)
        LOGGER.debug("Listing style profiles for chat_id=%s", chat_id_str)
        cursor = self._conn.execute(
            """
            SELECT *
            FROM style_profiles
            WHERE chat_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (chat_id_str,),
        )
        rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS style_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    style_prompt TEXT NOT NULL,
                    analysis_json TEXT,
                    sample_messages_json TEXT,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(chat_id, sender_id)
                )
                """
            )
        LOGGER.info("Ensured style_profiles table exists")

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        analysis_raw = row["analysis_json"]
        samples_raw = row["sample_messages_json"]
        try:
            analysis = json.loads(analysis_raw) if analysis_raw else {}
        except Exception:
            analysis = {}
        try:
            samples = json.loads(samples_raw) if samples_raw else []
        except Exception:
            samples = []
        return {
            "id": int(row["id"]),
            "chat_id": row["chat_id"],
            "sender_id": row["sender_id"],
            "label": row["label"],
            "style_prompt": row["style_prompt"],
            "analysis": analysis,
            "sample_messages": samples,
            "message_count": int(row["message_count"] or 0),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


__all__ = ["StyleProfileStore"]
