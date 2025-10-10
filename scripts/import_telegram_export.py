from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.storage.chat_store import ChatStore
from app.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Telegram JSON export into ChatStore")
    parser.add_argument("json_path", type=Path, help="Path to Telegram export JSON file")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Chat history SQLite database path (defaults to CHAT_HISTORY_DB_PATH)",
    )
    parser.add_argument(
        "--external-id",
        type=str,
        default=None,
        help="Override external conversation id (defaults to tg:<chat_id> from JSON)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override conversation title (defaults to JSON name)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Insert even if conversation already has messages",
    )
    return parser.parse_args()


def flatten_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        parts: List[str] = []
        for chunk in text:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, dict):
                val = chunk.get("text")
                if isinstance(val, str):
                    parts.append(val)
        return "".join(parts)
    return str(text)


def format_service_message(msg: Dict[str, Any]) -> str:
    actor = msg.get("actor") or "(unknown)"
    action = msg.get("action") or msg.get("type") or "service"
    members: Iterable[str] = msg.get("members") or []
    members_text = ", ".join(members)
    if members_text:
        return f"{actor} {action.replace('_', ' ')}: {members_text}"
    title = msg.get("title")
    if title:
        return f"{actor} {action.replace('_', ' ')}: {title}"
    text = flatten_text(msg.get("text"))
    if text:
        return f"{actor} {action.replace('_', ' ')}: {text}"
    return f"{actor} {action.replace('_', ' ')}"


def build_content(msg: Dict[str, Any]) -> str:
    base = flatten_text(msg.get("text"))
    attachments: List[str] = []
    if msg.get("photo"):
        attachments.append("[photo]")
    if msg.get("media_type"):
        attachments.append(f"[{msg['media_type']}]")
    if msg.get("file"):
        name = msg.get("file_name") or msg.get("file")
        attachments.append(f"[file: {name}]")
    if msg.get("sticker_emoji"):
        attachments.append(f"[sticker {msg['sticker_emoji']}]")
    if msg.get("inline_bot_buttons"):
        attachments.append("[buttons]")
    if attachments:
        if base:
            return base + "\n" + " ".join(attachments)
        return " ".join(attachments)
    return base


def infer_role(sender_name: str | None) -> str:
    if not sender_name:
        return "user"
    lowered = sender_name.lower()
    if "bot" in lowered or "agent" in lowered:
        return "assistant"
    return "user"


def import_messages(store: ChatStore, data: Dict[str, Any], external_id: str, force: bool) -> int:
    conn = store._conn  # noqa: SLF001 - internal access for migration utility
    existing = conn.execute(
        """
        SELECT COUNT(*)
        FROM messages m
        JOIN conversations c ON c.id = m.conversation_id
        WHERE c.external_id = ?
        """,
        (external_id,),
    ).fetchone()[0]
    if existing and not force:
        raise RuntimeError(
            f"Conversation {external_id} already has {existing} messages. Use --force to import anyway."
        )

    inserted = 0
    for msg in data.get("messages", []):
        msg_type = msg.get("type")
        if msg_type not in {"message", "service"}:
            continue
        if msg_type == "service":
            content = format_service_message(msg)
            role = "system"
            sender_id = msg.get("actor_id")
            created = datetime.fromtimestamp(int(msg["date_unixtime"]), tz=timezone.utc)
            metadata = {
                "telegram_message_id": msg.get("id"),
                "raw_type": msg_type,
                "action": msg.get("action"),
            }
        else:
            content = build_content(msg)
            role = infer_role(msg.get("from"))
            sender_id = msg.get("from_id")
            created = datetime.fromtimestamp(int(msg["date_unixtime"]), tz=timezone.utc)
            metadata = {
                "telegram_message_id": msg.get("id"),
                "raw_type": msg_type,
            }
            if msg.get("reply_to_message_id") is not None:
                metadata["reply_to_message_id"] = msg["reply_to_message_id"]
            if msg.get("edited_unixtime"):
                metadata["edited_at"] = datetime.fromtimestamp(
                    int(msg["edited_unixtime"]), tz=timezone.utc
                ).isoformat()
            if msg.get("reactions"):
                metadata["reactions"] = msg["reactions"]
            if msg.get("media_type"):
                metadata["media_type"] = msg["media_type"]
            if msg.get("file"):
                metadata["file"] = {
                    "name": msg.get("file_name"),
                    "path": msg.get("file"),
                    "mime_type": msg.get("mime_type"),
                    "size": msg.get("file_size"),
                }
            if msg.get("photo"):
                metadata["photo"] = {
                    "path": msg.get("photo"),
                    "size": msg.get("photo_file_size"),
                    "width": msg.get("width"),
                    "height": msg.get("height"),
                }

        sender_id_str = str(sender_id) if sender_id is not None else None
        metadata = {k: v for k, v in metadata.items() if v is not None}

        store.add_message(
            external_conversation_id=external_id,
            role=role,
            content=content or "",
            created_at=created,
            metadata=metadata or None,
            sender_id=sender_id_str,
        )
        inserted += 1

    return inserted


def main() -> None:
    args = parse_args()
    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    external_id = args.external_id or f"tg:{data.get('id')}"
    title = args.title or data.get("name")

    db_path = args.db_path or get_settings().chat_history_db_path
    store = ChatStore(db_path)
    inserted = import_messages(store, data, external_id, force=args.force)

    if title:
        store._conn.execute(  # noqa: SLF001 - internal access for utility
            "UPDATE conversations SET title = ? WHERE external_id = ?",
            (title, external_id),
        )
        store._conn.commit()

    print(f"Imported {inserted} messages into conversation {external_id} (db={db_path})")
    store.close()


if __name__ == "__main__":
    main()
