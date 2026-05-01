from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Optional

from agents import function_tool
from pydantic import Field

from app.features.reminder.tool import current_chat_id

if TYPE_CHECKING:
    from app.storage.chat_store import ChatStore

LOGGER = logging.getLogger(__name__)


def create_search_memory_tool(store: "ChatStore"):
    """Return a search_memory agent tool bound to *store*."""

    @function_tool(name_override="search_memory")
    async def search_memory(
        query: Annotated[
            str,
            Field(description="Keywords or phrase to search in past conversations."),
        ],
        limit: Annotated[
            Optional[int],
            Field(ge=1, le=20, description="Max results to return (1-20, default 5)."),
        ] = 5,
    ) -> dict:
        """Search past conversations for relevant context."""
        chat_id = current_chat_id.get()
        if chat_id is None:
            return {"error": "Unable to determine chat context."}

        external_id = f"chat-{chat_id}"
        hits = store.search_messages(
            query,
            external_conversation_id=external_id,
            limit=limit or 5,
        )
        if not hits:
            return {"query": query, "results": [], "count": 0}

        results = []
        for row in hits:
            results.append({
                "role": row.get("role"),
                "text": row.get("snippet") or (row.get("content") or "")[:200],
                "time": row.get("created_at"),
            })

        return {"query": query, "results": results, "count": len(results)}

    return search_memory


__all__ = ["create_search_memory_tool"]
