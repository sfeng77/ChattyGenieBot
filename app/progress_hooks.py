from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from agents import Agent
from agents.items import ItemHelpers, ModelResponse
from agents.lifecycle import RunHooks
from agents.run_context import RunContextWrapper, TContext
from agents.tool import Tool

from app.progress import ProgressDispatcher, ProgressEvent

LOGGER = logging.getLogger(__name__)


class ProgressHooks(RunHooks[TContext]):
    """Emit progress events during an agent run."""

    def __init__(
        self,
        dispatcher: ProgressDispatcher,
        chat_id: int,
        user_message: str,
        result_char_limit: int,
    ) -> None:
        self._dispatcher = dispatcher
        self._chat_id = chat_id
        self._user_message = user_message
        self._result_char_limit = max(result_char_limit, 32)
        self._start_time = time.monotonic()

    async def on_agent_start(self, context: RunContextWrapper[TContext], agent: Agent[TContext]) -> None:
        await self._emit(
            {
                "type": "turn_started",
                "text": self._truncate(self._user_message),
                "meta": {
                    "chat_id": self._chat_id,
                    "input_chars": len(self._user_message),
                    "agent": agent.name,
                },
            }
        )

    async def on_llm_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        preview = self._preview_prompt(input_items)
        await self._emit(
            {
                "type": "llm_started",
                "text": f"Model call started for {agent.name}",
                "meta": {
                    "chat_id": self._chat_id,
                    "prompt_items": len(input_items),
                    "prompt_preview": preview,
                    "system_prompt": self._truncate(system_prompt) if system_prompt else None,
                },
            }
        )

    async def on_llm_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        response: ModelResponse,
    ) -> None:
        summary = self._extract_model_summary(response)
        await self._emit(
            {
                "type": "llm_finished",
                "text": summary,
                "meta": {
                    "chat_id": self._chat_id,
                    "usage": {
                        "input_tokens": getattr(response.usage, "input_tokens", None),
                        "output_tokens": getattr(response.usage, "output_tokens", None),
                    },
                    "response_preview": summary,
                },
            }
        )

    async def on_tool_start(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
    ) -> None:
        await self._emit(
            {
                "type": "tool_started",
                "text": self._tool_name(tool),
                "meta": {
                    "chat_id": self._chat_id,
                    "agent": agent.name,
                },
            }
        )

    async def on_tool_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        tool: Tool,
        result: Any,
    ) -> None:
        event = self._summarize_tool_result(tool, result, agent.name)
        await self._emit(event)

    async def on_agent_end(
        self,
        context: RunContextWrapper[TContext],
        agent: Agent[TContext],
        output: Any,
    ) -> None:
        text = self._coerce_output(output)
        await self._emit(
            {
                "type": "turn_finished",
                "text": text,
                "meta": {
                    "chat_id": self._chat_id,
                    "agent": agent.name,
                },
            }
        )

    def _elapsed_seconds(self) -> float:
        return round(time.monotonic() - self._start_time, 3)

    async def _emit(self, event: ProgressEvent) -> None:
        meta = event.setdefault("meta", {})
        meta.setdefault("chat_id", self._chat_id)
        meta["elapsed_seconds"] = self._elapsed_seconds()
        await self._dispatcher.emit(event)

    def _truncate(self, value: object) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if len(value) <= self._result_char_limit:
            return value
        return f"{value[: self._result_char_limit - 3]}..."

    def _preview_prompt(self, input_items: list[Any]) -> str:
        if not input_items:
            return ""
        parts: list[str] = []
        for item in input_items:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                    continue
                if isinstance(content, list):
                    text_segments = [seg.get("text") for seg in content if isinstance(seg, dict) and seg.get("type") == "output_text"]
                    if text_segments:
                        parts.append(" ".join(filter(None, text_segments)))
                        continue
            elif isinstance(item, str):
                parts.append(item)
        if not parts:
            return ""
        preview = " \n".join(parts)
        return self._truncate(preview)

    def _extract_model_summary(self, response: ModelResponse) -> str:
        for item in reversed(response.output):
            text = ItemHelpers.extract_last_text(item)
            if text:
                return self._truncate(text)
        return "Model response received"

    def _tool_name(self, tool: Tool) -> str:
        name = getattr(tool, "name", None)
        if name:
            return name
        return tool.__class__.__name__

    def _summarize_tool_result(self, tool: Tool, result: Any, agent_name: str) -> ProgressEvent:
        meta: dict[str, Any] = {
            "chat_id": self._chat_id,
            "agent": agent_name,
        }
        error_text: Optional[str] = None
        summary_text: Optional[str] = None

        payload: Any | None = None
        decoded: Any = result
        if isinstance(decoded, (bytes, bytearray)):
            decoded = decoded.decode("utf-8", errors="ignore")
        if isinstance(decoded, dict):
            payload = decoded
        elif isinstance(decoded, str):
            try:
                payload = json.loads(decoded)
            except Exception:  # noqa: BLE001
                summary_text = self._truncate(decoded)
            else:
                if payload is None:
                    payload = decoded
        else:
            payload = decoded

        urls: list[str] = []

        if isinstance(payload, dict):
            query = payload.get("query")
            if isinstance(query, str):
                meta["query"] = self._truncate(query)
            error = payload.get("error")
            if isinstance(error, str) and error:
                error_text = self._truncate(error)
            results = payload.get("results")
            if isinstance(results, list):
                meta["results_count"] = len(results)
                if results:
                    first = results[0]
                    if isinstance(first, dict):
                        top_url = first.get("url") or first.get("link")
                        if isinstance(top_url, str) and top_url.strip():
                            meta["top_url"] = top_url.strip()
                    for item in results:
                        if isinstance(item, dict):
                            link = item.get("url") or item.get("link")
                            if isinstance(link, str) and link.strip():
                                cleaned_link = link.strip()
                                if cleaned_link not in urls:
                                    urls.append(cleaned_link)
                snippet = _first_snippet(results)
                if snippet:
                    summary_text = self._truncate(snippet)
            if summary_text is None:
                details = payload.get("summary") or payload.get("message")
                if isinstance(details, str) and details:
                    summary_text = self._truncate(details)
        elif isinstance(payload, list):
            meta["results_count"] = len(payload)
            snippet = _first_snippet(payload)
            if snippet:
                summary_text = self._truncate(snippet)
            for item in payload:
                if isinstance(item, dict):
                    link = item.get("url") or item.get("link")
                    if isinstance(link, str) and link.strip():
                        cleaned = link.strip()
                        if cleaned not in urls:
                            urls.append(cleaned)
        elif payload is not None and summary_text is None:
            summary_text = self._truncate(payload)

        event_type = "tool_finished"
        if error_text:
            event_type = "tool_error"
            meta["error"] = error_text
        if summary_text:
            meta["summary"] = summary_text
        if urls:
            meta["results_urls"] = urls

        return {
            "type": event_type,
            "text": self._tool_name(tool),
            "meta": meta,
        }

    def _coerce_output(self, output: Any) -> str:
        if isinstance(output, str):
            return output.strip()
        if output is None:
            return ""
        return self._truncate(output)


def _first_snippet(items: list[Any]) -> Optional[str]:
    for item in items:
        if isinstance(item, dict):
            snippet = item.get("snippet")
            if isinstance(snippet, str) and snippet:
                return snippet
            title = item.get("title")
            if isinstance(title, str) and title:
                return title
        elif isinstance(item, str) and item:
            return item
    return None


__all__ = ["ProgressHooks"]
