from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List

from agents import (
    Agent,
    Runner,
    SQLiteSession,
    set_default_openai_api,
    set_default_openai_client,
    set_default_openai_key,
    set_tracing_disabled,
)
from agents import ModelSettings
from openai import AsyncOpenAI

from app.config import Settings
from app.progress import NullProgressDispatcher, ProgressDispatcher
from app.progress_hooks import ProgressHooks
from app.prompt import get_agent_instructions
from app.tools import create_disabled_web_search_tool, create_ollama_web_search_tool
from app.web_search_client import WebSearchClient

LOGGER = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session_db_path = settings.sessions_db_path
        self._progress_result_char_limit = settings.progress_tool_result_max_chars
        set_default_openai_api("chat_completions")
        set_default_openai_key(settings.openai_api_key)
        set_tracing_disabled(not settings.openai_tracing_enabled)
        if settings.openai_api_base:
            client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base,
            )
            set_default_openai_client(client)
        tools: List[object] = []
        self._web_search_tool = None
        web_search_available = False
        if settings.web_search_enabled:
            try:
                self._web_search_tool = self._build_web_search_tool()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to initialize web_search tool")
            else:
                tools.append(self._web_search_tool)
                web_search_available = True
        if self._web_search_tool is None:
            notice = (
                "web_search is currently unavailable."
                if settings.web_search_enabled
                else "web_search is disabled for this deployment."
            )
            self._web_search_tool = create_disabled_web_search_tool(message=notice)
            tools.append(self._web_search_tool)
        instructions = get_agent_instructions(web_search_available)
        self._agent = Agent(
            name="Chatty Genie",
            instructions=instructions,
            model=settings.openai_model,
            model_settings=ModelSettings(temperature=settings.openai_temperature),
            tools=tools,
        )
        self._sessions: Dict[int, SQLiteSession] = {}

    def _build_web_search_tool(self):
        client = WebSearchClient(
            base_url=self._settings.web_search_base_url,
            endpoint=self._settings.web_search_endpoint,
            timeout=self._settings.web_search_timeout,
            api_key=self._settings.web_search_api_key,
        )
        return create_ollama_web_search_tool(
            client=client,
            default_max_results=self._settings.web_search_default_max_results,
        )

    def _session_id(self, chat_id: int) -> str:
        return f"chat-{chat_id}"

    def _get_session(self, chat_id: int) -> SQLiteSession:
        session = self._sessions.get(chat_id)
        if session is None:
            session = SQLiteSession(self._session_id(chat_id), str(self._session_db_path))
            self._sessions[chat_id] = session
        return session

    async def run_message(self, chat_id: int, user_message: str) -> str:
        return await self._run(chat_id, user_message, dispatcher=None, enable_progress=False)

    async def run_message_with_progress(
        self,
        chat_id: int,
        user_message: str,
        dispatcher: ProgressDispatcher | None,
        enable_progress: bool,
    ) -> str:
        dispatcher = dispatcher or NullProgressDispatcher()
        return await self._run(chat_id, user_message, dispatcher=dispatcher, enable_progress=enable_progress)

    async def _run(
        self,
        chat_id: int,
        user_message: str,
        dispatcher: ProgressDispatcher | None,
        enable_progress: bool,
    ) -> str:
        session = self._get_session(chat_id)
        if self._settings.history_prune_enabled:
            try:
                await self._maybe_prune_session(session)
            except Exception:  # noqa: BLE001
                LOGGER.debug("History pruning failed", exc_info=True)
        hooks = None
        active_dispatcher: ProgressDispatcher | None = None
        if dispatcher is not None and enable_progress:
            active_dispatcher = dispatcher
            hooks = ProgressHooks(
                dispatcher=dispatcher,
                chat_id=chat_id,
                user_message=user_message,
                result_char_limit=self._progress_result_char_limit,
            )
        try:
            result = await Runner.run(
                self._agent,
                user_message,
                session=session,
                hooks=hooks,
            )
        except Exception as exc:  # noqa: BLE001
            if active_dispatcher is not None:
                await active_dispatcher.emit(
                    {
                        "type": "turn_failed",
                        "text": str(exc),
                        "meta": {"chat_id": chat_id, "exception_type": exc.__class__.__name__},
                    }
                )
            raise
        output = result.final_output
        if isinstance(output, str):
            return output.strip()
        if output is None:
            return ""
        return str(output).strip()

    async def _maybe_prune_session(self, session: SQLiteSession) -> None:
        keep_last = max(1, int(self._settings.history_keep_last_items))
        threshold = max(keep_last + 1, int(self._settings.history_prune_threshold_items))
        items = await session.get_items()
        if len(items) <= threshold:
            return
        older = items[:-keep_last]
        tail = items[-keep_last:]
        transcript = self._items_to_transcript(older, max_chars=self._settings.history_summary_max_chars * 4)
        summary = await self._summarize_transcript(transcript, max_chars=self._settings.history_summary_max_chars)
        if not summary:
            summary = self._fallback_summary(transcript, self._settings.history_summary_max_chars)
        summary_item: Dict[str, Any] = {"role": "system", "content": f"Earlier conversation summary (auto-generated):\n{summary}"}
        try:
            await session.clear_session()
            await session.add_items([summary_item] + tail)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Failed to rewrite session history", exc_info=True)

    def _items_to_transcript(self, items: list[Dict[str, Any]], max_chars: int) -> str:
        segments: list[str] = []
        for item in items:
            role = str(item.get("role") or "user")
            content = item.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                collected: list[str] = []
                for piece in content:
                    if not isinstance(piece, dict):
                        continue
                    for key in ("text", "input_text", "output_text", "content"):
                        val = piece.get(key)
                        if isinstance(val, str) and val.strip():
                            collected.append(val)
                            break
                text = " ".join(collected)
            elif content is not None:
                text = str(content)
            text = text.strip()
            if not text:
                continue
            segments.append(f"[{role}] {text}")
        transcript = "\n".join(segments)
        if len(transcript) > max_chars:
            return transcript[-max_chars:]
        return transcript

    async def _summarize_transcript(self, transcript: str, max_chars: int) -> str:
        if not transcript:
            return ""
        summarizer = Agent(
            name="Summarizer",
            instructions=("You are a concise summarizer. Summarize the prior conversation into a factual, neutral summary. Focus on key questions, decisions, facts, and follow-ups. Avoid speculation. Keep the summary under the requested character limit."),
            model=self._settings.openai_model,
            model_settings=ModelSettings(temperature=0.1),
            tools=[],
        )
        prompt = (
            f"Character limit: {max_chars}.\n"
            "Return only the summary text (no preface).\n\n"
            "Conversation:\n"
            f"{transcript}"
        )
        try:
            result = await Runner.run(summarizer, prompt, session=None, max_turns=1)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Summarizer call failed", exc_info=True)
            return ""
        output = result.final_output
        summary = output if isinstance(output, str) else ("" if output is None else str(output))
        summary = summary.strip()
        if len(summary) > max_chars:
            summary = summary[: max(0, max_chars - 3)].rstrip() + "..."
        return summary

    def _fallback_summary(self, transcript: str, max_chars: int) -> str:
        if not transcript:
            return ""
        trimmed = transcript[-max_chars:] if len(transcript) > max_chars else transcript
        return trimmed.strip()

    async def reset(self, chat_id: int) -> None:
        session = self._sessions.pop(chat_id, None)
        if session is None:
            session = SQLiteSession(self._session_id(chat_id), str(self._session_db_path))
        clear = getattr(session, "clear_session", None)
        if callable(clear):
            try:
                result = clear()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001
                LOGGER.debug("Failed to clear session %s", self._session_id(chat_id), exc_info=True)
        close = getattr(session, "close", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001
                LOGGER.debug("Failed to close session %s", self._session_id(chat_id), exc_info=True)

    async def aclose(self) -> None:
        for session in list(self._sessions.values()):
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Failed to close session %s", session, exc_info=True)
        self._sessions.clear()


__all__ = ["AgentRuntime"]

