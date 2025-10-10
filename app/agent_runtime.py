from __future__ import annotations

import inspect
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

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
from app.finance_client import AlphaVantageClient
from app.prompt import get_agent_instructions
from app.storage.chat_store import ChatStore
from app.tools import (
    create_disabled_finance_tool,
    create_disabled_vision_tool,
    create_disabled_web_search_tool,
    create_ollama_web_search_tool,
    create_stock_trend_tool,
    create_vision_tool,
)
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
        self._finance_tool = None
        self._vision_tool = None
        web_search_available = False
        finance_available = False
        vision_available = False
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
        if settings.finance_enabled:
            try:
                self._finance_tool = self._build_finance_tool()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to initialize stock_trend tool")
            else:
                tools.append(self._finance_tool)
                finance_available = True
        if self._finance_tool is None:
            notice = (
                "stock_trend is currently unavailable."
                if settings.finance_enabled
                else "stock_trend is disabled for this deployment."
            )
            self._finance_tool = create_disabled_finance_tool(message=notice)
            tools.append(self._finance_tool)
        if settings.vision_enabled:
            try:
                self._vision_tool = self._build_vision_tool()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to initialize vision tool")
            else:
                tools.append(self._vision_tool)
                vision_available = True
        if self._vision_tool is None:
            notice = (
                "vision_analyze is currently unavailable."
                if settings.vision_enabled
                else "vision_analyze is disabled for this deployment."
            )
            self._vision_tool = create_disabled_vision_tool(message=notice)
            tools.append(self._vision_tool)
        instructions = get_agent_instructions(web_search_available, finance_available, vision_available)
        self._agent = Agent(
            name="Chatty Genie",
            instructions=instructions,
            model=settings.openai_model,
            model_settings=ModelSettings(temperature=settings.openai_temperature),
            tools=tools,
        )
        self._sessions: Dict[int, SQLiteSession] = {}
        self._chat_store = ChatStore(settings.chat_history_db_path)

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

    def _build_finance_tool(self):
        provider = (self._settings.finance_provider or "").lower()
        if provider == "alpha_vantage":
            api_key = self._settings.finance_api_key
            if not api_key:
                raise ValueError("FINANCE_API_KEY must be set to use Alpha Vantage")
            client = AlphaVantageClient(
                api_key=api_key,
                timeout=self._settings.finance_timeout,
            )
            provider_label = "alpha_vantage"
        else:
            raise ValueError(f"Unsupported finance provider: {self._settings.finance_provider}")
        return create_stock_trend_tool(
            client=client,
            provider_name=provider_label,
            default_window_days=self._settings.finance_default_window_days,
            cache_ttl_minutes=self._settings.finance_cache_ttl_minutes,
        )

    def _build_vision_tool(self):
        return create_vision_tool(self._settings)

    def _session_id(self, chat_id: int) -> str:
        return f"chat-{chat_id}"

    def _history_id(self, chat_id: int) -> str:
        return self._session_id(chat_id)

    def _get_session(self, chat_id: int) -> SQLiteSession:
        session = self._sessions.get(chat_id)
        if session is None:
            session = SQLiteSession(self._session_id(chat_id), str(self._session_db_path))
            self._sessions[chat_id] = session
        return session

    async def run_message(self, chat_id: int, user_message: str, *, sender_id: str | None = None, log_user: bool = True) -> str:
        return await self._run(
            chat_id,
            user_message,
            dispatcher=None,
            enable_progress=False,
            sender_id=sender_id,
            log_user=log_user,
        )

    async def run_message_with_progress(
        self,
        chat_id: int,
        user_message: str,
        dispatcher: ProgressDispatcher | None,
        enable_progress: bool,
        *,
        sender_id: str | None = None,
        log_user: bool = True,
    ) -> str:
        dispatcher = dispatcher or NullProgressDispatcher()
        return await self._run(
            chat_id,
            user_message,
            dispatcher=dispatcher,
            enable_progress=enable_progress,
            sender_id=sender_id,
            log_user=log_user,
        )

    async def _run(
        self,
        chat_id: int,
        user_message: str,
        dispatcher: ProgressDispatcher | None,
        enable_progress: bool,
        *,
        sender_id: str | None = None,
        log_user: bool = True,
    ) -> str:
        session = self._get_session(chat_id)
        if self._settings.history_prune_enabled:
            try:
                await self._maybe_prune_session(session)
            except Exception:  # noqa: BLE001
                LOGGER.exception("History pruning failed", exc_info=True)
        history_id = self._history_id(chat_id)
        if log_user:
            try:
                self._chat_store.add_message(
                    external_conversation_id=history_id,
                    role="user",
                    content=user_message,
                    sender_id=sender_id,
                )
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to persist user message", exc_info=True)
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
            response = output.strip()
        elif output is None:
            response = ""
        else:
            response = str(output).strip()
        if response:
            try:
                self._chat_store.add_message(
                    external_conversation_id=history_id,
                    role="assistant",
                    content=response,
                    sender_id="assistant",
                )
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to persist assistant message", exc_info=True)
        return response

    def search_history(self, chat_id: int, query: str, *, limit: int = 50) -> List[Dict[str, Any]]:
        history_id = self._history_id(chat_id)
        try:
            return self._chat_store.search_messages(
                query,
                external_conversation_id=history_id,
                limit=limit,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("History search failed", exc_info=True)
            return []

    def log_message(
        self,
        chat_id: int,
        *,
        content: str,
        role: str = "user",
        sender_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a message into the chat history without running the agent."""
        history_id = self._history_id(chat_id)
        try:
            self._chat_store.add_message(
                external_conversation_id=history_id,
                role=role,
                content=content or "",
                created_at=created_at,
                metadata=metadata,
                sender_id=sender_id,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to log message", exc_info=True)

    def get_history_messages(
        self,
        chat_id: int,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        history_id = self._history_id(chat_id)
        try:
            return self._chat_store.get_messages_in_range(
                external_conversation_id=history_id,
                start=start,
                end=end,
                limit=limit,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("History fetch failed", exc_info=True)
            return []

    async def recap_history(
        self,
        chat_id: int,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        max_chars = max_chars or self._settings.history_summary_max_chars
        messages = self.get_history_messages(chat_id, start=start, end=end)
        if not messages:
            return ""
        transcript = self._messages_to_transcript(messages, max_chars * 4)
        summary = await self._summarize_transcript(transcript, max_chars=max_chars)
        if summary:
            return summary
        return self._fallback_summary(transcript, max_chars)

    async def answer_from_history(
        self,
        chat_id: int,
        question: str,
        *,
        top_k: int = 20,
        max_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        max_chars = max_chars or self._settings.history_summary_max_chars
        hits = self.search_history(chat_id, question, limit=top_k)
        if not hits:
            return {"answer": "", "context": []}
        context = self._order_history_hits(hits)
        context_text = self._format_history_context(context, max_chars * 6)
        prompt = (
            "You are a helpful assistant. Answer the user's question using only the provided chat history. "
            "If the history does not contain the answer, reply with 'I do not have enough information.'\n\n"
            f"History:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        responder = Agent(
            name="History QA",
            instructions="Answer questions based strictly on the given history.",
            model=self._settings.openai_model,
            model_settings=ModelSettings(temperature=0.1),
            tools=[],
        )
        try:
            result = await Runner.run(responder, prompt, session=None, max_turns=1)
            output = result.final_output
        except Exception:  # noqa: BLE001
            LOGGER.exception("History QA failed", exc_info=True)
            return {"answer": "", "context": context}
        answer = output if isinstance(output, str) else ("" if output is None else str(output))
        return {"answer": answer.strip(), "context": context}

    def _messages_to_transcript(self, messages: List[Dict[str, Any]], max_chars: int) -> str:
        items: List[Dict[str, Any]] = []
        for message in messages:
            items.append({"role": message.get("role", "user"), "content": message.get("content", "")})
        return self._items_to_transcript(items, max_chars=max_chars)

    def _order_history_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def sort_key(item: Dict[str, Any]) -> tuple:
            created = item.get("created_at") or ""
            return created, item.get("id", 0)

        return sorted(hits, key=sort_key)

    def _format_history_context(self, messages: List[Dict[str, Any]], max_chars: int) -> str:
        segments: List[str] = []
        for entry in messages:
            timestamp = entry.get("created_at", "")
            role = entry.get("role", "user")
            sender = entry.get("sender_id") or ""
            content = (entry.get("content") or "").strip()
            if not content:
                continue
            sender_part = f"[{sender}]" if sender else ""
            segments.append(f"[{timestamp}][{role}]{sender_part} {content}")
        text = "\n".join(segments)
        if len(text) > max_chars:
            return text[-max_chars:]
        return text

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
            LOGGER.exception("Failed to rewrite session history", exc_info=True)

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
            LOGGER.exception("Summarizer call failed", exc_info=True)
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
                LOGGER.exception("Failed to clear session %s", self._session_id(chat_id), exc_info=True)
        close = getattr(session, "close", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to close session %s", self._session_id(chat_id), exc_info=True)

    async def aclose(self) -> None:
        for session in list(self._sessions.values()):
            close = getattr(session, "close", None)
            if callable(close):
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception:  # noqa: BLE001
                    LOGGER.exception("Failed to close session %s", session, exc_info=True)
        self._sessions.clear()
        try:
            self._chat_store.close()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to close chat store", exc_info=True)


__all__ = ["AgentRuntime"]
