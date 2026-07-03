from __future__ import annotations
import re

import inspect
import logging
import time
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
from app.features.finance import AlphaVantageClient, YFinanceClient, create_disabled_finance_tool, create_stock_trend_tool
from app.features.memory import create_search_memory_tool
from app.features.reminder import current_chat_id
from app.features.vision import create_disabled_vision_tool, create_vision_tool
from app.features.web_search import WebSearchClient, create_disabled_web_search_tool, create_ollama_web_search_tool
from app.progress import NullProgressDispatcher, ProgressDispatcher, ProgressHooks
from app.prompt import CURRENT_TIME_PLACEHOLDER, current_datetime_line, get_agent_instructions
from app.storage.chat_store import ChatStore

LOGGER = logging.getLogger(__name__)

_AUTO_SUMMARY_PREFIX = "Earlier conversation summary (auto-generated):"


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
        self._chat_store = ChatStore(settings.chat_history_db_path)
        tools: List[object] = []
        self._search_memory_tool = create_search_memory_tool(self._chat_store)
        tools.append(self._search_memory_tool)
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
        static_instructions = get_agent_instructions(web_search_available, finance_available, vision_available)

        def _dynamic_instructions(ctx, agent) -> str:  # noqa: ANN001, ARG001
            time_section = f"# Current time\n- {current_datetime_line(self._settings.agent_timezone)}"
            return static_instructions.replace(CURRENT_TIME_PLACEHOLDER, time_section)

        self._agent = Agent(
            name="Agent Mushroom",
            instructions=_dynamic_instructions,
            model=settings.openai_model,
            model_settings=ModelSettings(
                temperature=settings.openai_temperature,
                extra_body={"think": settings.openai_think_enabled},
            ),
            tools=tools,
        )
        self._sessions: Dict[int, SQLiteSession] = {}
        self._last_activity: Dict[int, float] = {}
        self._ensure_session_activity_table()

    def _ensure_session_activity_table(self) -> None:
        try:
            conn = self._chat_store.get_connection()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_activity (
                        chat_id INTEGER PRIMARY KEY,
                        last_ts REAL NOT NULL
                    )
                    """
                )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to create session_activity table")

    def get_db_connection(self):
        """Return the shared SQLite connection for additional stores."""
        return self._chat_store.get_connection()

    def add_tool(self, tool) -> None:
        """Append a tool to the agent after initial construction."""
        self._agent.tools = list(self._agent.tools) + [tool]

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
        if provider == "yfinance":
            client = YFinanceClient(timeout=self._settings.finance_timeout)
            provider_label = "yfinance"
        elif provider == "alpha_vantage":
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

    def _get_last_activity(self, chat_id: int) -> Optional[float]:
        if chat_id in self._last_activity:
            return self._last_activity[chat_id]
        try:
            conn = self._chat_store.get_connection()
            row = conn.execute(
                "SELECT last_ts FROM session_activity WHERE chat_id = ?", (chat_id,)
            ).fetchone()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to read session activity for chat_id=%s", chat_id, exc_info=True)
            return None
        if row is None:
            return None
        last_ts = float(row[0])
        self._last_activity[chat_id] = last_ts
        return last_ts

    def _record_activity(self, chat_id: int) -> None:
        now_ts = time.time()
        self._last_activity[chat_id] = now_ts
        try:
            conn = self._chat_store.get_connection()
            with conn:
                conn.execute(
                    """
                    INSERT INTO session_activity(chat_id, last_ts) VALUES (?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET last_ts = excluded.last_ts
                    """,
                    (chat_id, now_ts),
                )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to persist session activity for chat_id=%s", chat_id, exc_info=True)

    async def _maybe_expire_session(self, chat_id: int) -> None:
        expiry_hours = self._settings.session_idle_expiry_hours
        if expiry_hours <= 0:
            return
        last_ts = self._get_last_activity(chat_id)
        if last_ts is None:
            return
        idle_hours = (time.time() - last_ts) / 3600.0
        if idle_hours <= expiry_hours:
            return
        LOGGER.info("Session chat-%s expired after %.1fh idle; starting fresh.", chat_id, idle_hours)
        await self.reset(chat_id)
        session = self._get_session(chat_id)
        try:
            await session.add_items(
                [
                    {
                        "role": "system",
                        "content": (
                            "(New session started after inactivity. Use search_memory for anything "
                            "from earlier conversations.)"
                        ),
                    }
                ]
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to seed fresh session for chat_id=%s", chat_id, exc_info=True)

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
        await self._maybe_expire_session(chat_id)
        self._record_activity(chat_id)
        session = self._get_session(chat_id)
        _chat_id_token = current_chat_id.set(chat_id)
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
        finally:
            current_chat_id.reset(_chat_id_token)
        try:
            await self._maybe_trim_tool_outputs(session)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Tool-output trimming failed", exc_info=True)
        output = result.final_output
        if isinstance(output, str):
            response = output.strip()
        elif output is None:
            response = ""
        else:
            response = str(output).strip()
        # Strip <think>...</think> blocks from reasoning models
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
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
        LOGGER.info("Recap fetched %d history messages", len(messages))
        if not messages:
            return ""
        # Use configurable transcript size for recap to include more context
        transcript_limit = int(getattr(self._settings, "history_recap_transcript_chars", max_chars * 4))
        if transcript_limit < 1000:
            transcript_limit = 1000
        transcript = self._messages_to_transcript(messages, transcript_limit)
        LOGGER.info("Recap transcript length: %d chars", len(transcript))
        LOGGER.info("Transcript preview: %s", transcript[:200].replace('\n', ' '))
        summary = await self._summarize_recap(transcript, max_chars=max_chars)
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

    def _is_auto_summary(self, item: Dict[str, Any]) -> bool:
        if not isinstance(item, dict) or item.get("role") != "system":
            return False
        content = item.get("content")
        return isinstance(content, str) and content.startswith(_AUTO_SUMMARY_PREFIX)

    async def _maybe_prune_session(self, session: SQLiteSession) -> None:
        keep_last = max(1, int(self._settings.history_keep_last_items))
        threshold = max(keep_last + 1, int(self._settings.history_prune_threshold_items))
        items = await session.get_items()
        if len(items) <= threshold:
            return
        older = items[:-keep_last]
        tail = items[-keep_last:]

        # Summarizing a previous auto-summary along with the older items would let
        # ancient topics survive indefinitely through summaries-of-summaries; only
        # ever carry forward the single most recent previous summary verbatim, and
        # summarize the rest fresh each time.
        previous_summaries = [item for item in older if self._is_auto_summary(item)]
        non_summary_older = [item for item in older if not self._is_auto_summary(item)]
        discarded = max(0, len(previous_summaries) - 1)
        if discarded:
            LOGGER.info("Pruning: discarding %d stale auto-summary item(s), keeping only the most recent", discarded)
        latest_previous_summary = previous_summaries[-1:]

        transcript = self._items_to_transcript(non_summary_older, max_chars=self._settings.history_summary_max_chars * 4)
        summary = await self._summarize_transcript(transcript, max_chars=self._settings.history_summary_max_chars)
        if not summary:
            summary = self._fallback_summary(transcript, self._settings.history_summary_max_chars)
        summary_item: Dict[str, Any] = {"role": "system", "content": f"{_AUTO_SUMMARY_PREFIX}\n{summary}"}
        try:
            await session.clear_session()
            await session.add_items(latest_previous_summary + [summary_item] + tail)
        except Exception:  # noqa: BLE001
            LOGGER.exception("Failed to rewrite session history", exc_info=True)

    def _trim_tool_output_item(self, item: Any, max_chars: int) -> Optional[Dict[str, Any]]:
        """Return a trimmed copy of a `function_call_output` item if its `output`
        exceeds `max_chars`, else None (not a recognized oversized tool-output item).

        Handles the item shape defensively: skip (return None) anything that
        isn't exactly a dict-shaped function_call_output with a string output,
        rather than risk corrupting an item we don't fully understand.
        """
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            return None
        output = item.get("output")
        if not isinstance(output, str) or len(output) <= max_chars:
            return None
        marker = "...[truncated for history]"
        trimmed_output = output[: max(0, max_chars - len(marker))] + marker
        return {**item, "output": trimmed_output}

    async def _maybe_trim_tool_outputs(self, session: SQLiteSession) -> None:
        """Shrink oversized tool-call results already written to `session` so
        future turns don't keep re-sending heavy payloads (e.g. full web_search
        JSON) as context. The model has already seen the untrimmed result in
        the turn that used it — this only affects what later turns see.
        """
        max_chars = self._settings.session_tool_result_max_chars
        if max_chars <= 0:
            return
        items = await session.get_items()
        rewritten: List[Any] = []
        trimmed_count = 0
        for item in items:
            trimmed = self._trim_tool_output_item(item, max_chars)
            if trimmed is not None:
                rewritten.append(trimmed)
                trimmed_count += 1
            else:
                rewritten.append(item)
        if not trimmed_count:
            return
        LOGGER.info("Trimmed %d oversized tool-output item(s) in session history", trimmed_count)
        await session.clear_session()
        await session.add_items(rewritten)

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

    async def _summarize_core(self, transcript: str, max_chars: int, *, style: str = "generic") -> str:
        if not transcript:
            return ""
        instructions = (
            "You are a concise summarizer. Summarize chat conversations into a factual, neutral summary. "
            "Focus on key questions, decisions, facts, and follow-ups. Prefer bullet points when helpful. "
            "Avoid speculation. Keep within the requested character limit."
        )
        summarizer = Agent(
            name="Summarizer",
            instructions=instructions,
            model=self._settings.openai_model,
            model_settings=ModelSettings(temperature=0.1),
            tools=[],
        )
        if style == "recap":
            body = (
                "Using the following transcript, create a concise recap that highlights:\n"
                "- Main topics or themes\n"
                "- Decisions or conclusions\n"
                "Keep the recap brief and formatted as bullet points when appropriate.\n"
                "Use the language of the original discussions.\n\n"
                "Transcript:\n"
                f"{transcript}\n"
                "Recap:"
            )
        else:
            body = (
                "Conversation:\n"
                f"{transcript}"
            )
        prompt = (
            f"Character limit: {max_chars}.\n"
            "Return only the summary text (no preface).\n\n"
            f"{body}"
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

    async def _summarize_transcript(self, transcript: str, max_chars: int) -> str:
        return await self._summarize_core(transcript, max_chars, style="generic")

    async def _summarize_recap(self, transcript: str, max_chars: int) -> str:
        return await self._summarize_core(transcript, max_chars, style="recap")

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
