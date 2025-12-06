from __future__ import annotations

import inspect
import json
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
from app.storage.style_profile_store import StyleProfileStore
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
            name="Agent Mushroom",
            instructions=instructions,
            model=settings.openai_model,
            model_settings=ModelSettings(temperature=settings.openai_temperature),
            tools=tools,
        )
        self._sessions: Dict[int, SQLiteSession] = {}
        self._chat_store = ChatStore(settings.chat_history_db_path)
        self._style_store = StyleProfileStore(self._chat_store.get_connection())

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

    def list_chat_senders(self, chat_id: int) -> List[str]:
        """Return distinct sender_ids for user messages in a chat's history."""
        history_id = self._history_id(chat_id)
        try:
            return self._chat_store.list_senders(external_conversation_id=history_id)
        except Exception:  # noqa: BLE001
            LOGGER.exception("List chat senders failed", exc_info=True)
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

    async def learn_user_style(
        self,
        chat_id: int,
        sender_id: str,
        *,
        label: Optional[str] = None,
        max_messages: Optional[int] = None,
        min_messages: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        effective_max_messages = max_messages or int(self._settings.style_learn_max_messages)
        effective_min_messages = min_messages or int(self._settings.style_learn_min_messages)
        effective_max_chars = max_chars or int(self._settings.style_learn_max_chars)
        history_limit = max(effective_max_messages * 4, effective_max_messages + 20)
        messages = self.get_history_messages(chat_id, limit=history_limit)
        filtered = []
        sender_id_str = str(sender_id)
        for message in messages:
            role = message.get("role") or "user"
            sid = message.get("sender_id")
            if role != "user":
                continue
            if sid is None:
                continue
            raw_sid = str(sid)
            if raw_sid == "assistant":
                continue
            # Normalize historical formats: "user{telegram_id}" or raw id as string
            if raw_sid.startswith("user") and len(raw_sid) > 4 and raw_sid[4:].isdigit():
                raw_sid = raw_sid[4:]
            if raw_sid != sender_id_str:
                continue
            content = (message.get("content") or "").strip()
            if not content:
                continue
            filtered.append(message)
        if not filtered:
            raise ValueError("No user messages found to learn from.")
        if len(filtered) < effective_min_messages:
            raise ValueError(f"Not enough messages to learn style (found {len(filtered)}, need at least {effective_min_messages}).")
        if len(filtered) > effective_max_messages:
            filtered = filtered[-effective_max_messages:]
        # Input layer: prefer messages that carry reasoning or explanation
        reasoning_keywords = ("因为", "所以", "但是", "如果", "其实", "我觉得", "我感觉", "我一般会", "I think", "because", "however", "but", "if ")
        scored_messages: list[tuple[int, Dict[str, Any]]] = []
        for message in filtered:
            text = (message.get("content") or "").strip()
            if not text:
                continue
            length_score = 1 if len(text) >= 15 else 0
            reasoning_score = 1 if any(k in text for k in reasoning_keywords) else 0
            score = length_score + reasoning_score
            scored_messages.append((score, message))
        # Sort by score (high first), then by original order
        scored_messages.sort(key=lambda pair: pair[0], reverse=True)
        # Keep top N but fall back to all if everything scored the same
        top_messages: list[Dict[str, Any]] = [m for score, m in scored_messages if score > 0]
        if not top_messages:
            top_messages = [m for _, m in scored_messages]
        if len(top_messages) > effective_max_messages:
            top_messages = top_messages[:effective_max_messages]
        items: List[Dict[str, Any]] = []
        sample_messages: List[str] = []
        for message in top_messages:
            text = (message.get("content") or "").strip()
            if not text:
                continue
            items.append({"role": "user", "content": text})
            sample_messages.append(text)
        transcript = self._items_to_transcript(items, max_chars=effective_max_chars)
        if not transcript.strip():
            raise ValueError("User transcript is empty after preprocessing.")
        instructions = (
            "You analyze chat messages from a single person and extract both their writing style and behavioral patterns. "
            "Given example messages, identify tone, formality, languages, habits, subject-matter expertise, and how they tend to handle conflict, decisions, risk, and cooperation. "
            "Then construct a compact style guide for an AI assistant.\n"
            "Always respond with a single JSON object and nothing else."
        )
        style_agent = Agent(
            name="Style Learner",
            instructions=instructions,
            model=self._settings.openai_model,
            model_settings=ModelSettings(temperature=self._settings.openai_temperature),
            tools=[],
        )
        schema_hint = (
            "Return a JSON object with the following structure:\n"
            "{\n"
            '  "style_prompt": "SYSTEM PROMPT FOR AN AI ASSISTANT...",\n'
            '  "analysis": {\n'
            '    "tone": "short description of overall tone",\n'
            '    "formality": "informal / neutral / formal",\n'
            '    "languages": ["en", "zh", "..."],\n'
            '    "knowledge_domains": ["workplace", "relationships", "learning", "..."],\n'
            '    "habits": ["common turns of phrase, 习惯用语, punctuation quirks"],\n'
            '    "emoji_usage": "description of emoji usage",\n'
            '    "sentence_style": "short / long / bullet-heavy / etc.",\n'
            '    "behavior_profile": {\n'
            '      "conflict_style": {"label": "confrontational / smooth_cooperative / avoidant / unknown", "score": 1, "evidence": ["..."]},\n'
            '      "decision_style": {"label": "analysis_first / action_first / gut_feeling / unknown", "score": 4, "evidence": ["..."]},\n'
            '      "risk_tolerance": {"score": 3, "evidence": ["..."]},\n'
            '      "reasoning_pattern": {"label": "top_down / bottom_up / stream_of_consciousness / unknown", "evidence": ["..."]},\n'
            '      "depth_preference": {"label": "high_level / detailed_with_examples / mixed / unknown", "evidence": ["..."]},\n'
            '      "empathetic_style": {"label": "high / medium / low / unknown", "evidence": ["..."]},\n'
            '      "disagreement_style": {"label": "direct_challenge / softening_corrections / avoidant / unknown", "evidence": ["..."]},\n'
            '      "core_themes": ["pragmatic", "product_thinking", "learning_methods", "..."],\n'
            '      "communication_style": {"label": "direct / diplomatic / teasing / indirect / unknown", "score": 5, "evidence": ["..."]},\n'
            '      "cooperation_style": {"label": "persuasive_collaborator / lone_fighter / consensus_builder / unknown", "score": 3, "evidence": ["..."]},\n'
            '      "emotional_heat": {"score": 3, "evidence": ["..."]}\n'
            '    },\n'
            '    "personality": {\n'
            '      "mbti": "e.g. INTP, ESFJ, or unknown",\n'
            '      "mbti_confidence": 0.0,\n'
            '      "mbti_rationale": "brief explanation or empty string if unknown"\n'
            '    },\n'
            '    "other_notes": "any other relevant traits"\n'
            "  }\n"
            "}\n"
            'The "style_prompt" must be written as instructions to an AI assistant about how to respond in this user\'s style. '
            "Do not include personal identifiers or concrete private details; focus on style, tone, and behavioral tendencies. "
            "If there is not enough information for a dimension, use neutral defaults (e.g. label=\"unknown\", score=0)."
            "Do not simply copy these example scores. Infer a score between 0 and 5 for each dimension based on the messages; use 0 only when there is not enough information."
        )

        prompt = (
            "You are given example chat messages written by a single user. "
            "Study how they write and produce a style and behavior guide.\n\n"
            "Example messages (in chronological order):\n"
            f"{transcript}\n\n"
            f"{schema_hint}\n\n"
            "Return only the JSON object."
        )
        try:
            result = await Runner.run(style_agent, prompt, session=None, max_turns=1)
            output = result.final_output
        except Exception as exc:
            LOGGER.exception("Style learning model call failed", exc_info=True)
            raise RuntimeError(f"Style learning failed: {exc}") from exc
        if isinstance(output, str):
            text = output.strip()
        elif output is None:
            text = ""
        else:
            text = str(output).strip()
        if not text:
            raise RuntimeError("Style learning returned an empty response.")
        try:
            payload = json.loads(text)
        except Exception as exc:
            LOGGER.warning("Failed to parse style learner JSON, falling back to generic prompt: %s", exc)
            style_prompt = (
                "When responding, mimic this user's chat style based on the provided examples. "
                "Use their usual tone, level of formality, preferred languages (including any EN/中文 mixing), "
                "and common turns of phrase, but do not reveal private details or pretend to actually be them."
            )
            analysis: Dict[str, Any] = {
                "raw_response": text,
            }
        else:
            if not isinstance(payload, dict):
                raise RuntimeError("Style learning response was not a JSON object.")
            style_prompt_val = payload.get("style_prompt")
            analysis_val = payload.get("analysis")
            style_prompt = (style_prompt_val or "").strip() if isinstance(style_prompt_val, str) else ""
            if not style_prompt:
                style_prompt = (
                    "When responding, mimic this user's chat style based on the provided examples. "
                    "Use their usual tone, level of formality, preferred languages (including any EN/中文 mixing), "
                    "and common turns of phrase, but do not reveal private details or pretend to actually be them."
                )
            analysis = analysis_val if isinstance(analysis_val, dict) else {}
        resolved_label = label or f"user-{sender_id_str}"
        LOGGER.info(
            "Learned style profile for chat_id=%s sender_id=%s label=%s (messages_used=%s)",
            chat_id,
            sender_id_str,
            resolved_label,
            len(sample_messages),
        )
        profile = self._style_store.upsert_profile(
            chat_id=chat_id,
            sender_id=sender_id_str,
            label=resolved_label,
            style_prompt=style_prompt,
            analysis=analysis,
            sample_messages=sample_messages[:5],
            message_count=len(sample_messages),
        )
        return profile

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
