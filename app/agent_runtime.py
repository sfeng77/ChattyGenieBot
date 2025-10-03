from __future__ import annotations

import inspect
import logging
from typing import Dict

from agents import Agent, Runner, SQLiteSession, set_default_openai_api, set_default_openai_client, set_default_openai_key, set_tracing_disabled
from agents import ModelSettings
from openai import AsyncOpenAI

from app.config import Settings
from app.prompt import AGENT_INSTRUCTIONS

LOGGER = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._session_db_path = settings.sessions_db_path
        set_default_openai_api("chat_completions")
        set_default_openai_key(settings.openai_api_key)
        set_tracing_disabled(not settings.openai_tracing_enabled)
        if settings.openai_api_base:
            client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_api_base,
            )
            set_default_openai_client(client)
        self._agent = Agent(
            name="Chatty Genie",
            instructions=AGENT_INSTRUCTIONS,
            model=settings.openai_model,
            model_settings=ModelSettings(temperature=settings.openai_temperature),
        )
        self._sessions: Dict[int, SQLiteSession] = {}

    def _session_id(self, chat_id: int) -> str:
        return f"chat-{chat_id}"

    def _get_session(self, chat_id: int) -> SQLiteSession:
        session = self._sessions.get(chat_id)
        if session is None:
            session = SQLiteSession(self._session_id(chat_id), str(self._session_db_path))
            self._sessions[chat_id] = session
        return session

    async def run_message(self, chat_id: int, user_message: str) -> str:
        session = self._get_session(chat_id)
        result = await Runner.run(self._agent, user_message, session=session)
        output = result.final_output
        if isinstance(output, str):
            return output.strip()
        if output is None:
            return ""
        return str(output).strip()

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
