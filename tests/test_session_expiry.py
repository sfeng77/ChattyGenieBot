import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agent_runtime import AgentRuntime
from app.config import Settings, get_settings


def _make_settings(tmp_path: Path, **overrides) -> Settings:
    # model_copy(update=...) does a direct attribute assignment on an already-
    # validated Settings instance — unlike Settings(**kwargs), it can't be
    # silently shadowed by a real .env value taking source-precedence. This
    # also keeps tests from ever touching the real production DBs.
    updates = dict(
        web_search_enabled=False,
        finance_enabled=False,
        vision_enabled=False,
        asr_enabled=False,
        history_prune_enabled=False,
        sessions_db_path=tmp_path / "sessions.db",
        chat_history_db_path=tmp_path / "history.db",
    )
    updates.update(overrides)
    return get_settings().model_copy(update=updates)


async def _contents(runtime: AgentRuntime, chat_id: int) -> list[str]:
    session = runtime._get_session(chat_id)  # noqa: SLF001
    items = await session.get_items()
    return [str(item.get("content")) for item in items if isinstance(item, dict)]


@pytest.mark.asyncio
async def test_run_expires_session_after_long_idle(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, session_idle_expiry_hours=1.0)
    runtime = AgentRuntime(settings)
    chat_id = 111

    session = runtime._get_session(chat_id)  # noqa: SLF001
    await session.add_items([{"role": "user", "content": "old message"}])
    runtime._record_activity(chat_id)  # noqa: SLF001
    runtime._last_activity[chat_id] = time.time() - 2 * 3600  # noqa: SLF001 (2h idle > 1h threshold)

    fake_result = SimpleNamespace(final_output="ok")
    with patch("app.agent_runtime.Runner.run", new=AsyncMock(return_value=fake_result)):
        await runtime.run_message(chat_id, "hello", log_user=False)

    contents = await _contents(runtime, chat_id)
    assert not any("old message" in c for c in contents)
    assert any("New session started after inactivity" in c for c in contents)

    await runtime.aclose()


@pytest.mark.asyncio
async def test_run_keeps_session_when_activity_recent(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, session_idle_expiry_hours=1.0)
    runtime = AgentRuntime(settings)
    chat_id = 222

    session = runtime._get_session(chat_id)  # noqa: SLF001
    await session.add_items([{"role": "user", "content": "recent message"}])
    runtime._record_activity(chat_id)  # noqa: SLF001 (just now, well under the 1h threshold)

    fake_result = SimpleNamespace(final_output="ok")
    with patch("app.agent_runtime.Runner.run", new=AsyncMock(return_value=fake_result)):
        await runtime.run_message(chat_id, "hello", log_user=False)

    contents = await _contents(runtime, chat_id)
    assert any("recent message" in c for c in contents)
    assert not any("New session started after inactivity" in c for c in contents)

    await runtime.aclose()


@pytest.mark.asyncio
async def test_zero_expiry_hours_disables_feature(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, session_idle_expiry_hours=0)
    runtime = AgentRuntime(settings)
    chat_id = 333

    session = runtime._get_session(chat_id)  # noqa: SLF001
    await session.add_items([{"role": "user", "content": "ancient message"}])
    runtime._record_activity(chat_id)  # noqa: SLF001
    runtime._last_activity[chat_id] = time.time() - 100 * 3600  # noqa: SLF001

    fake_result = SimpleNamespace(final_output="ok")
    with patch("app.agent_runtime.Runner.run", new=AsyncMock(return_value=fake_result)):
        await runtime.run_message(chat_id, "hello", log_user=False)

    contents = await _contents(runtime, chat_id)
    assert any("ancient message" in c for c in contents)

    await runtime.aclose()
