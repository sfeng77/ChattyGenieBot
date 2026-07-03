from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from app.agent_runtime import AgentRuntime, _AUTO_SUMMARY_PREFIX
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
        history_prune_threshold_items=24,
        history_keep_last_items=8,
        sessions_db_path=tmp_path / "sessions.db",
        chat_history_db_path=tmp_path / "history.db",
    )
    updates.update(overrides)
    return get_settings().model_copy(update=updates)


class FakeSession:
    """In-memory stand-in for SQLiteSession's get_items/clear_session/add_items."""

    def __init__(self, items: List[Dict[str, Any]]) -> None:
        self._items = list(items)

    async def get_items(self, limit: int | None = None) -> List[Dict[str, Any]]:
        return list(self._items)

    async def clear_session(self) -> None:
        self._items = []

    async def add_items(self, items: List[Dict[str, Any]]) -> None:
        self._items.extend(items)


def _old_summary(label: str) -> Dict[str, Any]:
    return {"role": "system", "content": f"{_AUTO_SUMMARY_PREFIX}\n{label}"}


def _message(i: int) -> Dict[str, Any]:
    role = "user" if i % 2 == 0 else "assistant"
    return {"role": role, "content": f"msg{i}"}


@pytest.mark.asyncio
async def test_prune_keeps_only_latest_previous_summary(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)  # defaults: threshold=24, keep_last=8
    runtime = AgentRuntime(settings)
    runtime._summarize_transcript = AsyncMock(return_value="SUMMARY")  # noqa: SLF001

    messages = [_message(i) for i in range(30)]
    items = [_old_summary("OLD A")] + messages[:15] + [_old_summary("OLD B")] + messages[15:]
    assert len(items) == 32

    fake_session = FakeSession(items)
    await runtime._maybe_prune_session(fake_session)  # noqa: SLF001

    rewritten = fake_session._items  # noqa: SLF001
    assert len(rewritten) == 10  # 1 old summary + 1 new summary + 8 tail items

    assert rewritten[0]["role"] == "system"
    assert "OLD B" in rewritten[0]["content"]
    assert "OLD A" not in str(rewritten)  # the older auto-summary is gone

    assert rewritten[1]["role"] == "system"
    assert rewritten[1]["content"] == f"{_AUTO_SUMMARY_PREFIX}\nSUMMARY"

    assert rewritten[2:] == messages[22:30]

    await runtime.aclose()


@pytest.mark.asyncio
async def test_prune_noop_below_threshold(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    runtime = AgentRuntime(settings)
    runtime._summarize_transcript = AsyncMock(return_value="SUMMARY")  # noqa: SLF001

    items = [_message(i) for i in range(10)]  # below default threshold of 24
    fake_session = FakeSession(items)
    await runtime._maybe_prune_session(fake_session)  # noqa: SLF001

    assert fake_session._items == items  # noqa: SLF001 (untouched)
    runtime._summarize_transcript.assert_not_called()  # noqa: SLF001

    await runtime.aclose()


@pytest.mark.asyncio
async def test_prune_with_single_previous_summary_keeps_it(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    runtime = AgentRuntime(settings)
    runtime._summarize_transcript = AsyncMock(return_value="SUMMARY")  # noqa: SLF001

    messages = [_message(i) for i in range(30)]
    items = [_old_summary("ONLY OLD")] + messages
    fake_session = FakeSession(items)
    await runtime._maybe_prune_session(fake_session)  # noqa: SLF001

    rewritten = fake_session._items  # noqa: SLF001
    assert "ONLY OLD" in rewritten[0]["content"]
    assert rewritten[1]["content"] == f"{_AUTO_SUMMARY_PREFIX}\nSUMMARY"
    assert rewritten[2:] == messages[-8:]

    await runtime.aclose()
