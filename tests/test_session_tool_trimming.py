from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.agent_runtime import AgentRuntime
from app.config import Settings, get_settings


def _make_settings(tmp_path: Path, **overrides) -> Settings:
    updates = dict(
        web_search_enabled=False,
        finance_enabled=False,
        vision_enabled=False,
        asr_enabled=False,
        sessions_db_path=tmp_path / "sessions.db",
        chat_history_db_path=tmp_path / "history.db",
    )
    updates.update(overrides)
    return get_settings().model_copy(update=updates)


class FakeSession:
    """In-memory stand-in for SQLiteSession's get_items/clear_session/add_items."""

    def __init__(self, items: List[Dict[str, Any]]) -> None:
        self._items = list(items)
        self.rewrite_calls = 0

    async def get_items(self, limit: int | None = None) -> List[Dict[str, Any]]:
        return list(self._items)

    async def clear_session(self) -> None:
        self.rewrite_calls += 1
        self._items = []

    async def add_items(self, items: List[Dict[str, Any]]) -> None:
        self._items.extend(items)


def _tool_output(call_id: str, output: str) -> Dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _tool_call(call_id: str, name: str = "web_search") -> Dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


def _message(role: str, content: str) -> Dict[str, Any]:
    return {"role": role, "content": content}


@pytest.mark.asyncio
async def test_trims_oversized_tool_output(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, session_tool_result_max_chars=50)
    runtime = AgentRuntime(settings)

    big_output = "x" * 500
    items = [
        _message("user", "search something"),
        _tool_call("call_1"),
        _tool_output("call_1", big_output),
        _message("assistant", "here's what I found"),
    ]
    session = FakeSession(items)
    await runtime._maybe_trim_tool_outputs(session)  # noqa: SLF001

    assert session.rewrite_calls == 1
    rewritten = session._items  # noqa: SLF001
    assert rewritten[0] == items[0]
    assert rewritten[1] == items[1]
    assert rewritten[2]["output"].endswith("...[truncated for history]")
    assert len(rewritten[2]["output"]) == 50
    assert rewritten[2]["call_id"] == "call_1"
    assert rewritten[3] == items[3]

    await runtime.aclose()


@pytest.mark.asyncio
async def test_no_rewrite_when_nothing_exceeds_limit(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, session_tool_result_max_chars=600)
    runtime = AgentRuntime(settings)

    items = [
        _message("user", "search something"),
        _tool_call("call_1"),
        _tool_output("call_1", "a small result"),
        _message("assistant", "here's what I found"),
    ]
    session = FakeSession(items)
    await runtime._maybe_trim_tool_outputs(session)  # noqa: SLF001

    assert session.rewrite_calls == 0
    assert session._items == items  # noqa: SLF001 (byte-identical, untouched)

    await runtime.aclose()


@pytest.mark.asyncio
async def test_zero_max_chars_disables_trimming(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, session_tool_result_max_chars=0)
    runtime = AgentRuntime(settings)

    items = [_tool_output("call_1", "x" * 5000)]
    session = FakeSession(items)
    await runtime._maybe_trim_tool_outputs(session)  # noqa: SLF001

    assert session.rewrite_calls == 0
    assert session._items == items  # noqa: SLF001

    await runtime.aclose()


@pytest.mark.asyncio
async def test_unrecognized_items_pass_through_unchanged(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, session_tool_result_max_chars=10)
    runtime = AgentRuntime(settings)

    weird_item = {"type": "reasoning", "summary": ["x" * 500]}  # no "output" str field
    items = [weird_item, _tool_output("call_1", "y" * 500)]
    session = FakeSession(items)
    await runtime._maybe_trim_tool_outputs(session)  # noqa: SLF001

    rewritten = session._items  # noqa: SLF001
    assert rewritten[0] == weird_item  # passed through untouched, not corrupted
    assert rewritten[1]["output"].endswith("...[truncated for history]")

    await runtime.aclose()
