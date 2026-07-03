"""Mock tool layer for benchmarking.

Instead of redefining tools, we wrap the *real* FunctionTool objects from the
production agent: the name / description / JSON schema the model sees are
identical to production (those are part of the system under test), but the
invocation body is replaced with one that records the arguments and returns a
canned response.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional

from agents import function_tool
from agents.tool import FunctionTool

# ---------------------------------------------------------------------------
# Call recording
# ---------------------------------------------------------------------------


class CallRecorder:
    """Collects every tool invocation made during a single run."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def record(self, tool_name: str, args: Dict[str, Any]) -> None:
        self.calls.append({"tool": tool_name, "args": args})

    def reset(self) -> None:
        self.calls.clear()

    def tools_called(self) -> List[str]:
        return [c["tool"] for c in self.calls]

    def first_call(self, tool_name: str) -> Optional[Dict[str, Any]]:
        for c in self.calls:
            if c["tool"] == tool_name:
                return c
        return None


# ---------------------------------------------------------------------------
# Default canned responses (used when a task doesn't override them)
# ---------------------------------------------------------------------------

DEFAULT_RESPONSES: Dict[str, Any] = {
    "web_search": {
        "query": "<echo>",
        "results": [
            {
                "title": "Mock search result",
                "url": "https://example.com/mock",
                "snippet": "This is a mock snippet returned by the benchmark harness.",
            }
        ],
        "count": 1,
    },
    "stock_trend": {
        "symbol": "<echo>",
        "provider": "mock",
        "window_days": 30,
        "summary": "Mock trend: +3.2% over the window, low volatility.",
        "last_close": 123.45,
    },
    "vision_analyze": {
        "summary": "Mock vision analysis: a photo of a cat on a desk.",
    },
    "search_memory": {
        "query": "<echo>",
        "results": [],
        "count": 0,
    },
    "set_reminder": {
        "success": True,
        "scheduled_for": "2026-01-01T09:00:00+00:00",
        "message": "<echo>",
    },
}


def _resolve_response(tool_name: str, args: Dict[str, Any], mock_config: Dict[str, Any]) -> Any:
    """Pick the canned response for this tool: task override > default."""
    cfg = (mock_config or {}).get(tool_name, {})
    if "error" in cfg:
        return {"error": cfg["error"]}
    response = cfg.get("response", DEFAULT_RESPONSES.get(tool_name, {"ok": True}))
    # Shallow echo substitution so responses look plausible.
    if isinstance(response, dict):
        response = {
            k: (next(iter(args.values()), "") if v == "<echo>" else v)
            for k, v in response.items()
        }
    return response


# ---------------------------------------------------------------------------
# Wrapping
# ---------------------------------------------------------------------------


def wrap_tool(tool: FunctionTool, recorder: CallRecorder, mock_config: Dict[str, Any]) -> FunctionTool:
    """Clone a real FunctionTool, replacing only its invocation body."""

    async def fake_invoke(ctx: Any, args_json: str) -> Any:  # noqa: ANN401
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError:
            args = {"_raw": args_json}
        recorder.record(tool.name, args)
        return json.dumps(_resolve_response(tool.name, args, mock_config), ensure_ascii=False)

    return dataclasses.replace(tool, on_invoke_tool=fake_invoke)


def make_mock_reminder_tool(recorder: CallRecorder, mock_config: Dict[str, Any]) -> FunctionTool:
    """set_reminder lives in bot.py (needs the Telegram JobQueue), so a bare
    AgentRuntime doesn't have it. Recreate its schema here so it can still be
    benchmarked. Descriptions mirror app/features/reminder/tool.py."""

    @function_tool(name_override="set_reminder")
    async def set_reminder(time_expression: str, message: str) -> str:
        """Schedule a Telegram reminder for the current chat at the specified time.

        Args:
            time_expression: Natural language time for the reminder, e.g.
                '明天早上9点', 'in 2 hours', 'next Monday at 3pm'.
            message: Reminder content to send, e.g. '开会', 'take medicine'.
        """
        args = {"time_expression": time_expression, "message": message}
        recorder.record("set_reminder", args)
        return json.dumps(_resolve_response("set_reminder", args, mock_config), ensure_ascii=False)

    return set_reminder


def build_benchmark_tools(
    real_tools: List[Any],
    recorder: CallRecorder,
    mock_config: Dict[str, Any],
) -> List[Any]:
    """Wrap all real FunctionTools and append the mock reminder tool."""
    wrapped: List[Any] = []
    seen_names = set()
    for tool in real_tools:
        if isinstance(tool, FunctionTool):
            wrapped.append(wrap_tool(tool, recorder, mock_config))
            seen_names.add(tool.name)
        else:
            wrapped.append(tool)
    if "set_reminder" not in seen_names:
        wrapped.append(make_mock_reminder_tool(recorder, mock_config))
    return wrapped
