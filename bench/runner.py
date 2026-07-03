"""Benchmark runner.

Usage (from repo root, with your normal .env in place):

    python -m bench.runner --tasks bench/tasks.json --repeats 5
    python -m bench.runner --tasks bench/tasks.json --repeats 5 --model gpt-4o-mini
    python -m bench.runner --tasks bench/tasks.json --filter tool_selection

Reuses your real .env / Settings (same model, same tool-availability flags,
therefore the same system prompt as production) but redirects both SQLite
databases to a temp directory and disables history pruning, so benchmark runs
never touch real chat data and never trigger the summarizer.

Each (task, repeat) uses a fresh chat_id => clean session, no cross-task
contamination.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

from app.agent_runtime import AgentRuntime
from app.config import get_settings

from bench.mock_tools import CallRecorder, build_benchmark_tools
from bench.scorer import score_run

BASE_CHAT_ID = 990_000_000  # far away from any real Telegram chat id

_PLACEHOLDER_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def _next_wednesday(today: date) -> date:
    """Wednesday of the calendar week following `today`'s week (i.e. "下周三")."""
    this_monday = today - timedelta(days=today.weekday())
    next_monday = this_monday + timedelta(days=7)
    return next_monday + timedelta(days=2)


def compute_date_placeholders(tz_name: str) -> Dict[str, str]:
    today = datetime.now(ZoneInfo(tz_name)).date()
    next_wed = _next_wednesday(today)
    return {
        "TODAY": today.strftime("%Y-%m-%d"),
        "TODAY_CN": f"{today.month}月{today.day}日",
        "NEXT_WEDNESDAY": next_wed.strftime("%Y-%m-%d"),
        "NEXT_WEDNESDAY_CN": f"{next_wed.month}月{next_wed.day}日",
    }


def _substitute_placeholders(value: Any, mapping: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return _PLACEHOLDER_PATTERN.sub(lambda m: mapping.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [_substitute_placeholders(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _substitute_placeholders(item, mapping) for key, item in value.items()}
    return value


def render_task_templates(tasks: List[Dict[str, Any]], tz_name: str) -> List[Dict[str, Any]]:
    """Deep-replace {{TODAY}} / {{TODAY_CN}} / {{NEXT_WEDNESDAY}} (+ _CN variant)
    placeholders in every string field of each task, computed in `tz_name`."""
    mapping = compute_date_placeholders(tz_name)
    return [_substitute_placeholders(task, mapping) for task in tasks]


def _seed_items_from_turns(turns: List[List[str]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for user_text, assistant_text in turns:
        items.append({"role": "user", "content": user_text})
        items.append({"role": "assistant", "content": assistant_text})
    return items


def _seed_items_from_generate(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build filler turns locally from simple templates — no model calls."""
    topics = spec.get("topics", [])
    pairs_per_topic = int(spec.get("pairs_per_topic", 0))
    items: List[Dict[str, Any]] = []
    for topic in topics:
        for i in range(1, pairs_per_topic + 1):
            items.append({"role": "user", "content": f"关于{topic}的问题 #{i}: 你怎么看?"})
            items.append(
                {
                    "role": "assistant",
                    "content": f"关于{topic},这是第{i}条填充回复,仅用于撑大对话上下文,内容本身无实际意义。",
                }
            )
    return items


def build_seed_items(seed_history: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build session items for a task's `seed_history` field (Form A: `turns`,
    Form B: `generate`)."""
    if "turns" in seed_history:
        return _seed_items_from_turns(seed_history["turns"])
    if "generate" in seed_history:
        return _seed_items_from_generate(seed_history["generate"])
    return []


def build_runtime(tmp_dir: Path, model_override: str | None, think_override: bool | None = None) -> AgentRuntime:
    base = get_settings()
    updates: Dict[str, Any] = {
        "sessions_db_path": tmp_dir / "sessions.db",
        "chat_history_db_path": tmp_dir / "chat_history.db",
        "history_prune_enabled": False,
    }
    if model_override:
        updates["openai_model"] = model_override
    if think_override is not None:
        updates["openai_think_enabled"] = think_override
    settings = base.model_copy(update=updates)
    return AgentRuntime(settings)


async def run_benchmark(
    tasks: List[Dict[str, Any]],
    repeats: int,
    model_override: str | None,
    out_path: Path,
    think_override: bool | None = None,
) -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="agent-bench-"))
    runtime = build_runtime(tmp_dir, model_override, think_override)
    real_tools = list(runtime._agent.tools)  # noqa: SLF001 (see README note)

    recorder = CallRecorder()
    records: List[Dict[str, Any]] = []
    chat_counter = 0

    for task in tasks:
        # Tools are wrapped per task because canned responses differ per task.
        runtime._agent.tools = build_benchmark_tools(  # noqa: SLF001
            real_tools, recorder, task.get("mock_responses", {})
        )
        for rep in range(repeats):
            chat_counter += 1
            chat_id = BASE_CHAT_ID + chat_counter
            recorder.reset()
            seed_history = task.get("seed_history")
            if seed_history:
                seed_items = build_seed_items(seed_history)
                if seed_items:
                    session = runtime._get_session(chat_id)  # noqa: SLF001 (see README note)
                    await session.add_items(seed_items)
            started = time.monotonic()
            error: str | None = None
            response = ""
            try:
                response = await runtime.run_message(chat_id, task["input"])
            except Exception as exc:  # noqa: BLE001
                error = f"{exc.__class__.__name__}: {exc}"
            elapsed = round(time.monotonic() - started, 2)

            score = score_run(task, list(recorder.calls), response)
            if error:
                score["passed"] = False
                score["responded"] = False
            records.append(
                {
                    "task_id": task["id"],
                    "category": task.get("category", "uncategorized"),
                    "repeat": rep,
                    "input": task["input"],
                    "response": response,
                    "error": error,
                    "elapsed_seconds": elapsed,
                    **score,
                }
            )
            mark = "PASS" if score["passed"] else "FAIL"
            print(f"[{mark}] {task['id']} rep {rep + 1}/{repeats} "
                  f"({elapsed}s, tools={score['detail']['called']})")

    resolved_model = runtime._settings.openai_model  # noqa: SLF001 (see README note)
    resolved_think = runtime._settings.openai_think_enabled  # noqa: SLF001 (see README note)
    await runtime.aclose()
    report = summarize(records)
    report["model"] = resolved_model
    report["think_enabled"] = resolved_think
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": report, "runs": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_report(report)
    print(f"\nFull results written to {out_path}")


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)
        by_task[r["task_id"]].append(r)

    def rate(items: List[Dict[str, Any]], key: str) -> float:
        return round(sum(1 for i in items if i[key]) / len(items), 3) if items else 0.0

    mean_elapsed = (
        round(sum(r["elapsed_seconds"] for r in records) / len(records), 2) if records else 0.0
    )

    return {
        "total_runs": len(records),
        "overall": {
            "pass_rate": rate(records, "passed"),
            "tool_selection_acc": rate(records, "tool_selection_ok"),
            "param_acc": rate(records, "params_ok"),
            "no_extra_call_rate": rate(records, "no_extra_calls"),
            "response_rate": rate(records, "responded"),
            "content_rate": rate(records, "content_ok"),
            "mean_elapsed_seconds": mean_elapsed,
        },
        "by_category": {
            cat: {"runs": len(items), "pass_rate": rate(items, "passed"),
                  "tool_selection_acc": rate(items, "tool_selection_ok"),
                  "param_acc": rate(items, "params_ok"),
                  "content_rate": rate(items, "content_ok")}
            for cat, items in sorted(by_category.items())
        },
        "flaky_tasks": sorted(
            tid for tid, items in by_task.items()
            if 0 < sum(1 for i in items if i["passed"]) < len(items)
        ),
        "always_failing_tasks": sorted(
            tid for tid, items in by_task.items()
            if not any(i["passed"] for i in items)
        ),
    }


def print_report(report: Dict[str, Any]) -> None:
    think_label = "on" if report.get("think_enabled") else "off"
    print("\n" + "=" * 60)
    print(f"AGENT BENCHMARK REPORT  (model: {report.get('model', 'unknown')}, think: {think_label})")
    print("=" * 60)
    o = report["overall"]
    print(f"Runs: {report['total_runs']}   Pass rate: {o['pass_rate']:.0%}   "
          f"mean elapsed: {o['mean_elapsed_seconds']}s")
    print(f"  tool selection: {o['tool_selection_acc']:.0%}   params: {o['param_acc']:.0%}   "
          f"no-extra-calls: {o['no_extra_call_rate']:.0%}   responded: {o['response_rate']:.0%}   "
          f"content: {o['content_rate']:.0%}")
    print("-" * 60)
    for cat, stats in report["by_category"].items():
        print(f"{cat:<18} pass {stats['pass_rate']:>4.0%}   "
              f"select {stats['tool_selection_acc']:>4.0%}   "
              f"params {stats['param_acc']:>4.0%}   "
              f"content {stats['content_rate']:>4.0%}   ({stats['runs']} runs)")
    if report["flaky_tasks"]:
        print(f"\nFlaky tasks (sometimes pass): {', '.join(report['flaky_tasks'])}")
    if report["always_failing_tasks"]:
        print(f"Always failing: {', '.join(report['always_failing_tasks'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent benchmark.")
    parser.add_argument("--tasks", default="bench/tasks.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--model", default=None,
                        help="Override OPENAI_MODEL, e.g. a strong cloud model for A/B comparison")
    parser.add_argument("--think", choices=["on", "off"], default=None,
                        help="Override OPENAI_THINK_ENABLED (gpt-oss reasoning) for this run, e.g. --think on")
    parser.add_argument("--filter", default=None,
                        help="Only run tasks whose category or id contains this substring")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    tasks = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    if args.filter:
        tasks = [t for t in tasks
                 if args.filter in t.get("category", "") or args.filter in t["id"]]
    if not tasks:
        print("No tasks matched.", file=sys.stderr)
        sys.exit(1)
    tasks = render_task_templates(tasks, get_settings().agent_timezone)

    think_override = None if args.think is None else (args.think == "on")
    out = Path(args.out) if args.out else Path(
        f"bench/results/run-{time.strftime('%Y%m%d-%H%M%S')}"
        f"{'-' + args.model.replace('/', '_').replace(':', '_') if args.model else ''}"
        f"{'-think_' + args.think if args.think else ''}.json"
    )
    asyncio.run(run_benchmark(tasks, args.repeats, args.model, out, think_override))


if __name__ == "__main__":
    main()
