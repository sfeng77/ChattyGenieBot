"""Scoring logic for benchmark runs.

Each run of a task is scored on independent axes so the report can localize
failures:
  - tool_selection : all expected tools called, no forbidden tools called
  - params         : expected_params are a (case-insensitive) subset of the
                     args of the first call to that tool
  - no_extra_calls : no tools outside expected + allowed were called
  - responded      : agent produced a non-empty final answer
  - content        : response_must_contain / response_must_not_contain hold
A run "passes" overall only if every applicable axis passes.
"""
from __future__ import annotations

from typing import Any, Dict, List


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _params_match(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    """Check expected is a subset of actual. Values may be a list of accepted
    alternatives. Returns {ok, mismatches}."""
    mismatches = []
    for key, want in expected.items():
        got = actual.get(key)
        accepted = want if isinstance(want, list) else [want]
        if _norm(got) not in [_norm(a) for a in accepted]:
            mismatches.append({"param": key, "expected": want, "actual": got})
    return {"ok": not mismatches, "mismatches": mismatches}


def _content_check(task: Dict[str, Any], response: str) -> Dict[str, Any]:
    """Check response_must_contain / response_must_not_contain (case-insensitive).

    Each element of response_must_contain may itself be a list of alternatives
    (OR semantics); the outer list is AND semantics. response_must_not_contain
    is a flat list; none of its strings may appear.
    """
    response_lower = (response or "").lower()
    violations: List[Dict[str, Any]] = []

    for item in task.get("response_must_contain", []):
        alternatives = item if isinstance(item, list) else [item]
        if not any(alt.lower() in response_lower for alt in alternatives):
            violations.append({"type": "missing", "expected_any_of": alternatives})

    for forbidden in task.get("response_must_not_contain", []):
        if forbidden.lower() in response_lower:
            violations.append({"type": "forbidden_present", "string": forbidden})

    return {"ok": not violations, "violations": violations}


def score_run(task: Dict[str, Any], calls: List[Dict[str, Any]], response: str) -> Dict[str, Any]:
    expected = task.get("expected_tools", [])
    forbidden = set(task.get("forbidden_tools", []))
    # search_memory is a benign reflex for a memory-enabled bot; allow it by
    # default unless a task explicitly forbids it.
    allowed = set(task.get("allowed_tools", ["search_memory"]))

    called = [c["tool"] for c in calls]
    called_set = set(called)

    missing = [t for t in expected if t not in called_set]
    forbidden_hit = sorted(called_set & forbidden)
    extra = sorted(called_set - set(expected) - allowed - forbidden)

    tool_selection_ok = not missing and not forbidden_hit

    param_results: Dict[str, Any] = {}
    params_ok = True
    for tool_name, expected_params in task.get("expected_params", {}).items():
        first = next((c for c in calls if c["tool"] == tool_name), None)
        if first is None:
            param_results[tool_name] = {"ok": False, "mismatches": [{"param": "*", "expected": "tool not called", "actual": None}]}
            params_ok = False
            continue
        result = _params_match(expected_params, first["args"])
        param_results[tool_name] = result
        params_ok = params_ok and result["ok"]

    responded = bool(response and response.strip())

    content_result = _content_check(task, response)
    content_ok = content_result["ok"]

    passed = tool_selection_ok and params_ok and not extra and responded and content_ok

    return {
        "passed": passed,
        "tool_selection_ok": tool_selection_ok,
        "params_ok": params_ok,
        "no_extra_calls": not extra,
        "responded": responded,
        "content_ok": content_ok,
        "detail": {
            "called": called,
            "missing_tools": missing,
            "forbidden_called": forbidden_hit,
            "extra_called": extra,
            "param_results": param_results,
            "num_tool_calls": len(calls),
            "content_violations": content_result["violations"],
        },
    }
