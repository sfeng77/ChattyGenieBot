from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from bench.runner import compute_date_placeholders, render_task_templates

TZ = "America/Los_Angeles"


def test_compute_date_placeholders_shapes() -> None:
    mapping = compute_date_placeholders(TZ)

    today = datetime.now(ZoneInfo(TZ)).date()
    assert mapping["TODAY"] == today.strftime("%Y-%m-%d")
    assert mapping["TODAY_CN"] == f"{today.month}月{today.day}日"

    next_wed = datetime.strptime(mapping["NEXT_WEDNESDAY"], "%Y-%m-%d").date()
    assert next_wed.weekday() == 2  # Wednesday
    assert next_wed > today
    # Must fall in the calendar week *after* the current one, not just the
    # nearest upcoming Wednesday (which "下周三" would not mean if today is
    # e.g. a Monday).
    this_monday = today - timedelta(days=today.weekday())
    assert next_wed >= this_monday + timedelta(days=7)


def test_render_task_templates_deep_replaces_nested_fields() -> None:
    mapping = compute_date_placeholders(TZ)
    tasks = [
        {
            "id": "t1",
            "input": "今天几号?",
            "response_must_contain": [["{{TODAY_CN}}", "{{TODAY}}"]],
            "mock_responses": {"web_search": {"response": {"snippet": "{{NEXT_WEDNESDAY}}"}}},
        }
    ]

    rendered = render_task_templates(tasks, TZ)

    assert rendered[0]["response_must_contain"] == [[mapping["TODAY_CN"], mapping["TODAY"]]]
    assert rendered[0]["mock_responses"]["web_search"]["response"]["snippet"] == mapping["NEXT_WEDNESDAY"]
    # Fields without placeholders are untouched.
    assert rendered[0]["id"] == "t1"


def test_render_task_templates_leaves_unknown_placeholders_untouched() -> None:
    tasks = [{"id": "t1", "note": "{{NOT_A_REAL_PLACEHOLDER}}"}]

    rendered = render_task_templates(tasks, TZ)

    assert rendered[0]["note"] == "{{NOT_A_REAL_PLACEHOLDER}}"
