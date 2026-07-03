from bench.scorer import score_run


def test_legacy_task_without_content_fields_scores_as_before() -> None:
    task = {
        "id": "legacy_01",
        "expected_tools": ["web_search"],
        "forbidden_tools": ["stock_trend"],
    }
    calls = [{"tool": "web_search", "args": {"query": "test"}}]

    result = score_run(task, calls, "Here is your answer.")

    assert result["content_ok"] is True
    assert result["detail"]["content_violations"] == []
    assert result["passed"] is True


def test_response_must_not_contain_fails_when_present() -> None:
    task = {
        "id": "temporal_stale",
        "expected_tools": ["web_search"],
        "response_must_not_contain": ["3月6", "March 6"],
    }
    calls = [{"tool": "web_search", "args": {}}]

    result = score_run(task, calls, "今天西雅图晴天,3月6日的旧数据已过期。")

    assert result["content_ok"] is False
    assert result["passed"] is False
    assert result["detail"]["content_violations"] == [
        {"type": "forbidden_present", "string": "3月6"}
    ]


def test_response_must_contain_passes_when_satisfied() -> None:
    task = {
        "id": "temporal_today",
        "expected_tools": [],
        "response_must_contain": ["2026-07-02"],
    }

    result = score_run(task, [], "Today is 2026-07-02, a Thursday.")

    assert result["content_ok"] is True
    assert result["passed"] is True


def test_response_must_contain_supports_or_alternatives() -> None:
    task = {
        "id": "temporal_or",
        "expected_tools": [],
        "response_must_contain": [["7月2日", "2026-07-02"]],
    }

    result = score_run(task, [], "今天是7月2日,星期四。")

    assert result["content_ok"] is True
    assert result["passed"] is True


def test_response_must_contain_fails_when_no_alternative_matches() -> None:
    task = {
        "id": "temporal_or_fail",
        "expected_tools": [],
        "response_must_contain": [["7月2日", "2026-07-02"]],
    }

    result = score_run(task, [], "今天天气不错。")

    assert result["content_ok"] is False
    assert result["passed"] is False
    assert result["detail"]["content_violations"] == [
        {"type": "missing", "expected_any_of": ["7月2日", "2026-07-02"]}
    ]


def test_content_check_is_case_insensitive() -> None:
    task = {
        "id": "case_insensitive",
        "expected_tools": [],
        "response_must_not_contain": ["MARCH 6"],
    }

    result = score_run(task, [], "The forecast mentioned march 6 as stale data.")

    assert result["content_ok"] is False
