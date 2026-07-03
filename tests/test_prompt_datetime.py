from datetime import datetime
from zoneinfo import ZoneInfo

from app.prompt import current_datetime_line


def test_current_datetime_line_contains_today_in_configured_timezone() -> None:
    tz_name = "America/Los_Angeles"
    expected_date = datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")

    line = current_datetime_line(tz_name)

    assert expected_date in line
    assert "Current date and time:" in line
    assert "today" in line.lower()


def test_current_datetime_line_respects_timezone_argument() -> None:
    la_line = current_datetime_line("America/Los_Angeles")
    tokyo_line = current_datetime_line("Asia/Tokyo")

    assert la_line != tokyo_line
