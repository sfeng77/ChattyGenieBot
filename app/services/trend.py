from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Sequence

from app.finance_client import DailyBar

_SPARKLINE_CHARS = ".,-~=*#@"


@dataclass(slots=True)
class TrendWindow:
    bars: List[DailyBar]

    @property
    def has_enough_data(self) -> bool:
        return len(self.bars) >= 2


def select_window(bars: Sequence[DailyBar], days: int) -> TrendWindow:
    if not bars:
        return TrendWindow(bars=[])
    ordered = sorted(bars, key=lambda bar: bar.date)
    end_date = ordered[-1].date
    cutoff = end_date - timedelta(days=max(1, days))
    window = [bar for bar in ordered if bar.date >= cutoff]
    if len(window) < 2 and len(ordered) >= 2:
        window = ordered[-2:]
    return TrendWindow(bars=window)


def percent_change(window: TrendWindow) -> float:
    if not window.has_enough_data:
        return 0.0
    start = window.bars[0].close
    end = window.bars[-1].close
    if start == 0:
        return 0.0
    return (end / start - 1.0) * 100.0


def slope(window: TrendWindow) -> float:
    if not window.has_enough_data:
        return 0.0
    closes = [bar.close for bar in window.bars]
    n = len(closes)
    x_values = list(range(n))
    x_mean = sum(x_values) / n
    y_mean = sum(closes) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, closes))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def volatility_pct(window: TrendWindow) -> float:
    if len(window.bars) < 3:
        return 0.0
    returns = []
    for prev, curr in zip(window.bars, window.bars[1:]):
        if prev.close == 0:
            continue
        returns.append(math.log(curr.close / prev.close))
    if len(returns) < 2:
        return 0.0
    return statistics.stdev(returns) * 100.0


def high_low(window: TrendWindow) -> tuple[float, float]:
    if not window.bars:
        return (0.0, 0.0)
    highs = [bar.high for bar in window.bars]
    lows = [bar.low for bar in window.bars]
    return (max(highs), min(lows))


def sparkline(window: TrendWindow) -> str:
    if not window.bars:
        return ""
    closes = [bar.close for bar in window.bars]
    if len(set(closes)) == 1:
        mid_index = len(_SPARKLINE_CHARS) // 2
        return _SPARKLINE_CHARS[mid_index] * len(closes)
    min_close = min(closes)
    max_close = max(closes)
    span = max_close - min_close
    if span == 0:
        mid_index = len(_SPARKLINE_CHARS) // 2
        return _SPARKLINE_CHARS[mid_index] * len(closes)
    chars = []
    for value in closes:
        normalized = (value - min_close) / span
        index = round(normalized * (len(_SPARKLINE_CHARS) - 1))
        chars.append(_SPARKLINE_CHARS[index])
    return "".join(chars)


def compact_series(window: TrendWindow) -> List[dict[str, str | float]]:
    return [{"date": bar.date.strftime("%Y-%m-%d"), "close": round(bar.close, 4)} for bar in window.bars]


__all__ = [
    "TrendWindow",
    "select_window",
    "percent_change",
    "slope",
    "volatility_pct",
    "high_low",
    "sparkline",
    "compact_series",
]
