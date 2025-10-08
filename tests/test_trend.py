import unittest
from datetime import datetime, timedelta

from app.finance_client import DailyBar
from app.services import trend


class TrendTests(unittest.TestCase):
    def setUp(self) -> None:
        base = datetime(2025, 10, 1)
        closes = [100, 101, 102, 104, 103, 105]
        self.bars = [
            DailyBar(
                symbol="TEST",
                date=base + timedelta(days=index),
                open=value - 1,
                high=value + 1,
                low=value - 2,
                close=value,
                volume=1_000 + index,
            )
            for index, value in enumerate(closes)
        ]

    def test_select_window_respects_days(self) -> None:
        window = trend.select_window(self.bars, days=3)
        self.assertGreaterEqual(len(window.bars), 2)
        self.assertTrue(all(bar.date >= window.bars[0].date for bar in window.bars))

    def test_percent_change(self) -> None:
        window = trend.select_window(self.bars, days=5)
        change = trend.percent_change(window)
        self.assertAlmostEqual(change, ((window.bars[-1].close / window.bars[0].close) - 1) * 100)

    def test_sparkline_length(self) -> None:
        window = trend.select_window(self.bars, days=4)
        spark = trend.sparkline(window)
        self.assertEqual(len(spark), len(window.bars))

    def test_volatility_non_negative(self) -> None:
        window = trend.select_window(self.bars, days=5)
        self.assertGreaterEqual(trend.volatility_pct(window), 0.0)


if __name__ == "__main__":
    unittest.main()
