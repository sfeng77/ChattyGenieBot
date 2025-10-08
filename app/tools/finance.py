from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Annotated, Dict, List, Optional

from agents import function_tool
from pydantic import Field

from app.finance_client import FinanceClient
from app.services import trend

LOGGER = logging.getLogger(__name__)

SymbolArg = Annotated[
    str,
    Field(
        description="Ticker symbol (e.g., AAPL)",
        min_length=1,
        max_length=12,
        pattern=r"^[A-Za-z0-9.\-]+$",
    ),
]
DaysArg = Annotated[
    Optional[int],
    Field(
        ge=2,
        le=30,
        description="Calendar day window for the trend (defaults to deployment setting).",
    ),
]


class _CacheEntry:
    __slots__ = ("timestamp", "bars")

    def __init__(self, timestamp: datetime, bars: List):
        self.timestamp = timestamp
        self.bars = bars


class _FinanceCache:
    def __init__(self, ttl: timedelta) -> None:
        self._ttl = ttl
        self._entries: Dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> tuple[Optional[List], float]:
        async with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None, 0.0
            age = (datetime.utcnow() - entry.timestamp).total_seconds()
            if age > self._ttl.total_seconds():
                self._entries.pop(key, None)
                return None, age
            return entry.bars, age

    async def set(self, key: str, bars: List) -> None:
        async with self._lock:
            self._entries[key] = _CacheEntry(datetime.utcnow(), bars)


def create_disabled_finance_tool(message: str | None = None):
    note = message or "stock_trend is disabled for this deployment."

    @function_tool(name_override="stock_trend")
    async def stock_trend(symbol: SymbolArg, days: DaysArg = None) -> Dict[str, object]:
        LOGGER.debug("stock_trend requested while disabled", extra={"symbol": symbol})
        return {"error": note, "symbol": symbol.upper()}

    return stock_trend


def create_stock_trend_tool(
    client: FinanceClient,
    provider_name: str,
    default_window_days: int,
    cache_ttl_minutes: int,
    max_fetch_bars: int = 120,
):
    cache = _FinanceCache(ttl=timedelta(minutes=max(cache_ttl_minutes, 1)))

    @function_tool(name_override="stock_trend")
    async def stock_trend(symbol: SymbolArg, days: DaysArg = None) -> Dict[str, object]:
        ticker = symbol.upper()
        window_days = days or default_window_days
        if window_days < 2:
            window_days = 2
        cached, age_seconds = await cache.get(ticker)
        cache_hit = cached is not None
        if cached is None:
            try:
                bars = await client.daily_bars(ticker, limit=max_fetch_bars)
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("stock_trend fetch failed", extra={"symbol": ticker})
                return {
                    "symbol": ticker,
                    "error": f"stock_trend failed: {exc}",
                    "provider": provider_name,
                }
            await cache.set(ticker, bars)
        else:
            bars = cached

        if not bars:
            return {
                "symbol": ticker,
                "error": "No pricing data available for the requested ticker.",
                "provider": provider_name,
            }

        window = trend.select_window(bars, window_days)
        if not window.has_enough_data:
            return {
                "symbol": ticker,
                "error": "Insufficient historical data to compute trend.",
                "provider": provider_name,
            }

        price = window.bars[-1].close
        change_pct = trend.percent_change(window)
        slope_value = trend.slope(window)
        volatility = trend.volatility_pct(window)
        high, low = trend.high_low(window)
        spark = trend.sparkline(window)
        series = trend.compact_series(window)
        as_of = window.bars[-1].date.strftime("%Y-%m-%d")

        return {
            "symbol": ticker,
            "provider": provider_name,
            "as_of": as_of,
            "price": round(price, 4),
            "change_pct": round(change_pct, 2),
            "window_days": window_days,
            "high": round(high, 4),
            "low": round(low, 4),
            "slope": round(slope_value, 4),
            "volatility_pct": round(volatility, 3),
            "sparkline": spark,
            "bars": series,
            "cache": {"hit": cache_hit, "age_seconds": round(age_seconds, 2)},
            "note": "Prices reflect recent daily closes. Data may be delayed and is not financial advice.",
        }

    return stock_trend


__all__ = ["create_stock_trend_tool", "create_disabled_finance_tool"]
