from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Protocol

import httpx

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DailyBar:
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


class FinanceClient(Protocol):
    async def daily_bars(self, symbol: str, limit: int = 100) -> List[DailyBar]:
        """Fetch recent daily bars for a ticker, most recent first."""


class AlphaVantageClient:
    """Adapter for the Alpha Vantage TIME_SERIES_DAILY endpoint."""

    _BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        if not api_key:
            raise ValueError("FINANCE_API_KEY must be configured for Alpha Vantage")
        self._api_key = api_key
        self._timeout = timeout

    async def daily_bars(self, symbol: str, limit: int = 100) -> List[DailyBar]:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self._api_key,
            "outputsize": "compact" if limit <= 100 else "full",
        }
        LOGGER.info("Requesting Alpha Vantage daily bars", extra={"symbol": symbol, "limit": limit})
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(self._BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error for symbol {symbol}: {data['Error Message']}")
        if "Note" in data:
            raise RuntimeError(f"Alpha Vantage rate limit or notice: {data['Note']}")

        series = data.get("Time Series (Daily)")
        if not isinstance(series, dict):
            raise RuntimeError(f"Alpha Vantage response missing daily series for {symbol}")

        bars: List[DailyBar] = []
        for date_str, values in series.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError as exc:
                LOGGER.warning("Skipping invalid date from Alpha Vantage", exc_info=exc)
                continue
            if not isinstance(values, dict):
                continue
            try:
                open_price = float(values.get("1. open"))
                high_price = float(values.get("2. high"))
                low_price = float(values.get("3. low"))
                close_price = float(values.get("4. close"))
                volume = int(float(values.get("5. volume")))
            except (TypeError, ValueError) as exc:
                LOGGER.warning("Skipping malformed bar", exc_info=exc)
                continue
            bars.append(
                DailyBar(
                    symbol=symbol.upper(),
                    date=date,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                )
            )

        bars.sort(key=lambda bar: bar.date, reverse=True)
        if limit and len(bars) > limit:
            bars = bars[:limit]
        LOGGER.info("Fetched %s Alpha Vantage bars", len(bars))
        return bars


__all__ = ["AlphaVantageClient", "DailyBar", "FinanceClient"]
