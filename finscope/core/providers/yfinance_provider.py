"""
Yahoo Finance provider — EQUITY quotes and history via Yahoo's public chart API.

No API key required. Mirrors the endpoint surface already used by FinceptTerminal,
adapted to FinScope contracts. Uses browser User-Agent header for API compliance.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests

from finscope.core.contracts import AssetClass, OHLCV, Quote, Series, ProviderError
from finscope.core.providers.base import Provider, register

_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}


def _get(url: str, params: Optional[dict] = None, timeout: float = 20.0) -> dict:
    try:
        r = requests.get(url, params=params, headers=_UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise ProviderError(f"yfinance: {e}") from e


@register
class YFinanceProvider(Provider):
    name = "yfinance"
    asset_classes = (AssetClass.EQUITY,)
    requires_key = False
    priority = 50
    can_quote = True
    can_history = True

    @staticmethod
    def _to_quote(symbol: str, chart_data: dict) -> Optional[Quote]:
        """Extract a Quote from Yahoo's chart API response."""
        try:
            meta = chart_data.get("chart", {}).get("result", [{}])[0].get("meta", {})
            if not meta:
                return None

            regular_market_price = meta.get("regularMarketPrice")
            if regular_market_price is None:
                return None

            previous_close = meta.get("chartPreviousClose", meta.get("previousClose"))
            change_pct = None
            if previous_close and previous_close > 0:
                change_pct = ((regular_market_price - previous_close) / previous_close) * 100

            return Quote(
                symbol=symbol.upper(),
                name=meta.get("longName", symbol),
                price=float(regular_market_price),
                source="yfinance",
                asset_class=AssetClass.EQUITY,
                change_24h_pct=change_pct,
                volume_24h=meta.get("regularMarketVolume"),
                market_cap=None,
            )
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ProviderError(f"yfinance: failed to parse quote for {symbol}: {e}") from e

    def quotes(self, symbols: List[str]) -> List[Quote]:
        out: List[Quote] = []
        for sym in symbols:
            try:
                url = f"{_BASE}/{sym.upper()}"
                data = _get(url)
                q = self._to_quote(sym, data)
                if q:
                    out.append(q)
            except ProviderError:
                continue
        if not out:
            raise ProviderError(f"yfinance: no quotes found for {symbols}")
        return out

    def history(self, symbol: str, days: int = 90) -> Series:
        url = f"{_BASE}/{symbol.upper()}"
        params = {
            "range": f"{days}d",
            "interval": "1d",
        }
        data = _get(url, params=params)

        try:
            result = data.get("chart", {}).get("result", [{}])[0]
            timestamps = result.get("timestamp", [])
            quotes = result.get("indicators", {}).get("quote", [{}])[0]

            if not timestamps or not quotes:
                raise ProviderError(f"yfinance: no candles for {symbol}")

            candles: List[OHLCV] = []
            opens = quotes.get("open", [])
            highs = quotes.get("high", [])
            lows = quotes.get("low", [])
            closes = quotes.get("close", [])
            volumes = quotes.get("volume", [])

            for i, ts in enumerate(timestamps):
                o = opens[i] if i < len(opens) and opens[i] is not None else 0.0
                h = highs[i] if i < len(highs) and highs[i] is not None else 0.0
                l = lows[i] if i < len(lows) and lows[i] is not None else 0.0
                c = closes[i] if i < len(closes) and closes[i] is not None else 0.0
                v = volumes[i] if i < len(volumes) and volumes[i] is not None else 0.0

                if c > 0:  # skip zero/null closes
                    candles.append(OHLCV(
                        ts=float(ts),
                        open=float(o),
                        high=float(h),
                        low=float(l),
                        close=float(c),
                        volume=float(v),
                    ))

            if not candles:
                raise ProviderError(f"yfinance: no valid candles for {symbol}")

            return Series(symbol=symbol.upper(), candles=candles, source="yfinance",
                         asset_class=AssetClass.EQUITY)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ProviderError(f"yfinance: failed to parse history for {symbol}: {e}") from e

    def healthy(self) -> bool:
        try:
            _get(f"{_BASE}/AAPL", timeout=8.0)
            return True
        except ProviderError:
            return False
