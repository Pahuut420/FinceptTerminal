"""
Stooq provider — EQUITY quotes and history via Stooq's free CSV endpoint.

No API key required. Stooq provides daily historical data and quotes for US equities
via simple CSV export endpoints (no authentication).
"""
from __future__ import annotations

import time
from typing import List, Optional

import requests

from finscope.core.contracts import AssetClass, OHLCV, Quote, Series, ProviderError
from finscope.core.providers.base import Provider, register

_QUOTE_URL = "https://stooq.com/q/l/"
_HISTORY_URL = "https://stooq.com/q/d/l/"
_UA = {"User-Agent": "FinScope/0.1 (+finceptterminal)"}


def _get_csv(url: str, params: Optional[dict] = None, timeout: float = 20.0) -> str:
    try:
        r = requests.get(url, params=params, headers=_UA, timeout=timeout)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        raise ProviderError(f"stooq: {e}") from e


@register
class StooqProvider(Provider):
    name = "stooq"
    asset_classes = (AssetClass.EQUITY,)
    requires_key = False
    priority = 60  # fallback to yfinance
    can_quote = True
    can_history = True

    def quotes(self, symbols: List[str]) -> List[Quote]:
        """Fetch quotes for symbols. Stooq quote endpoint returns CSV with columns:
        Symbol,Name,Close,Open,High,Low,Volume,Time,Bid,Ask
        """
        out: List[Quote] = []
        for sym in symbols:
            try:
                params = {
                    "s": f"{sym.upper()}.us",
                    "f": "sd2t2ohlcv",
                    "h": "",
                    "e": "csv",
                }
                csv_text = _get_csv(_QUOTE_URL, params=params)
                lines = csv_text.strip().split("\n")
                if len(lines) < 2:
                    continue

                # Parse header and first data row
                header = lines[0].split(",")
                data = lines[1].split(",")

                if len(header) < 3 or len(data) < 3:
                    continue

                # Typical format: Symbol,Name,Close,Open,High,Low,Volume,Time
                try:
                    symbol_col = next(i for i, h in enumerate(header) if "Symbol" in h)
                    name_col = next(i for i, h in enumerate(header) if "Name" in h)
                    close_col = next(i for i, h in enumerate(header) if "Close" in h)
                    time_col = next(i for i, h in enumerate(header) if "Time" in h)
                except StopIteration:
                    continue

                price = float(data[close_col]) if data[close_col] else None
                if not price:
                    continue

                q = Quote(
                    symbol=data[symbol_col].replace(".us", "").upper(),
                    name=data[name_col] if name_col < len(data) else data[symbol_col],
                    price=price,
                    source="stooq",
                    asset_class=AssetClass.EQUITY,
                    change_24h_pct=None,
                    volume_24h=None,
                )
                out.append(q)
            except (ValueError, IndexError, KeyError):
                continue

        if not out:
            raise ProviderError(f"stooq: no quotes found for {symbols}")
        return out

    def history(self, symbol: str, days: int = 90) -> Series:
        """Fetch daily OHLCV history. Stooq history endpoint returns CSV:
        Date,Open,High,Low,Close,Volume
        """
        try:
            params = {
                "s": f"{symbol.upper()}.us",
                "i": "d",
            }
            csv_text = _get_csv(_HISTORY_URL, params=params)
            lines = csv_text.strip().split("\n")

            if len(lines) < 2:
                raise ProviderError(f"stooq: no history for {symbol}")

            candles: List[OHLCV] = []
            header = lines[0].split(",")

            # Typical: Date,Open,High,Low,Close,Volume
            date_col = next(i for i, h in enumerate(header) if "Date" in h)
            open_col = next(i for i, h in enumerate(header) if "Open" in h)
            high_col = next(i for i, h in enumerate(header) if "High" in h)
            low_col = next(i for i, h in enumerate(header) if "Low" in h)
            close_col = next(i for i, h in enumerate(header) if "Close" in h)
            vol_col = next(i for i, h in enumerate(header) if "Volume" in h)

            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) <= max(date_col, open_col, high_col, low_col, close_col, vol_col):
                    continue

                try:
                    date_str = parts[date_col].strip()
                    close = float(parts[close_col])
                    if close <= 0:
                        continue

                    ts = time.mktime(time.strptime(date_str, "%Y-%m-%d"))
                    candles.append(OHLCV(
                        ts=ts,
                        open=float(parts[open_col]) if parts[open_col] else close,
                        high=float(parts[high_col]) if parts[high_col] else close,
                        low=float(parts[low_col]) if parts[low_col] else close,
                        close=close,
                        volume=float(parts[vol_col]) if vol_col < len(parts) and parts[vol_col] else 0.0,
                    ))
                except (ValueError, IndexError):
                    continue

            if not candles:
                raise ProviderError(f"stooq: no valid candles for {symbol}")

            return Series(symbol=symbol.upper(), candles=candles, source="stooq",
                         asset_class=AssetClass.EQUITY)
        except (ValueError, StopIteration, IndexError) as e:
            raise ProviderError(f"stooq: failed to parse history for {symbol}: {e}") from e

    def healthy(self) -> bool:
        try:
            _get_csv(_QUOTE_URL, params={"s": "AAPL.us", "f": "sd2t2ohlcv", "h": "", "e": "csv"}, timeout=8.0)
            return True
        except ProviderError:
            return False
