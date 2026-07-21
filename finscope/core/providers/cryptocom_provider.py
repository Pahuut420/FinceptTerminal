"""
Crypto.com provider — CRYPTO quotes, top, history, and orderbook via public REST API.

No API key required. Uses Crypto.com's public exchange API for tickers, candles, and
order book data. Instrument names use the underscore format (e.g., BTC_USD, ETH_USDT).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests

from finscope.core.contracts import AssetClass, OHLCV, Quote, Series, ProviderError
from finscope.core.providers.base import Provider, register

_BASE = "https://api.crypto.com/exchange/v1/public"
_UA = {"User-Agent": "FinScope/0.1 (+finceptterminal)"}


def _get(method: str, params: Optional[dict] = None, timeout: float = 20.0) -> dict:
    try:
        url = f"{_BASE}/{method}"
        r = requests.get(url, params=params, headers=_UA, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "0":
            raise ProviderError(f"cryptocom: API error {data.get('code')}: {data.get('message')}")
        return data.get("result", {})
    except requests.RequestException as e:
        raise ProviderError(f"cryptocom: {e}") from e


@register
class CryptoComProvider(Provider):
    name = "cryptocom"
    asset_classes = (AssetClass.CRYPTO,)
    requires_key = False
    priority = 40  # tried before coinpaprika
    can_quote = True
    can_top = True
    can_history = True

    @staticmethod
    def _normalize_symbol(instrument: str) -> str:
        """Convert BTC_USD -> BTC, ETH_USDT -> ETH."""
        return instrument.split("_")[0] if "_" in instrument else instrument

    @staticmethod
    def _to_quote(ticker: dict) -> Quote:
        """Convert a ticker dict to a Quote."""
        instrument = ticker.get("instrument_name", "?")
        symbol = CryptoComProvider._normalize_symbol(instrument)

        price = float(ticker.get("a", 0.0))  # ask price
        if not price:
            price = float(ticker.get("b", 0.0))  # bid price
        if not price:
            price = float(ticker.get("h24", 0.0))  # high 24h

        open_24h = float(ticker.get("o", price))
        change_24h_pct = None
        if open_24h and price:
            change_24h_pct = ((price - open_24h) / open_24h) * 100

        return Quote(
            symbol=symbol,
            name=instrument,
            price=price,
            source="cryptocom",
            asset_class=AssetClass.CRYPTO,
            change_24h_pct=change_24h_pct,
            volume_24h=float(ticker.get("v", 0.0)),
            market_cap=None,
        )

    def quotes(self, symbols: List[str]) -> List[Quote]:
        """Fetch quotes for symbols. Converts BTC -> BTC_USD."""
        data = _get("get-tickers")
        tickers = data.get("tickers", [])

        if not tickers:
            raise ProviderError("cryptocom: no tickers found")

        want = {s.upper() for s in symbols}
        out: List[Quote] = []

        for ticker in tickers:
            instrument = ticker.get("instrument_name", "").upper()
            symbol = self._normalize_symbol(instrument)

            if symbol in want:
                try:
                    q = self._to_quote(ticker)
                    out.append(q)
                except (ValueError, KeyError):
                    continue

        if not out:
            raise ProviderError(f"cryptocom: no quotes found for {symbols}")
        return out

    def top(self, limit: int = 25, asset_class: AssetClass = AssetClass.CRYPTO) -> List[Quote]:
        """Fetch top coins by volume."""
        data = _get("get-tickers")
        tickers = data.get("tickers", [])

        if not tickers:
            raise ProviderError("cryptocom: no tickers found")

        # Sort by volume descending
        tickers.sort(key=lambda t: float(t.get("v", 0.0)), reverse=True)

        out: List[Quote] = []
        for ticker in tickers[:limit]:
            try:
                q = self._to_quote(ticker)
                out.append(q)
            except (ValueError, KeyError):
                continue

        if not out:
            raise ProviderError("cryptocom: no quotes found for top")
        return out

    def history(self, symbol: str, days: int = 90) -> Series:
        """Fetch daily OHLCV history. Crypto.com uses timeframe for interval."""
        sym_upper = symbol.upper()
        instrument = f"{sym_upper}_USD"

        # Map days to timeframe: crypto.com uses 1H, 4H, 1D, 1W, 1M
        # For daily candles: 1D gets 365 max; for longer periods need multiple fetches or accept truncation
        timeframe = "1D"

        params = {
            "instrument_name": instrument,
            "timeframe": timeframe,
        }

        try:
            data = _get("get-candlestick", params=params)
            candles_data = data.get("candlesticks", [])

            if not candles_data:
                raise ProviderError(f"cryptocom: no candles for {symbol}")

            candles: List[OHLCV] = []
            for candle in candles_data:
                # Crypto.com format: [t, o, h, l, c, v]
                try:
                    t = int(candle[0]) // 1000  # convert ms to seconds
                    o = float(candle[1])
                    h = float(candle[2])
                    l = float(candle[3])
                    c = float(candle[4])
                    v = float(candle[5]) if len(candle) > 5 else 0.0

                    if c > 0:
                        candles.append(OHLCV(ts=t, open=o, high=h, low=l, close=c, volume=v))
                except (IndexError, ValueError):
                    continue

            if not candles:
                raise ProviderError(f"cryptocom: no valid candles for {symbol}")

            return Series(symbol=sym_upper, candles=candles, source="cryptocom",
                         asset_class=AssetClass.CRYPTO)
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"cryptocom: failed to parse history for {symbol}: {e}") from e

    def orderbook(self, symbol: str, depth: int = 10) -> dict:
        """Fetch live orderbook (bids/asks). Returns dict with spread info."""
        sym_upper = symbol.upper()
        instrument = f"{sym_upper}_USD"

        params = {
            "instrument_name": instrument,
            "depth": depth,
        }

        data = _get("get-book", params=params)

        bids = data.get("bids", [])  # [[price, size], ...]
        asks = data.get("asks", [])

        # Compute spread
        bid_price = float(bids[0][0]) if bids and len(bids[0]) > 0 else None
        ask_price = float(asks[0][0]) if asks and len(asks[0]) > 0 else None

        spread = None
        spread_pct = None
        if bid_price is not None and ask_price is not None and ask_price > 0:
            spread = ask_price - bid_price
            spread_pct = (spread / bid_price) * 100 if bid_price > 0 else None

        return {
            "symbol": sym_upper,
            "instrument": instrument,
            "bids": [[float(b[0]), float(b[1])] for b in bids],
            "asks": [[float(a[0]), float(a[1])] for a in asks],
            "spread": spread,
            "spread_pct": spread_pct,
        }

    def healthy(self) -> bool:
        try:
            _get("get-tickers", timeout=8.0)
            return True
        except ProviderError:
            return False
