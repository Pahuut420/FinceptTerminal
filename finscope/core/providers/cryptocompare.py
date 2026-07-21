"""
CryptoCompare provider — crypto quotes / history.

Public min-api endpoints (min-api.cryptocompare.com/data). No API key is
required for basic usage (unauthenticated calls are simply rate-limited more
aggressively). Only quotes()/history() are implemented — CryptoCompare's free
tier has no free-text asset-discovery/top-movers endpoint as clean as
CoinPaprika's or CoinCap's, so this provider stays intentionally narrow.
"""
from __future__ import annotations

from typing import List, Optional

import requests

from finscope.core.contracts import AssetClass, OHLCV, Quote, Series, ProviderError
from finscope.core.providers.base import Provider, register

_BASE = "https://min-api.cryptocompare.com/data"
_UA = {"User-Agent": "FinScope/0.1 (+finceptterminal)"}


def _get(path: str, params: Optional[dict] = None, timeout: float = 20.0):
    try:
        r = requests.get(f"{_BASE}/{path}", params=params, headers=_UA, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise ProviderError(f"cryptocompare: {e}") from e
    except ValueError as e:  # bad/non-JSON body
        raise ProviderError(f"cryptocompare: invalid JSON response: {e}") from e
    if isinstance(data, dict) and data.get("Response") == "Error":
        raise ProviderError(f"cryptocompare: {data.get('Message', 'unknown error')}")
    return data


def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@register
class CryptoCompareProvider(Provider):
    name = "cryptocompare"
    asset_classes = (AssetClass.CRYPTO,)
    requires_key = False
    priority = 50
    can_quote = True
    can_history = True

    def quotes(self, symbols: List[str]) -> List[Quote]:
        syms = [s.upper() for s in symbols if s]
        if not syms:
            return []
        data = _get("pricemultifull", params={"fsyms": ",".join(syms), "tsyms": "USD"})
        raw = (data or {}).get("RAW") or {}
        out: List[Quote] = []
        for sym in syms:
            bucket = raw.get(sym) or {}
            row = bucket.get("USD")
            if not row:
                continue
            out.append(Quote(
                symbol=sym,
                name=sym,
                price=_f(row.get("PRICE")) or 0.0,
                source="cryptocompare",
                asset_class=AssetClass.CRYPTO,
                change_24h_pct=_f(row.get("CHANGEPCT24HOUR")),
                volume_24h=_f(row.get("VOLUME24HOURTO")) or _f(row.get("VOLUME24HOUR")),
                market_cap=_f(row.get("MKTCAP")),
                extra={"supply": row.get("SUPPLY")},
            ))
        return out

    def history(self, symbol: str, days: int = 90) -> Series:
        sym = symbol.upper()
        data = _get("v2/histoday", params={"fsym": sym, "tsym": "USD", "limit": max(1, days)})
        payload = (data or {}).get("Data") or {}
        rows = payload.get("Data")
        if not isinstance(rows, list):
            raise ProviderError(f"cryptocompare: unexpected histoday payload for {symbol}")
        candles: List[OHLCV] = []
        for row in rows:
            candles.append(OHLCV(
                ts=float(row.get("time") or 0),
                open=float(row.get("open") or 0.0),
                high=float(row.get("high") or 0.0),
                low=float(row.get("low") or 0.0),
                close=float(row.get("close") or 0.0),
                volume=float(row.get("volumeto") or row.get("volumefrom") or 0.0),
            ))
        return Series(symbol=sym, candles=candles, source="cryptocompare",
                      asset_class=AssetClass.CRYPTO)

    def healthy(self) -> bool:
        try:
            _get("price", params={"fsym": "BTC", "tsym": "USD"}, timeout=8.0)
            return True
        except ProviderError:
            return False
