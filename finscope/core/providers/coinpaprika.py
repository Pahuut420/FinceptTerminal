"""
CoinPaprika provider — crypto quotes / top / history / search.

Public endpoints, no API key required (71k+ assets). Mirrors the endpoint
surface already used by FinceptTerminal's `scripts/coinpaprika_data.py`, adapted
to FinScope contracts.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests

from finscope.core.contracts import AssetClass, OHLCV, Quote, Series, ProviderError
from finscope.core.providers.base import Provider, register

_BASE = "https://api.coinpaprika.com/v1"
_UA = {"User-Agent": "FinScope/0.1 (+finceptterminal)"}


def _get(path: str, params: Optional[dict] = None, timeout: float = 20.0):
    try:
        r = requests.get(f"{_BASE}/{path}", params=params, headers=_UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise ProviderError(f"coinpaprika: {e}") from e


@register
class CoinPaprikaProvider(Provider):
    name = "coinpaprika"
    asset_classes = (AssetClass.CRYPTO,)
    requires_key = False
    can_quote = True
    can_top = True
    can_history = True
    can_search = True

    # cache the (large) tickers payload briefly to avoid hammering the API
    _tickers_cache: Optional[List[dict]] = None
    _tickers_ts: float = 0.0
    _TTL = 30.0

    def _tickers(self) -> List[dict]:
        now = time.time()
        if self._tickers_cache is None or now - self._tickers_ts > self._TTL:
            data = _get("tickers", params={"limit": 500})
            if not isinstance(data, list):
                raise ProviderError(f"coinpaprika: unexpected tickers payload: {type(data)}")
            self._tickers_cache = data
            self._tickers_ts = now
        return self._tickers_cache

    @staticmethod
    def _to_quote(t: dict) -> Quote:
        usd = (t.get("quotes") or {}).get("USD") or {}
        return Quote(
            symbol=(t.get("symbol") or t.get("id") or "?").upper(),
            name=t.get("name") or t.get("id") or "?",
            price=float(usd.get("price") or 0.0),
            source="coinpaprika",
            asset_class=AssetClass.CRYPTO,
            change_24h_pct=usd.get("percent_change_24h"),
            volume_24h=usd.get("volume_24h"),
            market_cap=usd.get("market_cap"),
            rank=t.get("rank"),
            extra={"id": t.get("id")},
        )

    def _id_for(self, symbol: str) -> Optional[str]:
        s = symbol.upper()
        # accept both "BTC" and "btc-bitcoin"
        if "-" in symbol:
            return symbol.lower()
        for t in self._tickers():
            if (t.get("symbol") or "").upper() == s:
                return t.get("id")
        return None

    def top(self, limit: int = 25, asset_class: AssetClass = AssetClass.CRYPTO) -> List[Quote]:
        rows = sorted(self._tickers(), key=lambda t: t.get("rank") or 10**9)
        return [self._to_quote(t) for t in rows[:limit]]

    def quotes(self, symbols: List[str]) -> List[Quote]:
        want = {s.upper() for s in symbols}
        # also allow "btc-bitcoin" style ids
        want_ids = {s.lower() for s in symbols if "-" in s}
        out: List[Quote] = []
        for t in self._tickers():
            sym = (t.get("symbol") or "").upper()
            if sym in want or (t.get("id") in want_ids):
                out.append(self._to_quote(t))
        return out

    def search(self, query: str, limit: int = 20) -> List[Quote]:
        q = query.lower()
        hits = [
            t for t in self._tickers()
            if q in (t.get("name") or "").lower() or q in (t.get("symbol") or "").lower()
        ]
        hits.sort(key=lambda t: t.get("rank") or 10**9)
        return [self._to_quote(t) for t in hits[:limit]]

    def history(self, symbol: str, days: int = 90) -> Series:
        cid = self._id_for(symbol)
        if not cid:
            raise ProviderError(f"coinpaprika: unknown symbol {symbol!r}")
        start = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400))
        data = _get(f"coins/{cid}/ohlcv/historical", params={"start": start})
        if not isinstance(data, list):
            raise ProviderError(f"coinpaprika: unexpected ohlcv payload for {symbol}")
        candles: List[OHLCV] = []
        for row in data:
            candles.append(OHLCV(
                ts=float(row.get("time_open_ts") or _parse_ts(row.get("time_open"))),
                open=float(row.get("open") or 0.0),
                high=float(row.get("high") or 0.0),
                low=float(row.get("low") or 0.0),
                close=float(row.get("close") or 0.0),
                volume=float(row.get("volume") or 0.0),
            ))
        return Series(symbol=symbol.upper(), candles=candles, source="coinpaprika",
                      asset_class=AssetClass.CRYPTO)

    def healthy(self) -> bool:
        try:
            _get("global", timeout=8.0)
            return True
        except ProviderError:
            return False


def _parse_ts(s: Optional[str]) -> float:
    if not s:
        return 0.0
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(s, fmt))
        except ValueError:
            continue
    return 0.0
