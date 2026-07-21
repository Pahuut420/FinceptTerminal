"""
CoinCap provider — crypto quotes / top / history / search.

Public endpoints (api.coincap.io v2), no API key required. Numeric fields in
CoinCap's payloads arrive as strings (or null), so every value is parsed
defensively. The (large) assets list is cached briefly — mirroring
CoinPaprika's ticker cache — so quotes()/search() can filter in-memory instead
of hitting the network per symbol.
"""
from __future__ import annotations

import time
from typing import List, Optional

import requests

from finscope.core.contracts import AssetClass, OHLCV, Quote, Series, ProviderError
from finscope.core.providers.base import Provider, register

_BASE = "https://api.coincap.io/v2"
_UA = {"User-Agent": "FinScope/0.1 (+finceptterminal)"}


def _get(path: str, params: Optional[dict] = None, timeout: float = 20.0):
    try:
        r = requests.get(f"{_BASE}/{path}", params=params, headers=_UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise ProviderError(f"coincap: {e}") from e
    except ValueError as e:  # bad/non-JSON body
        raise ProviderError(f"coincap: invalid JSON response: {e}") from e


def _f(v) -> Optional[float]:
    """CoinCap returns numeric fields as strings (or null); parse safely."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rank_key(a: dict) -> int:
    try:
        return int(a.get("rank") or 10**9)
    except (TypeError, ValueError):
        return 10**9


@register
class CoinCapProvider(Provider):
    name = "coincap"
    asset_classes = (AssetClass.CRYPTO,)
    requires_key = False
    priority = 50
    can_quote = True
    can_top = True
    can_history = True
    can_search = True

    # cache the (large) assets payload briefly to avoid hammering the API
    _assets_cache: Optional[List[dict]] = None
    _assets_ts: float = 0.0
    _TTL = 30.0

    def _assets(self) -> List[dict]:
        now = time.time()
        if self._assets_cache is None or now - self._assets_ts > self._TTL:
            data = _get("assets", params={"limit": 2000})
            rows = data.get("data") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                raise ProviderError(f"coincap: unexpected assets payload: {type(data)}")
            self._assets_cache = rows
            self._assets_ts = now
        return self._assets_cache

    @staticmethod
    def _to_quote(a: dict) -> Quote:
        rank = a.get("rank")
        try:
            rank_i = int(rank) if rank is not None else None
        except (TypeError, ValueError):
            rank_i = None
        return Quote(
            symbol=(a.get("symbol") or a.get("id") or "?").upper(),
            name=a.get("name") or a.get("id") or "?",
            price=_f(a.get("priceUsd")) or 0.0,
            source="coincap",
            asset_class=AssetClass.CRYPTO,
            change_24h_pct=_f(a.get("changePercent24Hr")),
            volume_24h=_f(a.get("volumeUsd24Hr")),
            market_cap=_f(a.get("marketCapUsd")),
            rank=rank_i,
            extra={"id": a.get("id")},
        )

    def _id_for(self, symbol: str) -> Optional[str]:
        s = symbol.upper()
        slug = symbol.lower()
        for a in self._assets():
            if (a.get("symbol") or "").upper() == s or a.get("id") == slug:
                return a.get("id")
        return None

    def top(self, limit: int = 25, asset_class: AssetClass = AssetClass.CRYPTO) -> List[Quote]:
        rows = sorted(self._assets(), key=_rank_key)
        return [self._to_quote(a) for a in rows[:limit]]

    def quotes(self, symbols: List[str]) -> List[Quote]:
        want = {s.upper() for s in symbols}
        want_ids = {s.lower() for s in symbols}
        out: List[Quote] = []
        for a in self._assets():
            sym = (a.get("symbol") or "").upper()
            if sym in want or a.get("id") in want_ids:
                out.append(self._to_quote(a))
        return out

    def search(self, query: str, limit: int = 20) -> List[Quote]:
        q = query.lower()
        hits = [
            a for a in self._assets()
            if q in (a.get("name") or "").lower() or q in (a.get("symbol") or "").lower()
        ]
        hits.sort(key=_rank_key)
        return [self._to_quote(a) for a in hits[:limit]]

    def history(self, symbol: str, days: int = 90) -> Series:
        cid = self._id_for(symbol)
        if not cid:
            raise ProviderError(f"coincap: unknown symbol {symbol!r}")
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 86400 * 1000
        data = _get(f"assets/{cid}/history",
                    params={"interval": "d1", "start": start_ms, "end": end_ms})
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise ProviderError(f"coincap: unexpected history payload for {symbol}")
        candles: List[OHLCV] = []
        for row in rows:
            price = _f(row.get("priceUsd"))
            ts_ms = row.get("time")
            if price is None or ts_ms is None:
                continue
            try:
                ts = float(ts_ms) / 1000.0
            except (TypeError, ValueError):
                continue
            candles.append(OHLCV(ts=ts, open=price, high=price, low=price, close=price, volume=0.0))
        return Series(symbol=symbol.upper(), candles=candles, source="coincap",
                      asset_class=AssetClass.CRYPTO)

    def healthy(self) -> bool:
        try:
            _get("assets", params={"limit": 1}, timeout=8.0)
            return True
        except ProviderError:
            return False
