"""
DeFiLlama provider — onchain protocol TVL quotes / top / search.

Public endpoints (api.llama.fi), no API key required. DeFiLlama tracks
protocol Total Value Locked (TVL), not token prices, so this provider treats
`price` and `market_cap` as the protocol's TVL in USD — the closest analogue
an onchain protocol has to a market quote. The raw fields (slug, category,
chain, coingecko-reported mcap where available) are preserved under `extra`
for callers that want the distinction.
"""
from __future__ import annotations

import time
from typing import List, Optional

import requests

from finscope.core.contracts import AssetClass, Quote, ProviderError
from finscope.core.providers.base import Provider, register

_BASE = "https://api.llama.fi"
_UA = {"User-Agent": "FinScope/0.1 (+finceptterminal)"}


def _get(path: str, params: Optional[dict] = None, timeout: float = 20.0):
    try:
        r = requests.get(f"{_BASE}/{path}", params=params, headers=_UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise ProviderError(f"defillama: {e}") from e
    except ValueError as e:  # bad/non-JSON body
        raise ProviderError(f"defillama: invalid JSON response: {e}") from e


def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@register
class DefiLlamaProvider(Provider):
    name = "defillama"
    asset_classes = (AssetClass.ONCHAIN,)
    requires_key = False
    priority = 50
    can_quote = True
    can_top = True
    can_search = True

    # cache the (large) protocols payload briefly to avoid hammering the API
    _protocols_cache: Optional[List[dict]] = None
    _protocols_ts: float = 0.0
    _TTL = 60.0

    def _protocols(self) -> List[dict]:
        now = time.time()
        if self._protocols_cache is None or now - self._protocols_ts > self._TTL:
            data = _get("protocols")
            if not isinstance(data, list):
                raise ProviderError(f"defillama: unexpected protocols payload: {type(data)}")
            rows = sorted(data, key=lambda p: _f(p.get("tvl")) or 0.0, reverse=True)
            self._protocols_cache = rows
            self._protocols_ts = now
        return self._protocols_cache

    @staticmethod
    def _symbol_for(p: dict) -> str:
        sym = p.get("symbol")
        if sym and sym != "-":
            return sym.upper()
        return (p.get("slug") or p.get("name") or "?").upper()

    @classmethod
    def _to_quote(cls, p: dict, rank: Optional[int] = None) -> Quote:
        tvl = _f(p.get("tvl")) or 0.0
        return Quote(
            symbol=cls._symbol_for(p),
            name=p.get("name") or p.get("slug") or "?",
            price=tvl,
            source="defillama",
            asset_class=AssetClass.ONCHAIN,
            change_24h_pct=_f(p.get("change_1d")),
            volume_24h=None,
            market_cap=tvl,
            rank=rank,
            extra={
                "slug": p.get("slug"),
                "category": p.get("category"),
                "chain": p.get("chain"),
                "mcap_coingecko": p.get("mcap"),
            },
        )

    def top(self, limit: int = 25, asset_class: AssetClass = AssetClass.ONCHAIN) -> List[Quote]:
        rows = self._protocols()
        return [self._to_quote(p, rank=i + 1) for i, p in enumerate(rows[:limit])]

    def quotes(self, symbols: List[str]) -> List[Quote]:
        want = {s.upper() for s in symbols}
        want_slugs = {s.lower() for s in symbols}
        out: List[Quote] = []
        for i, p in enumerate(self._protocols()):
            sym = self._symbol_for(p)
            slug = (p.get("slug") or "").lower()
            if sym in want or slug in want_slugs:
                out.append(self._to_quote(p, rank=i + 1))
        return out

    def search(self, query: str, limit: int = 20) -> List[Quote]:
        q = query.lower()
        rows = self._protocols()
        hits = [
            (i, p) for i, p in enumerate(rows)
            if q in (p.get("name") or "").lower()
            or q in (p.get("symbol") or "").lower()
            or q in (p.get("slug") or "").lower()
        ]
        return [self._to_quote(p, rank=i + 1) for i, p in hits[:limit]]

    def healthy(self) -> bool:
        try:
            _get("protocols", timeout=8.0)
            return True
        except ProviderError:
            return False
