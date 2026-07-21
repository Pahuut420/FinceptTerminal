"""
DemoProvider — bundled real market snapshot, used as a graceful fallback.

The data in `finscope/data/demo_seed.json` is a *real* Crypto.com snapshot
(top-30 USD pairs + BTC/ETH/SOL daily candles) captured 2026-07-21. It ships so
the terminal renders live-quality data with no network and in CI. Its priority
is high (900) so any live provider is tried first; DemoProvider only answers when
the network is unreachable.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from finscope.core.contracts import AssetClass, OHLCV, Quote, Series, ProviderError
from finscope.core.providers.base import Provider, register

_SEED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                          "data", "demo_seed.json")


def _load_seed() -> dict:
    try:
        with open(_SEED_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:  # pragma: no cover
        raise ProviderError(f"demo: cannot load seed: {e}") from e


@register
class DemoProvider(Provider):
    name = "demo"
    asset_classes = (AssetClass.CRYPTO,)
    requires_key = False
    priority = 900  # fallback: tried after any live provider
    can_quote = True
    can_top = True
    can_history = True
    can_search = True

    def __init__(self) -> None:
        self._seed: Optional[dict] = None

    @property
    def seed(self) -> dict:
        if self._seed is None:
            self._seed = _load_seed()
        return self._seed

    def _quote(self, row: dict) -> Quote:
        return Quote(
            symbol=row["symbol"], name=row.get("name", row["symbol"]),
            price=float(row.get("price") or 0.0), source="demo",
            asset_class=AssetClass.CRYPTO,
            change_24h_pct=row.get("change_24h_pct"),
            volume_24h=row.get("volume_24h"), market_cap=row.get("market_cap"),
            rank=row.get("rank"),
        )

    def top(self, limit: int = 25, asset_class: AssetClass = AssetClass.CRYPTO) -> List[Quote]:
        rows = self.seed.get("quotes", [])
        return [self._quote(r) for r in rows[:limit]]

    def quotes(self, symbols: List[str]) -> List[Quote]:
        want = {s.upper() for s in symbols}
        return [self._quote(r) for r in self.seed.get("quotes", [])
                if r["symbol"].upper() in want]

    def search(self, query: str, limit: int = 20) -> List[Quote]:
        q = query.lower()
        hits = [r for r in self.seed.get("quotes", [])
                if q in r["symbol"].lower() or q in r.get("name", "").lower()]
        return [self._quote(r) for r in hits[:limit]]

    def history(self, symbol: str, days: int = 90) -> Series:
        hist = self.seed.get("history", {}).get(symbol.upper())
        if not hist:
            raise ProviderError(f"demo: no bundled history for {symbol}")
        candles = [
            OHLCV(ts=float(c["ts"]), open=float(c["open"]), high=float(c["high"]),
                  low=float(c["low"]), close=float(c["close"]), volume=float(c.get("volume", 0.0)))
            for c in hist
        ]
        return Series(symbol=symbol.upper(), candles=candles, source="demo",
                      asset_class=AssetClass.CRYPTO)

    def healthy(self) -> bool:
        try:
            return bool(self.seed.get("quotes"))
        except ProviderError:
            return False
