"""
DataHub — the single entry point the UI and analytics talk to.

Responsibilities:
  * hold the registered providers (auto-discovered from finscope.core.providers)
  * route a request to a capable provider for the asset class
  * cache results with a short TTL so the live dashboard doesn't hammer APIs
  * degrade gracefully: a ProviderError never crashes the terminal — the hub
    falls back to cache, then to the next capable provider, then to empty.

The hub is deliberately synchronous and dependency-light. The refresh loop in
the UI calls it on a background thread.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

from finscope.core.contracts import AssetClass, Quote, Series, ProviderError
from finscope.core import providers as _providers


class _TTLCache:
    def __init__(self, ttl: float = 20.0):
        self.ttl = ttl
        self._store: Dict[str, Tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            hit = self._store.get(key)
        if not hit:
            return None
        ts, val = hit
        if time.time() - ts > self.ttl:
            return None
        return val

    def put(self, key: str, val: object) -> None:
        with self._lock:
            self._store[key] = (time.time(), val)


class DataHub:
    DEFAULT_WATCHLIST = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"]

    def __init__(self, ttl: float = 20.0, watchlist: Optional[List[str]] = None):
        # importing the package has already auto-registered providers
        self.providers = {p.name: p for p in _providers.all_providers()}
        self.cache = _TTLCache(ttl)
        self.watchlist: List[str] = list(watchlist or self.DEFAULT_WATCHLIST)
        self.last_error: Optional[str] = None

    # ---- introspection ----
    def provider_names(self) -> List[str]:
        return sorted(self.providers)

    def providers_for(self, asset_class: AssetClass, capability: str):
        return _providers.providers_for(asset_class, capability)

    # ---- routed data access (each catches ProviderError and moves on) ----
    def top(self, asset_class: AssetClass = AssetClass.CRYPTO, limit: int = 25) -> List[Quote]:
        key = f"top:{asset_class.value}:{limit}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        # Merge across providers until `limit` distinct symbols are collected --
        # a single-symbol provider (e.g. blockchain.com, BTC-only) returning a
        # short but non-empty list must not short-circuit providers that could
        # actually satisfy the requested count.
        merged: List[Quote] = []
        seen: set = set()
        for p in self.providers_for(asset_class, "top"):
            if len(merged) >= limit:
                break
            try:
                rows = p.top(limit=limit, asset_class=asset_class)
            except ProviderError as e:
                self.last_error = str(e)
                continue
            for q in rows:
                if q.symbol in seen:
                    continue
                seen.add(q.symbol)
                merged.append(q)
                if len(merged) >= limit:
                    break
        if merged:
            self.cache.put(key, merged)
            return merged
        return cached or []

    def quotes(self, symbols: List[str], asset_class: AssetClass = AssetClass.CRYPTO) -> List[Quote]:
        key = f"q:{asset_class.value}:{','.join(sorted(symbols))}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        # Same merge principle: a provider covering only some of the requested
        # symbols (e.g. blockchain.com only ever answers for BTC) must not stop
        # the search -- ask the remaining providers for whatever's still missing.
        merged: List[Quote] = []
        remaining = list(dict.fromkeys(symbols))  # de-dup, preserve caller order
        for p in self.providers_for(asset_class, "quote"):
            if not remaining:
                break
            try:
                rows = p.quotes(remaining)
            except ProviderError as e:
                self.last_error = str(e)
                continue
            found = {q.symbol.upper() for q in rows}
            merged.extend(rows)
            remaining = [s for s in remaining if s.upper() not in found]
        if merged:
            self.cache.put(key, merged)
            return merged
        return cached or []

    def quote(self, symbol: str, asset_class: AssetClass = AssetClass.CRYPTO) -> Optional[Quote]:
        rows = self.quotes([symbol], asset_class=asset_class)
        return rows[0] if rows else None

    def history(self, symbol: str, days: int = 90,
                asset_class: AssetClass = AssetClass.CRYPTO) -> Optional[Series]:
        key = f"h:{asset_class.value}:{symbol}:{days}"
        cached = self.cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        for p in self.providers_for(asset_class, "history"):
            try:
                s = p.history(symbol, days=days)
                if s and len(s):
                    self.cache.put(key, s)
                    return s
            except ProviderError as e:
                self.last_error = str(e)
        return cached  # may be None

    def search(self, query: str, asset_class: AssetClass = AssetClass.CRYPTO,
               limit: int = 20) -> List[Quote]:
        for p in self.providers_for(asset_class, "search"):
            try:
                rows = p.search(query, limit=limit)
                if rows:
                    return rows
            except ProviderError as e:
                self.last_error = str(e)
        return []

    def watchlist_quotes(self) -> List[Quote]:
        return self.quotes(self.watchlist, asset_class=AssetClass.CRYPTO)

    def health(self) -> Dict[str, bool]:
        out: Dict[str, bool] = {}
        for name, p in self.providers.items():
            try:
                out[name] = bool(p.healthy())
            except Exception:
                out[name] = False
        return out
