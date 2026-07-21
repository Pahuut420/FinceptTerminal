"""
LakeProvider — serve OHLCV history from the local DuckDB/Parquet lake first.

Priority 10 (highest) so once the lake is filled, backtests read the stored
history instead of hitting the network; symbols not in the lake fall through to
live providers. Unavailable (and silently skipped) if duckdb/pyarrow aren't
installed or the lake is empty.
"""
from __future__ import annotations

from typing import List

from finscope.core.contracts import AssetClass, Series, ProviderError
from finscope.core.providers.base import Provider, register

try:
    from finscope.lake.store import DuckDBLake, lake_available
    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False


@register
class LakeProvider(Provider):
    name = "lake"
    asset_classes = (AssetClass.CRYPTO, AssetClass.EQUITY, AssetClass.ONCHAIN,
                     AssetClass.FX, AssetClass.COMMODITY, AssetClass.OTHER)
    requires_key = False
    priority = 10  # tried before all live providers
    can_history = True

    def __init__(self) -> None:
        self._lake = None

    def _get(self):
        if not _HAVE or not lake_available():
            return None
        if self._lake is None:
            try:
                self._lake = DuckDBLake()
            except Exception:
                return None
        return self._lake

    def history(self, symbol: str, days: int = 90) -> Series:
        lake = self._get()
        if lake is None:
            raise ProviderError("lake: unavailable")
        for ac in (AssetClass.CRYPTO, AssetClass.EQUITY, AssetClass.ONCHAIN, AssetClass.OTHER):
            s = lake.read_series(symbol, asset_class=ac, days=days)
            if s and len(s):
                return s
        raise ProviderError(f"lake: no stored history for {symbol}")

    def healthy(self) -> bool:
        return self._get() is not None
