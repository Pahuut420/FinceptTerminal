"""
DuckDBLake — per-symbol Parquet OHLCV store queried through DuckDB.

Layout:  {lake_dir}/ohlcv/{ASSET_CLASS}/{SYMBOL}.parquet
Schema:  ts (double, POSIX secs), open, high, low, close, volume, source

Writes are idempotent (dedupe on ts, keep latest). Reads return a FinScope
`Series`. DuckDB/pyarrow are optional — `lake_available()` is False when absent
and callers degrade to live/demo providers.
"""
from __future__ import annotations

import os
from typing import List, Optional

from finscope.core.contracts import AssetClass, OHLCV, Series

try:
    import duckdb  # type: ignore
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False


def lake_available() -> bool:
    return _HAVE


def default_lake_dir() -> str:
    return os.environ.get("FINSCOPE_LAKE_DIR", os.path.expanduser("~/.finscope/lake"))


class DuckDBLake:
    def __init__(self, lake_dir: Optional[str] = None):
        if not _HAVE:
            raise RuntimeError("data lake needs duckdb + pyarrow (pip install duckdb pyarrow)")
        self.lake_dir = lake_dir or default_lake_dir()
        self.ohlcv_dir = os.path.join(self.lake_dir, "ohlcv")
        os.makedirs(self.ohlcv_dir, exist_ok=True)

    # ---- paths ----
    def _path(self, symbol: str, asset_class: AssetClass) -> str:
        d = os.path.join(self.ohlcv_dir, asset_class.value)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{symbol.upper()}.parquet")

    # ---- write ----
    def write_series(self, series: Series) -> int:
        """Upsert a Series' candles. Returns total rows after merge."""
        rows = {c.ts: c for c in series.candles}
        path = self._path(series.symbol, series.asset_class)
        if os.path.exists(path):
            existing = pq.read_table(path).to_pylist()
            for r in existing:
                rows.setdefault(r["ts"], OHLCV(ts=r["ts"], open=r["open"], high=r["high"],
                                               low=r["low"], close=r["close"],
                                               volume=r.get("volume", 0.0)))
        merged = [rows[t] for t in sorted(rows)]
        table = pa.table({
            "ts": [c.ts for c in merged], "open": [c.open for c in merged],
            "high": [c.high for c in merged], "low": [c.low for c in merged],
            "close": [c.close for c in merged], "volume": [c.volume for c in merged],
            "source": [series.source or "lake"] * len(merged),
        })
        pq.write_table(table, path)
        return len(merged)

    # ---- read ----
    def read_series(self, symbol: str, asset_class: AssetClass = AssetClass.CRYPTO,
                    days: Optional[int] = None) -> Optional[Series]:
        path = self._path(symbol, asset_class)
        if not os.path.exists(path):
            return None
        con = duckdb.connect(database=":memory:")
        try:
            q = f"SELECT ts,open,high,low,close,volume FROM read_parquet('{path}') ORDER BY ts"
            recs = con.execute(q).fetchall()
        finally:
            con.close()
        if days:
            recs = recs[-days:]
        candles = [OHLCV(ts=r[0], open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5])
                   for r in recs]
        if not candles:
            return None
        return Series(symbol=symbol.upper(), candles=candles, source="lake",
                      asset_class=asset_class)

    # ---- introspection ----
    def symbols(self, asset_class: Optional[AssetClass] = None) -> List[str]:
        out = []
        classes = [asset_class] if asset_class else list(AssetClass)
        for ac in classes:
            d = os.path.join(self.ohlcv_dir, ac.value)
            if os.path.isdir(d):
                out += [f"{ac.value}:{f[:-8]}" for f in os.listdir(d) if f.endswith(".parquet")]
        return sorted(out)

    def stats(self) -> dict:
        syms = self.symbols()
        total = 0
        con = duckdb.connect(database=":memory:")
        try:
            glob = os.path.join(self.ohlcv_dir, "*", "*.parquet")
            import glob as _g
            if _g.glob(glob):
                total = con.execute(
                    f"SELECT count(*) FROM read_parquet('{glob}')").fetchone()[0]
        finally:
            con.close()
        return {"lake_dir": self.lake_dir, "symbols": len(syms), "rows": total,
                "detail": syms}
