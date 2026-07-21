"""
Lake ingestion — fill the DuckDB/Parquet lake with OHLCV history.

Two sources:
  * bootstrap_from_seed(): the real Crypto.com snapshot bundled in demo_seed.json
    (BTC/ETH/SOL daily candles) — works offline, no network.
  * ingest_from_hub(): pull history via the live providers (coinpaprika, coincap,
    cryptocompare, yfinance, ...) into the lake — needs network on the user's box.

Filling the lake is what turns the small-sample backtests into statistically
meaningful ones.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from finscope.core.contracts import AssetClass, OHLCV, Series
from finscope.core.datahub import DataHub
from finscope.lake.store import DuckDBLake


def bootstrap_from_seed(lake: Optional[DuckDBLake] = None) -> dict:
    lake = lake or DuckDBLake()
    seed_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "demo_seed.json")
    with open(seed_path, "r", encoding="utf-8") as f:
        seed = json.load(f)
    written = {}
    for sym, hist in seed.get("history", {}).items():
        candles = [OHLCV(ts=float(c["ts"]), open=float(c["open"]), high=float(c["high"]),
                         low=float(c["low"]), close=float(c["close"]),
                         volume=float(c.get("volume", 0.0))) for c in hist]
        s = Series(symbol=sym, candles=candles, source="crypto.com-seed",
                   asset_class=AssetClass.CRYPTO)
        written[sym] = lake.write_series(s)
    return {"source": "demo_seed", "written": written}


def ingest_from_hub(symbols: List[str], days: int = 365,
                    asset_class: AssetClass = AssetClass.CRYPTO,
                    hub: Optional[DataHub] = None,
                    lake: Optional[DuckDBLake] = None) -> dict:
    hub = hub or DataHub()
    lake = lake or DuckDBLake()
    written, errors = {}, {}
    for sym in symbols:
        try:
            s = hub.history(sym, days=days, asset_class=asset_class)
            if s and len(s):
                written[sym] = lake.write_series(s)
            else:
                errors[sym] = "no history returned"
        except Exception as e:
            errors[sym] = str(e)
    return {"source": "hub", "written": written, "errors": errors}
