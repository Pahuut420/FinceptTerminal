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
import time
from typing import List, Optional

import numpy as np

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


def generate_synthetic(symbols: List[str], bars: int = 730, seed: int = 7,
                       prefix: str = "SYN_", drift: float = 0.0,
                       lake: Optional[DuckDBLake] = None) -> dict:
    """Write SYNTHETIC OHLCV (fat-tailed GBM) for pipeline-scale + null testing.

    NOT real market data. Anchored on each symbol's real seed price/vol, then a
    Student-t random walk. DRIFTLESS by default (drift=0) so it has no predictable
    structure — a correct (no-lookahead) strategy should score ~0 Sharpe on it.
    A materially non-zero Sharpe on driftless synthetic reveals a lookahead/
    overfitting bug. Pass drift>0 for a realistic-scale (trending) series instead.
    Written under a `SYN_` prefix so real seed symbols are never overwritten.
    """
    lake = lake or DuckDBLake()
    seed_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "demo_seed.json")
    seed_data = json.load(open(seed_path, encoding="utf-8"))
    written = {}
    for i, sym in enumerate(symbols):
        hist = seed_data.get("history", {}).get(sym.upper())
        if hist and len(hist) > 2:
            cl = np.array([c["close"] for c in hist], dtype=float)
            lr = np.diff(np.log(cl))
            mu, sd, last = float(np.mean(lr)), float(np.std(lr)) or 0.03, float(cl[-1])
        else:
            q = next((q for q in seed_data.get("quotes", []) if q["symbol"] == sym.upper()), None)
            mu, sd, last = 0.0, 0.03, float(q["price"]) if q else 100.0
        rng = np.random.default_rng(seed + i)
        df = 4  # Student-t degrees of freedom -> fat tails
        innov = rng.standard_t(df, size=bars) * sd / np.sqrt(df / (df - 2))
        logrets = drift + innov  # driftless by default -> a clean null test
        prices = last * np.exp(np.cumsum(logrets))
        prices *= last / prices[-1]  # pin the final price to the real last close
        # fixed epoch so re-running synth is idempotent (same ts -> clean upsert, no
        # accumulation / discontinuities from two mismatched generations)
        start_ts = 1_600_000_000.0
        candles = []
        for j, p in enumerate(prices):
            o = float(prices[j - 1]) if j > 0 else float(p * np.exp(-logrets[0]))
            wick = abs(rng.normal(0, sd * 0.5)) * p
            candles.append(OHLCV(ts=start_ts + j * 86400, open=o,
                                 high=float(max(o, p) + wick), low=float(min(o, p) - wick),
                                 close=float(p), volume=float(abs(rng.normal(1e6, 2e5)))))
        s = Series(symbol=f"{prefix}{sym.upper()}", candles=candles, source="synthetic",
                   asset_class=AssetClass.CRYPTO)
        written[f"{prefix}{sym.upper()}"] = lake.write_series(s)
    return {"source": "synthetic", "bars": bars, "seed": seed, "written": written,
            "WARNING": "SYNTHETIC data (fat-tailed GBM) for pipeline-scale + null testing "
                       "ONLY. Not real market history; high Sharpe here = a lookahead bug."}

