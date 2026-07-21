"""
FinScope data lake — DuckDB + Parquet columnar store for OHLCV history.

Zero-infra, embedded: OHLCV is written as per-symbol Parquet and queried via
DuckDB. Filling the lake with real history is what makes strategy backtests
statistically meaningful (the small-sample caveat goes away). Optional dependency:
if duckdb/pyarrow are absent the lake is simply unavailable and the rest of
FinScope keeps working.
"""
from finscope.lake.store import DuckDBLake, lake_available

__all__ = ["DuckDBLake", "lake_available"]
