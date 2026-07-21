"""
Tests for finscope.core.contracts.

Covers:
  * Series.from_closes builds N candles (o=h=l=c, evenly spaced timestamps)
  * Series.returns() has length N-1 and matches hand-computed values
  * Quote.is_up sign behavior (positive / negative / zero / None change)

Works both under pytest (functions named test_*) and standalone:
    python3 finscope/tests/test_contracts.py
"""
from __future__ import annotations

import math
import os
import sys

# Make the repo root importable when run as a standalone script (pytest already
# manages this via its rootdir insertion, but this keeps `python3 <file>` working
# too since sys.path[0] is otherwise just this file's own directory).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from finscope.core.contracts import AssetClass, OHLCV, Quote, Series


# ---- Series.from_closes ------------------------------------------------------

def test_from_closes_builds_n_candles():
    closes = [100.0, 101.5, 99.0, 102.25, 103.0]
    series = Series.from_closes("BTC", closes, step=3600.0)
    assert len(series) == len(closes)
    assert len(series.candles) == 5
    # o == h == l == c for every synthesized candle
    for candle, c in zip(series.candles, closes):
        assert candle.open == c
        assert candle.high == c
        assert candle.low == c
        assert candle.close == c
    assert series.closes() == closes


def test_from_closes_timestamps_evenly_spaced():
    closes = [1.0, 2.0, 3.0, 4.0]
    series = Series.from_closes("ETH", closes, start_ts=1000.0, step=60.0)
    ts = series.timestamps()
    assert ts == [1000.0, 1060.0, 1120.0, 1180.0]


def test_from_closes_carries_symbol_source_asset_class():
    series = Series.from_closes("SOL", [1.0, 2.0], source="unit-test",
                                asset_class=AssetClass.CRYPTO)
    assert series.symbol == "SOL"
    assert series.source == "unit-test"
    assert series.asset_class == AssetClass.CRYPTO


def test_from_closes_empty_list():
    series = Series.from_closes("EMPTY", [])
    assert len(series) == 0
    assert series.closes() == []
    assert series.last() is None


# ---- Series.returns() ---------------------------------------------------------

def test_returns_length_is_n_minus_1():
    closes = [100.0, 110.0, 121.0, 108.9]
    series = Series.from_closes("BTC", closes)
    rets = series.returns()
    assert len(rets) == len(closes) - 1


def test_returns_values_hand_verified():
    # 100 -> 110 is +10%; 110 -> 121 is +10%; 121 -> 108.9 is -10%
    closes = [100.0, 110.0, 121.0, 108.9]
    series = Series.from_closes("BTC", closes)
    rets = series.returns()
    expected = [0.10, 0.10, -0.10]
    assert len(rets) == len(expected)
    for got, want in zip(rets, expected):
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9)


def test_returns_single_candle_is_empty():
    series = Series.from_closes("SOL", [42.0])
    assert series.returns() == []


def test_returns_handles_zero_previous_close_without_raising():
    # A degenerate series with a zero close must not raise ZeroDivisionError;
    # contracts.py explicitly guards this with `if prev else 0.0`.
    series = Series(symbol="X", candles=[
        OHLCV(ts=0.0, open=0.0, high=0.0, low=0.0, close=0.0),
        OHLCV(ts=1.0, open=5.0, high=5.0, low=5.0, close=5.0),
    ])
    rets = series.returns()
    assert rets == [0.0]


# ---- Series.last() / __len__ --------------------------------------------------

def test_last_returns_final_candle():
    closes = [10.0, 20.0, 30.0]
    series = Series.from_closes("BTC", closes)
    assert series.last().close == 30.0


def test_len_matches_candle_count():
    series = Series.from_closes("BTC", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert len(series) == 6


# ---- Quote.is_up ---------------------------------------------------------------

def test_quote_is_up_true_for_positive_change():
    q = Quote(symbol="BTC", name="Bitcoin", price=100.0, source="demo",
              change_24h_pct=2.5)
    assert q.is_up is True


def test_quote_is_up_false_for_negative_change():
    q = Quote(symbol="BTC", name="Bitcoin", price=100.0, source="demo",
              change_24h_pct=-0.01)
    assert q.is_up is False


def test_quote_is_up_true_for_exact_zero_change():
    q = Quote(symbol="BTC", name="Bitcoin", price=100.0, source="demo",
              change_24h_pct=0.0)
    assert q.is_up is True


def test_quote_is_up_true_when_change_is_none():
    # Documented behavior: `(self.change_24h_pct or 0.0) >= 0` treats a missing
    # change as 0.0, which counts as "up".
    q = Quote(symbol="BTC", name="Bitcoin", price=100.0, source="demo",
              change_24h_pct=None)
    assert q.is_up is True


def test_quote_default_asset_class_is_other():
    q = Quote(symbol="XYZ", name="Unknown", price=1.0, source="demo")
    assert q.asset_class == AssetClass.OTHER


# ---- test runner (standalone mode) --------------------------------------------

def _run_standalone() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            failed += 1
        else:
            print(f"PASS {name}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_standalone() else 0)
