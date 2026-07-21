"""
Tests for finscope.analytics.engine.AnalyticsEngine.

Every numeric assertion below is hand-verified (not just "does it run"):
  * simple_returns([100, 110, 121]) == [0.1, 0.1]                (10% then 10%)
  * annualized_vol of a constant series == 0.0                    (zero variance)
  * max_drawdown([100, 50, 75]) == -0.5                           (peak 100 -> trough 50)
  * sharpe of a strictly-rising series > 0                        (positive excess return)
  * historical_var negative for a mixed series                    (5th percentile is a loss)
  * correlation of a series with itself == 1.0 on the diagonal
  * sma/ema length checks
  * rsi bounds 0..100

Works both under pytest (functions named test_*) and standalone:
    python3 finscope/tests/test_analytics.py
"""
from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from finscope.analytics.engine import AnalyticsEngine
from finscope.core.contracts import Series


def _isclose(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


# ---- simple_returns -------------------------------------------------------------

def test_simple_returns_hand_verified():
    eng = AnalyticsEngine()
    rets = eng.simple_returns([100, 110, 121])
    assert len(rets) == 2
    assert _isclose(rets[0], 0.1)
    assert _isclose(rets[1], 0.1)


def test_simple_returns_too_short_is_empty():
    eng = AnalyticsEngine()
    assert eng.simple_returns([100]).size == 0
    assert eng.simple_returns([]).size == 0


# ---- annualized_vol -------------------------------------------------------------

def test_annualized_vol_constant_series_is_zero():
    eng = AnalyticsEngine()
    # A perfectly flat price series has zero-variance returns.
    constant_returns = [0.0, 0.0, 0.0, 0.0, 0.0]
    assert eng.annualized_vol(constant_returns) == 0.0

    # Same conclusion reached end-to-end from a constant-price Series.
    series = Series.from_closes("FLAT", [50.0] * 10)
    rets = eng.simple_returns(series.closes())
    assert eng.annualized_vol(rets) == 0.0


def test_annualized_vol_of_short_series_is_zero():
    eng = AnalyticsEngine()
    # Fewer than 2 observations -> engine defines this as 0.0 (can't estimate std).
    assert eng.annualized_vol([0.05]) == 0.0
    assert eng.annualized_vol([]) == 0.0


def test_annualized_vol_positive_for_volatile_series():
    eng = AnalyticsEngine()
    rets = [0.05, -0.03, 0.02, -0.04, 0.01]
    assert eng.annualized_vol(rets) > 0.0


# ---- max_drawdown ----------------------------------------------------------------

def test_max_drawdown_hand_verified():
    eng = AnalyticsEngine()
    # Peak is 100, trough after the peak is 50 -> drawdown = (50-100)/100 = -0.5
    dd = eng.max_drawdown([100, 50, 75])
    assert _isclose(dd, -0.5)


def test_max_drawdown_monotonic_up_is_zero():
    eng = AnalyticsEngine()
    dd = eng.max_drawdown([10, 20, 30, 40])
    assert _isclose(dd, 0.0)


def test_max_drawdown_empty_is_zero():
    eng = AnalyticsEngine()
    assert eng.max_drawdown([]) == 0.0


# ---- sharpe -----------------------------------------------------------------------

def test_sharpe_positive_for_strictly_rising_series():
    eng = AnalyticsEngine()
    closes = list(range(100, 130))  # strictly increasing by a constant step
    rets = eng.simple_returns(closes)
    sh = eng.sharpe(rets)
    assert sh > 0.0


def test_sharpe_zero_when_std_is_zero():
    eng = AnalyticsEngine()
    # Constant returns -> zero standard deviation -> engine returns 0.0 (guards div/0)
    assert eng.sharpe([0.01, 0.01, 0.01, 0.01]) == 0.0


def test_sharpe_short_series_is_zero():
    eng = AnalyticsEngine()
    assert eng.sharpe([0.01]) == 0.0


# ---- historical_var ----------------------------------------------------------------

def test_historical_var_negative_for_mixed_series():
    eng = AnalyticsEngine()
    closes = [100, 105, 95, 110, 90, 120]
    rets = eng.simple_returns(closes)
    var95 = eng.historical_var(rets, confidence=0.95)
    assert var95 < 0.0


def test_historical_var_hand_verified_percentile():
    eng = AnalyticsEngine()
    rets = [-0.05, -0.02, 0.0, 0.01, 0.03]
    # Hand-verified: np.percentile's default linear interpolation puts the 5th
    # percentile of this sorted 5-point sample at index 0.05*(5-1)=0.2, i.e.
    # -0.05 + 0.2*(-0.02 - -0.05) = -0.044. Confirm the engine matches numpy's
    # own (independently documented) formula rather than re-deriving it here.
    expected = -0.044
    got = eng.historical_var(rets, confidence=0.95)
    assert _isclose(got, expected, tol=1e-9)
    assert _isclose(got, float(np.percentile(rets, 5)), tol=1e-9)


def test_historical_var_empty_is_zero():
    eng = AnalyticsEngine()
    assert eng.historical_var([]) == 0.0


# ---- correlation --------------------------------------------------------------------

def test_correlation_of_series_with_itself_is_one_on_diagonal():
    eng = AnalyticsEngine()
    series = Series.from_closes("BTC", [100, 102, 101, 105, 103, 107, 110, 108, 111, 115])
    cm = eng.correlation([series, series])
    assert cm.symbols == ("BTC", "BTC")
    for i in range(len(cm.symbols)):
        assert _isclose(cm.matrix[i][i], 1.0, tol=1e-9)
    # off-diagonal is also 1.0 here since it's literally the same series twice
    assert _isclose(cm.matrix[0][1], 1.0, tol=1e-9)


def test_correlation_pair_helper_matches_matrix():
    eng = AnalyticsEngine()
    # NOTE: mirroring the *prices* (e.g. [8,7,6,5,4,3,2,1] against [1..8]) does
    # NOT give a perfect -1 correlation, because pct-return is a nonlinear
    # transform of price and breaks the linear relationship. Instead construct
    # series B's *returns* to be the exact negation of series A's returns --
    # that is what Pearson correlation actually measures here, so it
    # guarantees corr(A, B) == -1.0 exactly.
    a_closes = [100.0, 110.0, 90.0, 130.0, 80.0, 150.0]
    r_a = eng.simple_returns(a_closes)
    b_closes = [100.0]
    for r in r_a:
        b_closes.append(b_closes[-1] * (1 - r))
    a = Series.from_closes("A", a_closes)
    b = Series.from_closes("B", b_closes)
    cm = eng.correlation([a, b])
    assert _isclose(cm.pair("A", "B"), -1.0, tol=1e-6)
    assert cm.pair("A", "NONEXISTENT") is None


def test_correlation_falls_back_to_identity_with_fewer_than_two_usable_series():
    eng = AnalyticsEngine()
    # A single series shorter than the min-length-3 usability threshold.
    tiny = Series.from_closes("X", [1, 2])
    cm = eng.correlation([tiny])
    assert cm.matrix == ((1.0,),)


# ---- sma / ema ------------------------------------------------------------------------

def test_sma_length_is_n_minus_window_plus_1():
    eng = AnalyticsEngine()
    closes = list(range(1, 20))  # 19 points
    window = 5
    out = eng.sma(closes, window)
    assert len(out) == len(closes) - window + 1


def test_sma_value_hand_verified():
    eng = AnalyticsEngine()
    out = eng.sma([1, 2, 3, 4, 5], window=5)
    assert len(out) == 1
    assert _isclose(out[0], 3.0)  # mean of 1..5


def test_sma_too_short_is_empty():
    eng = AnalyticsEngine()
    assert eng.sma([1, 2], window=5).size == 0


def test_ema_length_matches_input():
    eng = AnalyticsEngine()
    closes = list(range(1, 20))
    out = eng.ema(closes, window=5)
    assert len(out) == len(closes)


def test_ema_first_value_equals_first_close():
    eng = AnalyticsEngine()
    out = eng.ema([10.0, 20.0, 30.0], window=3)
    assert _isclose(out[0], 10.0)


# ---- rsi --------------------------------------------------------------------------------

def test_rsi_bounds_for_monotonic_increase():
    eng = AnalyticsEngine()
    closes = list(range(1, 20))  # all gains, no losses
    rsi = eng.rsi(closes, window=14)
    assert 0.0 <= rsi <= 100.0
    assert _isclose(rsi, 100.0)  # avg_loss == 0 -> RSI defined as exactly 100


def test_rsi_bounds_for_monotonic_decrease():
    eng = AnalyticsEngine()
    closes = list(range(20, 0, -1))  # all losses, no gains
    rsi = eng.rsi(closes, window=14)
    assert 0.0 <= rsi <= 100.0
    assert _isclose(rsi, 0.0)


def test_rsi_bounds_for_mixed_series():
    eng = AnalyticsEngine()
    closes = [10, 12, 9, 15, 8, 20, 7, 25, 6, 30, 5, 35, 4, 40, 3]
    rsi = eng.rsi(closes, window=14)
    assert 0.0 <= rsi <= 100.0


def test_rsi_defaults_to_50_when_insufficient_data():
    eng = AnalyticsEngine()
    # size < window + 1 -> documented neutral default of 50.0
    assert eng.rsi([1, 2, 3], window=14) == 50.0


# ---- risk_metrics() aggregate report ------------------------------------------------------

def test_risk_metrics_aggregate_matches_primitives():
    eng = AnalyticsEngine()
    closes = [100, 110, 121, 108.9, 130, 95]
    series = Series.from_closes("BTC", closes)
    rm = eng.risk_metrics(series)
    assert rm.symbol == "BTC"
    assert rm.n == len(closes)
    assert rm.engine == "numpy"
    assert _isclose(rm.max_drawdown, eng.max_drawdown(closes))
    assert _isclose(rm.annualized_vol, eng.annualized_vol(eng.simple_returns(closes)))


# ---- test runner (standalone mode) --------------------------------------------------------

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
