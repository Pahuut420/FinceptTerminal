"""
Ruliology strategy suite — regression + causality guard.

The 40 Ruliology strategies (finscope/engines/ruliology/rul_*.py) derive signals
from Stephen Wolfram's computational universe (elementary cellular automata, rule
classes, computational irreducibility, the ruliad). They are authored by LLM
agents, so this test locks in the two properties that make them trustworthy and
that a future edit could silently break:

  1. Protocol/aggregation — every module registers into RULIOLOGY_STRATEGIES and
     every strategy is wired into the optimiser's search grid.
  2. Causality — bounded weights in [-1, 1], finite, correct length, and NO
     lookahead (a strategy's weight at bar t may not change when a *future* close
     is perturbed; equivalently, weights on a truncated series equal the prefix
     of weights on the full series).

The null-test property (≈0 OOS Sharpe on driftless synthetic) is a statistical,
multi-seed check run in research tooling, not asserted here — but no-lookahead,
verified deterministically below, is the structural precondition for it.
"""
from __future__ import annotations

import numpy as np

from finscope.core.contracts import Series, OHLCV
from finscope.engines.ruliology import RULIOLOGY_STRATEGIES
from finscope.research.optimizer import DEFAULT_GRIDS


def _series(prices, symbol="T"):
    candles = [
        OHLCV(ts=1_600_000_000 + i * 86400,
              open=float(prices[i - 1] if i > 0 else prices[0]),
              high=float(prices[i]), low=float(prices[i]),
              close=float(prices[i]), volume=1.0)
        for i in range(len(prices))
    ]
    return Series(symbol=symbol, candles=candles)


def _synthetic(n=60, seed=7):
    rng = np.random.RandomState(seed)
    return 100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, n))


def test_registry_nonempty_and_wired():
    assert len(RULIOLOGY_STRATEGIES) >= 40, "expected the full Ruliology batch"
    for name in RULIOLOGY_STRATEGIES:
        assert name in DEFAULT_GRIDS, f"{name} missing from optimiser search grid"


def test_all_build_bounded_and_finite():
    ser = _series(_synthetic())
    for name, cls in RULIOLOGY_STRATEGIES.items():
        w = np.asarray(cls().weights(ser))
        assert len(w) == len(ser), f"{name}: length mismatch"
        assert np.all(np.isfinite(w)), f"{name}: non-finite weight"
        assert np.all(np.abs(w) <= 1.0 + 1e-9), f"{name}: weight outside [-1, 1]"


def test_no_lookahead_last_close_perturbation():
    """Perturbing only the final close must not change any earlier weight."""
    px = _synthetic()
    px2 = px.copy()
    px2[-1] *= 1.5
    for name, cls in RULIOLOGY_STRATEGIES.items():
        w = np.asarray(cls().weights(_series(px)))
        w2 = np.asarray(cls().weights(_series(px2)))
        assert np.allclose(w[:-1], w2[:-1], atol=1e-9), f"{name}: lookahead detected"


def test_no_lookahead_truncation_invariance():
    """weights(px[:k]) must equal weights(px)[:k] for every prefix length k."""
    px = _synthetic(n=48)
    for name, cls in RULIOLOGY_STRATEGIES.items():
        full = np.asarray(cls().weights(_series(px)))
        for k in (12, 24, 36, 47):
            pref = np.asarray(cls().weights(_series(px[:k])))
            assert np.allclose(pref, full[:k], atol=1e-9), \
                f"{name}: truncation mismatch at k={k}"


def test_short_series_safe():
    """A degenerate 3-bar series must not raise and must stay bounded."""
    ser = _series([100.0, 101.0, 100.5])
    for name, cls in RULIOLOGY_STRATEGIES.items():
        w = np.asarray(cls().weights(ser))
        assert len(w) == 3 and np.all(np.isfinite(w)), f"{name}: unsafe on short series"
