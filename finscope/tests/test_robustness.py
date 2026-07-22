"""
Robustness study harness — smoke + invariant guard.

Keeps the study fast in CI by running a tiny subset (2 strategies, few seeds/bars)
while asserting the properties that matter:
  * it produces ranked rows with every regime scored,
  * results are deterministic (same seeds -> same numbers),
  * the null-regime score for a known-causal strategy stays modest (the harness
    isn't manufacturing edge from noise).
"""
from __future__ import annotations

import numpy as np

from finscope.research import robustness as rob


def test_run_study_shape_and_ranking():
    rows = rob.run_study(seeds=4, bars=180, base_seed=1000,
                         names=["momentum", "mean_reversion"])
    assert len(rows) == 2
    for r in rows:
        assert set(r.per_regime_mean) == set(rob.REGIMES)
        assert set(r.per_regime_pospct) == set(rob.REGIMES)
        assert 0.0 <= r.structured_pospct <= 1.0
        for v in r.per_regime_mean.values():
            assert np.isfinite(v)
    # sorted by structured_mean descending
    assert rows[0].structured_mean >= rows[1].structured_mean


def test_deterministic():
    a = rob.run_study(seeds=4, bars=180, base_seed=42, names=["momentum"])
    b = rob.run_study(seeds=4, bars=180, base_seed=42, names=["momentum"])
    assert a[0].structured_mean == b[0].structured_mean
    assert a[0].per_regime_mean == b[0].per_regime_mean


def test_null_regime_modest_for_causal_strategy():
    """A causal strategy must not post a large positive Sharpe on driftless noise."""
    rows = rob.run_study(seeds=12, bars=365, base_seed=2000, names=["momentum"])
    null_mean = rows[0].per_regime_mean["null"]
    assert null_mean < 0.8, f"momentum null OOS Sharpe {null_mean} — lookahead/overfit smell"
