"""Tests for token-optimizer, omega-crypto bridge, and Fable alpha3 strategies."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from finscope.core.contracts import Series
from finscope.economy.optimize import (compact, estimate_tokens, add_cache_control,
                                       PromptCache, TokenOptimizer)


def test_compact_reduces_and_cleans():
    raw = "Please could you just kindly, in order to, really summarize the market."
    c = compact(raw)
    assert estimate_tokens(c) < estimate_tokens(raw)
    assert ", ," not in c and " ," not in c


def test_cache_control_marks_prefix():
    cc = add_cache_control("SYS", [{"name": "t", "description": "d"}])
    assert cc["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert cc["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_prompt_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        pc = PromptCache(cache_dir=d)
        assert pc.get("m", "sys", [{"a": 1}]) is None
        pc.put("m", "sys", [{"a": 1}], "answer")
        assert pc.get("m", "sys", [{"a": 1}]) == "answer"


def test_token_optimizer_plan():
    os.environ["FINSCOPE_ECONOMY_DIR"] = tempfile.mkdtemp()
    p = TokenOptimizer().plan("You are FinScope.", [{"role": "user", "content": "hi"}])
    for k in ("cached_hit", "recommended_tier", "compaction_ratio", "budget_remaining"):
        assert k in p


def test_omega_strategy_causal():
    from finscope.integrations.omega_crypto import OmegaScoreStrategy, load_omega_params
    p = load_omega_params()
    assert "entry_threshold" in p
    closes = [100, 101, 99, 102, 105, 103, 108, 110, 107, 109, 112, 111, 115, 118]
    s = Series.from_closes("BTC", closes)
    w = OmegaScoreStrategy(params=p).weights(s)
    assert len(w) == len(closes)
    assert np.all(np.isfinite(w)) and np.all(np.abs(w) <= 1.0 + 1e-9)
    # no lookahead: change last close -> earlier weights unchanged
    closes2 = list(closes); closes2[-1] = 999
    w2 = OmegaScoreStrategy(params=p).weights(Series.from_closes("BTC", closes2))
    assert np.allclose(w[:-1], w2[:-1])


def test_alpha3_strategies():
    from finscope.engines import strategies_alpha3 as A
    assert len(A.ALPHA3_STRATEGIES) == 3
    closes = list(100.0 * np.exp(np.cumsum(np.random.default_rng(1).standard_normal(40) * 0.02)))
    s = Series.from_closes("X", closes)
    for name, cls in A.ALPHA3_STRATEGIES.items():
        w = cls().weights(s)
        assert len(w) == len(closes), name
        assert np.all(np.isfinite(w)) and np.all(np.abs(w) <= 1.0 + 1e-9), name
        # causality: perturbing last close leaves earlier weights unchanged
        c2 = list(closes); c2[-1] *= 1.5
        w2 = cls().weights(Series.from_closes("X", c2))
        assert np.allclose(w[:-1], w2[:-1]), f"{name} lookahead"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(1 if failed else 0)
