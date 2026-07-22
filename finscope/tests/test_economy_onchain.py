"""Tests for the economy (governor/treasury/controls) + onchain (DAA bridge/MRAP)."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from finscope.core.contracts import Series
from finscope.economy.governor import BudgetGovernor
from finscope.economy.controls import FundControls
from finscope.economy.treasury import TreasuryManager
from finscope.onchain.daa_bridge import DAABridge
from finscope.onchain.mrap import MRAPLoop


def _gov():
    return BudgetGovernor(cap=10.0, period_seconds=10**9,
                          ledger_path=os.path.join(tempfile.mkdtemp(), "l.json"))


def test_governor_cap_and_gate():
    g = _gov()
    assert g.gate(5.0)["approved"]
    g.spend(9.5, "fable", "big")
    assert g.remaining() == 0.5
    assert not g.gate(1.0)["approved"]          # would exceed cap
    assert g.spend(1.0)["approved"] is False     # not recorded


def test_governor_tier_routing_degrades():
    g = _gov()
    assert g.recommend_tier(units=1)["tier"] in ("fable", "sonnet", "haiku", "local")
    g.spend(10.0, "fable", "exhaust")
    assert g.recommend_tier(units=1)["tier"] == "local"   # broke -> $0 path only


def test_controls_clamp_whitelist_killswitch():
    c = FundControls(max_position=0.01, whitelist=["BTC"])
    assert c.check("BTC", 0.02)["approved"] is False      # over max_position
    assert abs(c.check("BTC", 0.02)["clamp"]) == 0.01
    assert c.check("DOGE", 0.005)["approved"] is False     # not whitelisted
    assert c.check("BTC", 0.005)["approved"] is True
    c.update_pnl(daily_loss=0.05, drawdown=0.0)            # trips daily-loss limit
    assert c.halted
    assert c.check("BTC", 0.005)["approved"] is False       # halted rejects all


def test_controls_live_gate_blocked_by_default():
    c = FundControls()
    os.environ.pop("FINSCOPE_ONCHAIN_LIVE", None)
    assert c.allow_live()["live_allowed"] is False


def test_treasury_respects_gross_and_cap():
    t = TreasuryManager(max_gross_exposure=0.03, controls=FundControls(max_position=0.01))
    allocs = t.allocate([{"symbol": "BTC", "strategy": "m", "size": 1.0},
                         {"symbol": "ETH", "strategy": "m", "size": 1.0}])
    for a in allocs:
        assert abs(a.approved_weight) <= 0.01 + 1e-9        # per-position cap enforced


def test_daa_bridge_paper_and_refuses_live():
    b = DAABridge(controls=FundControls())
    fill = b.submit_order("BTC", 0.005, price=100.0, live=False)
    assert fill["status"] == "filled" and fill["mode"] == "paper"
    os.environ.pop("FINSCOPE_ONCHAIN_LIVE", None)
    live = b.submit_order("BTC", 0.005, price=100.0, live=True)
    assert live["status"] == "refused"                      # never silently goes live


class _FakeHub:
    """Hub returning a fixed driftless synthetic series — hermetic, no lake needed."""
    def __init__(self, closes):
        self._s = Series.from_closes("X", closes)
        self.watchlist = ["X"]

    def history(self, symbol, days=0, **k):
        return self._s

    def quote(self, symbol, **k):
        return None


def test_mrap_no_blowup_on_noise():
    rng = np.random.default_rng(3)
    closes = list(100.0 * np.exp(np.cumsum(rng.standard_normal(200) * 0.02)))
    out = MRAPLoop(hub=_FakeHub(closes)).run("X", warmup=5)
    assert "error" not in out
    assert out["cycles"] > 100
    assert abs(out["total_return"]) < 10.0                  # no equity explosion
    assert abs(out["sharpe"]) < 50.0
    assert out["kill_switch_tripped"] in (True, False)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(1 if failed else 0)
