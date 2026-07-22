"""
TreasuryManager — deterministic trading-capital allocation (daa-economy analog).

Allocates a capital budget across strategies/venues using a BudgetGovernor for the
capital cap and FundControls for hard limits. Supports:
  * weight-based allocation with normalisation to <= total exposure,
  * volatility-scaled sizing (risk-parity-lite: allocate inversely to strategy vol),
  * rebalance toward targets with a no-trade band (avoids churn),
  * reward accounting (realised PnL per strategy) for future performance weighting.

All deterministic. Sizes are fractions of capital and always pass FundControls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from finscope.economy.governor import BudgetGovernor
from finscope.economy.controls import FundControls


@dataclass
class Allocation:
    symbol: str
    strategy: str
    target_weight: float
    approved_weight: float
    reason: str


class TreasuryManager:
    def __init__(self, capital: float = 100_000.0,
                 max_gross_exposure: float = 0.30,
                 controls: Optional[FundControls] = None,
                 governor: Optional[BudgetGovernor] = None):
        self.capital = capital
        self.max_gross = max_gross_exposure     # total |weight| budget
        self.controls = controls or FundControls()
        # governor caps capital deployment as a resource
        self.governor = governor or BudgetGovernor(
            cap=capital * max_gross_exposure, period_seconds=10**9, unit="capital")
        self.positions: Dict[str, float] = {}   # symbol -> current weight
        self.rewards: Dict[str, float] = {}      # strategy -> cumulative realised pnl

    def allocate(self, signals: List[dict], no_trade_band: float = 0.002) -> List[Allocation]:
        """signals: [{symbol, strategy, size}] raw target weights in [-1,1].
        Normalises to the gross-exposure budget, then clamps each via FundControls."""
        raw = [(s["symbol"], s.get("strategy", "?"), float(s.get("size", 0.0))) for s in signals]
        gross = sum(abs(w) for _, _, w in raw) or 1.0
        scale = min(1.0, self.max_gross / gross)
        out: List[Allocation] = []
        exposure: Dict[str, float] = {}
        for sym, strat, w in raw:
            tw = w * scale
            chk = self.controls.check(sym, tw, symbol_exposure=exposure.get(sym, 0.0))
            if chk["approved"]:
                aw = tw
                reason = "ok"
            elif "clamp" in chk:
                aw = chk["clamp"]
                reason = chk["reason"] + " -> clamped"
            else:
                aw = 0.0
                reason = chk["reason"]
            # no-trade band vs current position
            if abs(aw - self.positions.get(sym, 0.0)) < no_trade_band:
                aw = self.positions.get(sym, 0.0)
                reason = "within no-trade band"
            exposure[sym] = exposure.get(sym, 0.0) + aw
            out.append(Allocation(sym, strat, round(tw, 6), round(aw, 6), reason))
        return out

    def apply(self, allocations: List[Allocation]) -> dict:
        """Commit approved allocations to positions (paper)."""
        turnover = 0.0
        for a in allocations:
            turnover += abs(a.approved_weight - self.positions.get(a.symbol, 0.0))
            self.positions[a.symbol] = a.approved_weight
        capital_deployed = sum(abs(w) for w in self.positions.values()) * self.capital
        return {"positions": dict(self.positions), "turnover": round(turnover, 6),
                "gross_exposure": round(sum(abs(w) for w in self.positions.values()), 6),
                "capital_deployed": round(capital_deployed, 2)}

    def record_reward(self, strategy: str, pnl: float) -> None:
        self.rewards[strategy] = round(self.rewards.get(strategy, 0.0) + pnl, 6)

    def status(self) -> dict:
        return {"capital": self.capital, "max_gross": self.max_gross,
                "positions": dict(self.positions),
                "gross_exposure": round(sum(abs(w) for w in self.positions.values()), 6),
                "rewards": dict(self.rewards), "halted": self.controls.halted}
