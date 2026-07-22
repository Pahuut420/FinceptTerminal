"""
MRAPLoop — Monitor → Reason → Act → Reflect → Adapt (the DAA autonomy cycle).

A deterministic walk-forward paper-trading loop over the lake's history:

  Monitor  read price history up to bar t
  Reason   compute each strategy's causal signal; form an adaptive ensemble
  Act      TreasuryManager sizes it, FundControls clamps it, DAABridge paper-fills
  Reflect  realise PnL on the next bar; update equity, drawdown, kill-switch
  Adapt    reweight the ensemble toward strategies with better trailing performance

No lookahead (signals are causal), costs charged on turnover, and the kill-switch
trips on the configured daily-loss / drawdown limits. Runs entirely on stored data
— $0, reproducible, and the same wiring an operator would flip to live via DAABridge.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional

from finscope.core.datahub import DataHub
from finscope.engines import get_engine
from finscope.engines import strategies as strat_lib
from finscope.economy.treasury import TreasuryManager
from finscope.economy.controls import FundControls
from finscope.onchain.daa_bridge import DAABridge


def _build(name: str):
    if name in strat_lib.STRATEGIES:
        return strat_lib.build(name)
    from finscope.research.optimizer import _build_any
    return _build_any(name, {})


class MRAPLoop:
    def __init__(self, hub: Optional[DataHub] = None,
                 treasury: Optional[TreasuryManager] = None,
                 controls: Optional[FundControls] = None,
                 bridge: Optional[DAABridge] = None,
                 cost_bps: float = 5.0):
        self.hub = hub or DataHub()
        self.controls = controls or FundControls()
        self.treasury = treasury or TreasuryManager(controls=self.controls)
        self.bridge = bridge or DAABridge(controls=self.controls)
        self.engine = get_engine("native")
        self.cost_bps = cost_bps

    def run(self, symbol: str = "BTC", strategies: Optional[List[str]] = None,
            warmup: int = 5, adapt_halflife: int = 6) -> dict:
        strategies = strategies or ["skew_tilt_tsmom", "momentum", "breakout"]
        series = self.hub.history(symbol, days=3650)
        if not series or len(series) < warmup + 3:
            return {"error": f"insufficient history for {symbol} "
                             f"(have {len(series) if series else 0} bars); fill the lake"}
        closes = np.asarray(series.closes(), dtype=float)
        rets = np.zeros(len(closes)); rets[1:] = np.diff(closes) / closes[:-1]

        # precompute each strategy's causal weight vector
        wmats = {}
        for sname in strategies:
            try:
                wmats[sname] = np.clip(np.nan_to_num(_build(sname).weights(series)), -1, 1)
            except Exception:
                continue
        if not wmats:
            return {"error": "no strategies built"}

        alpha = 2.0 / (adapt_halflife + 1.0)
        perf = {s: 0.0 for s in wmats}       # trailing EWMA of per-strategy pnl (Adapt)
        equity = [1.0]
        pos = 0.0
        fills, cycles = [], 0
        killed_at = None

        for t in range(warmup, len(closes) - 1):
            # --- Adapt: ensemble weights from trailing performance (softmax over perf) ---
            names = list(wmats)
            scores = np.array([perf[s] for s in names])
            ex = np.exp(scores - scores.max()) if scores.size else np.array([1.0])
            ens_w = ex / ex.sum()
            # --- Reason: ensemble target weight for bar t (causal) ---
            target = float(sum(ens_w[i] * wmats[names[i]][t] for i in range(len(names))))
            # --- Act: treasury sizes + controls clamp + paper fill ---
            allocs = self.treasury.allocate(
                [{"symbol": symbol, "strategy": "ensemble", "size": target}])
            approved = allocs[0].approved_weight if allocs else 0.0
            fill = self.bridge.submit_order(symbol, approved, price=float(closes[t]), live=False)
            if fill.get("status") == "filled":
                fills.append(fill)
            # --- Reflect: realise pnl on next bar, update equity + risk, maybe kill ---
            turnover = abs(approved - pos)
            # defensive: clamp a single-bar return so bad/illiquid data can't blow up
            # the compounding equity (a >50% one-bar move is a data fault, not alpha)
            r_next = float(np.clip(rets[t + 1], -0.5, 0.5))
            pnl = approved * r_next - turnover * (self.cost_bps / 1e4)
            pos = approved if not self.controls.halted else 0.0
            equity.append(equity[-1] * (1.0 + pnl))
            peak = max(equity)
            dd = (peak - equity[-1]) / peak
            self.controls.update_pnl(daily_loss=max(0.0, -pnl), drawdown=dd)
            # per-strategy reward for Adapt (their own signal * next return)
            for s in names:
                perf[s] = (1 - alpha) * perf[s] + alpha * (wmats[s][t] * r_next)
            self.treasury.apply(allocs)
            cycles += 1
            if self.controls.halted and killed_at is None:
                killed_at = t

        eq = np.array(equity)
        pnl_series = np.diff(eq) / eq[:-1] if eq.size > 1 else np.array([0.0])
        sd = float(np.std(pnl_series, ddof=1)) if pnl_series.size > 1 else 0.0
        sharpe = float(np.mean(pnl_series) / sd * np.sqrt(365)) if sd > 0 else 0.0
        peak = np.maximum.accumulate(eq)
        maxdd = float(((eq - peak) / peak).min())
        return {
            "symbol": symbol, "cycles": cycles, "strategies": list(wmats),
            "total_return": round(float(eq[-1] - 1.0), 6),
            "sharpe": round(sharpe, 4), "max_drawdown": round(maxdd, 6),
            "fills": len(fills), "kill_switch_tripped": killed_at is not None,
            "killed_at_bar": killed_at,
            "final_ensemble_leader": max(perf, key=perf.get) if perf else None,
            "treasury": self.treasury.status(),
            "controls_halted": self.controls.halted,
            "note": ("small-sample — fill the lake (finscope.lake synth/ingest) for "
                     "meaningful magnitudes" if len(closes) < 60 else ""),
        }
