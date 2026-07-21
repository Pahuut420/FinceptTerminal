"""
NativeEngine — deterministic numpy backtester. Always available, $0.

No lookahead (strategies pre-shift their signals), transaction costs charged on
turnover, and the AutoResearch metric is the *out-of-sample* Sharpe (Sharpe on
the held-out tail). This is the engine the strategy-optimisation loop runs
against so only the mutate step (optional, cheap model) ever costs anything.
"""
from __future__ import annotations

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.base import Engine, BacktestResult, Strategy, register_engine
from finscope.engines import strategies as strat_lib


@register_engine
class NativeEngine(Engine):
    name = "native"
    kind = "backtest"
    description = "Deterministic numpy backtester (no lookahead, tx-cost, OOS Sharpe)."
    ppy = 365  # daily crypto

    def available(self) -> bool:
        return True

    def backtest(self, strategy: Strategy, series: Series, cost_bps: float = 5.0) -> BacktestResult:
        c = np.asarray(series.closes(), dtype=float)
        n = c.size
        if n < 3:
            return BacktestResult(strategy=getattr(strategy, "name", "?"), symbol=series.symbol,
                                  engine=self.name, n_bars=n, total_return=0, ann_return=0,
                                  ann_vol=0, sharpe=0, sortino=0, max_drawdown=0, win_rate=0,
                                  n_trades=0, turnover=0, oos_sharpe=0, verified=False,
                                  notes="insufficient bars")
        rets = np.zeros(n)
        rets[1:] = np.diff(c) / c[:-1]
        w = np.asarray(strategy.weights(series), dtype=float)
        w = np.clip(np.nan_to_num(w), -1.0, 1.0)
        if w.size != n:
            w = np.resize(w, n)
        # turnover cost per bar
        dpos = np.abs(np.diff(np.insert(w, 0, 0.0)))
        cost = dpos * (cost_bps / 1e4)
        pnl = w * rets - cost
        equity = np.cumprod(1.0 + pnl)

        def _sharpe(x: np.ndarray) -> float:
            x = x[np.isfinite(x)]
            sd = np.std(x, ddof=1) if x.size > 1 else 0.0
            return float(np.mean(x) / sd * np.sqrt(self.ppy)) if sd > 0 else 0.0

        def _sortino(x: np.ndarray) -> float:
            x = x[np.isfinite(x)]
            downside = x[x < 0]
            dd = np.std(downside, ddof=1) if downside.size > 1 else 0.0
            return float(np.mean(x) / dd * np.sqrt(self.ppy)) if dd > 0 else 0.0

        peak = np.maximum.accumulate(equity)
        mdd = float(((equity - peak) / peak).min())
        total_ret = float(equity[-1] - 1.0)
        ann_ret = float(np.mean(pnl) * self.ppy)
        ann_vol = float(np.std(pnl, ddof=1) * np.sqrt(self.ppy)) if n > 1 else 0.0
        trades = int((dpos > 1e-9).sum())
        active = pnl[np.abs(w) > 1e-9]
        win_rate = float((active > 0).mean()) if active.size else 0.0

        # out-of-sample: Sharpe on the held-out last 30% (>=3 bars)
        split = max(3, int(n * 0.7))
        oos = pnl[split:] if n - split >= 3 else pnl
        oos_sharpe = _sharpe(oos)

        return BacktestResult(
            strategy=getattr(strategy, "name", "strategy"), symbol=series.symbol,
            engine=self.name, n_bars=n, total_return=round(total_ret, 6),
            ann_return=round(ann_ret, 6), ann_vol=round(ann_vol, 6),
            sharpe=round(_sharpe(pnl), 4), sortino=round(_sortino(pnl), 4),
            max_drawdown=round(mdd, 6), win_rate=round(win_rate, 4),
            n_trades=trades, turnover=round(float(dpos.sum()), 4),
            oos_sharpe=round(oos_sharpe, 4),
            params=dict(getattr(strategy, "params", {}), cost_bps=cost_bps),
            equity_curve=[round(x, 6) for x in equity.tolist()],
            verified=True,
            notes=("small-sample: metrics illustrative until the data lake is filled"
                   if n < 60 else ""),
        )

    # ---- JSON surface for agents ----
    def run(self, request: dict) -> dict:
        op = request.get("op", "describe")
        if op == "describe":
            return self.describe()
        if op in ("backtest", "signal"):
            from finscope.core.datahub import DataHub
            hub = request.get("_hub") or DataHub()
            sym = request.get("symbol", "BTC")
            days = int(request.get("days", 365))
            series = hub.history(sym, days=days)
            if not series or len(series) < 3:
                return {"error": f"no history for {sym}"}
            strat = strat_lib.build(request.get("strategy", "momentum"),
                                    **request.get("params", {}))
            if op == "backtest":
                return self.backtest(strat, series, cost_bps=float(request.get("cost_bps", 5.0))
                                     ).to_dict(with_curve=request.get("with_curve", False))
            return self.signal(strat, series).to_dict()
        return {"error": f"native: unsupported op {op!r}"}
