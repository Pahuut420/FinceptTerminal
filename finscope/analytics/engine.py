"""
AnalyticsEngine — numpy quant primitives over FinScope Series.

Everything here is deterministic and offline. The Wolfram bridge (see
wolfram_bridge.py) can substitute a symbolic/verified backend for a few of these,
but the numpy path is always the fallback so the terminal works with no network
and no MCP.

Conventions:
  * returns are simple period-over-period unless noted
  * annualization uses `periods_per_year` (default 365 for daily crypto)
  * risk-free rate defaults to 0
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from finscope.core.contracts import Series, RiskMetrics, CorrelationMatrix


class AnalyticsEngine:
    def __init__(self, periods_per_year: int = 365, risk_free: float = 0.0):
        self.ppy = periods_per_year
        self.rf = risk_free

    # ---- primitives -------------------------------------------------------
    @staticmethod
    def simple_returns(closes: Sequence[float]) -> np.ndarray:
        a = np.asarray(closes, dtype=float)
        if a.size < 2:
            return np.array([])
        return np.diff(a) / a[:-1]

    @staticmethod
    def log_returns(closes: Sequence[float]) -> np.ndarray:
        a = np.asarray(closes, dtype=float)
        if a.size < 2:
            return np.array([])
        return np.diff(np.log(a))

    def annualized_vol(self, returns: Sequence[float]) -> float:
        r = np.asarray(returns, dtype=float)
        if r.size < 2:
            return 0.0
        return float(np.std(r, ddof=1) * math.sqrt(self.ppy))

    def annualized_return(self, returns: Sequence[float]) -> float:
        r = np.asarray(returns, dtype=float)
        if r.size == 0:
            return 0.0
        return float(np.mean(r) * self.ppy)

    def sharpe(self, returns: Sequence[float]) -> float:
        r = np.asarray(returns, dtype=float)
        if r.size < 2:
            return 0.0
        sd = np.std(r, ddof=1)
        if sd == 0:
            return 0.0
        excess = np.mean(r) - self.rf / self.ppy
        return float(excess / sd * math.sqrt(self.ppy))

    @staticmethod
    def max_drawdown(closes: Sequence[float]) -> float:
        a = np.asarray(closes, dtype=float)
        if a.size == 0:
            return 0.0
        peak = np.maximum.accumulate(a)
        dd = (a - peak) / peak
        return float(dd.min())  # negative number, e.g. -0.42

    @staticmethod
    def historical_var(returns: Sequence[float], confidence: float = 0.95) -> float:
        r = np.asarray(returns, dtype=float)
        if r.size == 0:
            return 0.0
        return float(np.percentile(r, (1 - confidence) * 100))

    @staticmethod
    def sma(closes: Sequence[float], window: int) -> np.ndarray:
        a = np.asarray(closes, dtype=float)
        if a.size < window or window <= 0:
            return np.array([])
        c = np.cumsum(np.insert(a, 0, 0.0))
        return (c[window:] - c[:-window]) / window

    @staticmethod
    def ema(closes: Sequence[float], window: int) -> np.ndarray:
        a = np.asarray(closes, dtype=float)
        if a.size == 0 or window <= 0:
            return np.array([])
        alpha = 2.0 / (window + 1.0)
        out = np.empty_like(a)
        out[0] = a[0]
        for i in range(1, a.size):
            out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
        return out

    @staticmethod
    def rsi(closes: Sequence[float], window: int = 14) -> float:
        a = np.asarray(closes, dtype=float)
        if a.size < window + 1:
            return 50.0
        delta = np.diff(a)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.mean(gains[-window:])
        avg_loss = np.mean(losses[-window:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    # ---- aggregate reports ------------------------------------------------
    def risk_metrics(self, series: Series) -> RiskMetrics:
        closes = series.closes()
        rets = self.simple_returns(closes)
        return RiskMetrics(
            symbol=series.symbol,
            n=len(closes),
            mean_return=float(np.mean(rets)) if rets.size else 0.0,
            daily_vol=float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0,
            annualized_vol=self.annualized_vol(rets),
            annualized_return=self.annualized_return(rets),
            sharpe=self.sharpe(rets),
            max_drawdown=self.max_drawdown(closes),
            var_95=self.historical_var(rets, 0.95),
            engine="numpy",
        )

    def correlation(self, series_list: List[Series]) -> CorrelationMatrix:
        """Pearson correlation of aligned return series. Series are truncated to
        the shortest common length (aligned from the most recent observation)."""
        usable = [s for s in series_list if len(s) >= 3]
        if len(usable) < 2:
            syms = tuple(s.symbol for s in series_list)
            n = len(syms)
            eye = tuple(tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n))
            return CorrelationMatrix(symbols=syms, matrix=eye, engine="numpy")
        rets = [self.simple_returns(s.closes()) for s in usable]
        m = min(r.size for r in rets)
        aligned = np.vstack([r[-m:] for r in rets])
        cm = np.corrcoef(aligned)
        cm = np.nan_to_num(cm, nan=0.0)
        return CorrelationMatrix(
            symbols=tuple(s.symbol for s in usable),
            matrix=tuple(tuple(float(x) for x in row) for row in cm),
            engine="numpy",
        )

    def monte_carlo_var(self, series: Series, horizon: int = 1, sims: int = 10000,
                        confidence: float = 0.95, seed: int = 7) -> float:
        """Parametric Monte-Carlo VaR over `horizon` periods (fraction of value)."""
        rets = self.simple_returns(series.closes())
        if rets.size < 2:
            return 0.0
        rng = np.random.default_rng(seed)
        mu, sd = float(np.mean(rets)), float(np.std(rets, ddof=1))
        draws = rng.normal(mu, sd, size=(sims, horizon)).sum(axis=1)
        return float(np.percentile(draws, (1 - confidence) * 100))
