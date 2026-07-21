"""
AdvancedRisk — numpy quant primitives that extend AnalyticsEngine with risk
metrics Bloomberg-style terminals expect: beta, Sortino, Calmar, downside
deviation, rolling volatility, parametric VaR, expected shortfall, drawdown
series, and a Hurst-exponent estimate.

Consumes `Series` objects (see finscope.core.contracts) so it slots in next
to AnalyticsEngine and WolframBridge without touching either file.

Conventions (mirrors engine.py):
  * returns are simple period-over-period unless noted
  * annualization uses `periods_per_year` (default 365 for daily crypto)
  * risk-free / target rate defaults to 0
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from finscope.core.contracts import Series


class AdvancedRisk:
    def __init__(self, periods_per_year: int = 365, risk_free: float = 0.0):
        self.ppy = periods_per_year
        self.rf = risk_free

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _returns(series: Series) -> np.ndarray:
        closes = series.closes()
        a = np.asarray(closes, dtype=float)
        if a.size < 2:
            return np.array([])
        return np.diff(a) / a[:-1]

    @staticmethod
    def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Align two return vectors to the shortest common length, taking the
        most recent observations from each (mirrors AnalyticsEngine.correlation)."""
        m = min(a.size, b.size)
        if m == 0:
            return a[:0], b[:0]
        return a[-m:], b[-m:]

    # ---- metrics ------------------------------------------------------------
    def beta(self, series: Series, benchmark: Series) -> float:
        """OLS beta of `series` returns regressed on `benchmark` returns,
        aligned to the shortest common length."""
        r = self._returns(series)
        b = self._returns(benchmark)
        r, b = self._align(r, b)
        if r.size < 2:
            return 0.0
        var_b = np.var(b, ddof=1)
        if var_b == 0:
            return 0.0
        cov = np.cov(r, b, ddof=1)[0, 1]
        return float(cov / var_b)

    def downside_deviation(self, series: Series, target: float = 0.0) -> float:
        """Annualized downside deviation of returns below `target` (per-period)."""
        r = self._returns(series)
        if r.size < 2:
            return 0.0
        downside = np.minimum(r - target, 0.0)
        # mean of squared downside deviations (population-style, standard
        # Sortino convention), then annualize the resulting per-period sigma.
        dd = math.sqrt(float(np.mean(np.square(downside))))
        return dd * math.sqrt(self.ppy)

    def sortino(self, series: Series, target: float = 0.0) -> float:
        """Annualized Sortino ratio: (annualized return - target*ppy) / downside deviation."""
        r = self._returns(series)
        if r.size < 2:
            return 0.0
        ann_return = float(np.mean(r)) * self.ppy
        dd = self.downside_deviation(series, target=target)
        if dd == 0:
            return 0.0
        return (ann_return - target * self.ppy) / dd

    @staticmethod
    def drawdown_series(series: Series) -> List[float]:
        """Running drawdown from peak at every point (<=0, e.g. -0.42 = -42%)."""
        closes = series.closes()
        a = np.asarray(closes, dtype=float)
        if a.size == 0:
            return []
        peak = np.maximum.accumulate(a)
        dd = np.where(peak != 0, (a - peak) / peak, 0.0)
        return [float(x) for x in dd]

    def calmar(self, series: Series) -> float:
        """Annualized return / |max drawdown|. 0.0 if there's no drawdown or
        no usable return history."""
        r = self._returns(series)
        if r.size == 0:
            return 0.0
        ann_return = float(np.mean(r)) * self.ppy
        dd = self.drawdown_series(series)
        max_dd = min(dd) if dd else 0.0
        if max_dd == 0:
            return 0.0
        return ann_return / abs(max_dd)

    def rolling_vol(self, series: Series, window: int = 14) -> List[float]:
        """Annualized rolling volatility (stdev of returns) over `window`
        periods. Returns one value per window-sized slice of the returns
        vector (length = len(returns) - window + 1), empty if too short."""
        r = self._returns(series)
        n = r.size
        if n < window or window < 2:
            return []
        out = np.empty(n - window + 1, dtype=float)
        for i in range(out.size):
            out[i] = np.std(r[i:i + window], ddof=1) * math.sqrt(self.ppy)
        return [float(x) for x in out]

    def parametric_var(self, series: Series, confidence: float = 0.95) -> float:
        """Gaussian (parametric) VaR of 1-period returns, as a negative
        fraction (e.g. -0.05 = -5%). Uses the inverse-normal CDF via
        `math.erf`-derived erfinv approximation-free closed form through
        `np` is unavailable, so we use the standard z-scores lookup for the
        common confidence levels with a normal-approximation fallback."""
        r = self._returns(series)
        if r.size < 2:
            return 0.0
        mu = float(np.mean(r))
        sd = float(np.std(r, ddof=1))
        z = self._z_score(confidence)
        return mu - z * sd

    @staticmethod
    def _z_score(confidence: float) -> float:
        """Inverse standard-normal CDF (quantile function) via Acklam's
        rational approximation — stdlib/numpy only, no scipy dependency."""
        p = 1.0 - confidence
        if p <= 0.0:
            p = 1e-12
        if p >= 1.0:
            p = 1 - 1e-12
        # Peter Acklam's algorithm for the inverse normal CDF.
        a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
        b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01)
        c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
        d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00)
        p_low = 0.02425
        p_high = 1 - p_low
        if p < p_low:
            q = math.sqrt(-2 * math.log(p))
            x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        elif p <= p_high:
            q = p - 0.5
            r_ = q * q
            x = (((((a[0] * r_ + a[1]) * r_ + a[2]) * r_ + a[3]) * r_ + a[4]) * r_ + a[5]) * q / \
                (((((b[0] * r_ + b[1]) * r_ + b[2]) * r_ + b[3]) * r_ + b[4]) * r_ + 1)
        else:
            q = math.sqrt(-2 * math.log(1 - p))
            x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        # x is the (1-confidence)-quantile of the standard normal, i.e. a
        # negative number for confidence > 0.5. VaR convention wants
        # mu - z*sd to be the *loss* threshold, so z = -x (positive z).
        return -x

    @staticmethod
    def expected_shortfall(series: Series, confidence: float = 0.95) -> float:
        """Historical Expected Shortfall / CVaR: mean of returns at or below
        the (1-confidence) percentile (the average tail loss), as a fraction."""
        r = AdvancedRisk._returns(series)
        if r.size == 0:
            return 0.0
        cutoff = float(np.percentile(r, (1 - confidence) * 100))
        tail = r[r <= cutoff]
        if tail.size == 0:
            tail = np.sort(r)[:max(1, int(round(r.size * (1 - confidence))))]
        return float(np.mean(tail))

    @staticmethod
    def hurst_exponent(series: Series) -> float:
        """Rescaled-range (R/S) estimate of the Hurst exponent on the close
        series. H ~ 0.5 = random walk, H > 0.5 = trending/persistent,
        H < 0.5 = mean-reverting. Returns 0.5 (random-walk default) when
        there isn't enough history to estimate reliably."""
        closes = series.closes()
        a = np.asarray(closes, dtype=float)
        n = a.size
        if n < 20:
            return 0.5

        # Use log returns as the increment series (standard R/S convention).
        log_ret = np.diff(np.log(a))
        n = log_ret.size
        if n < 20:
            return 0.5

        max_k = int(math.log2(n))
        lags = [k for k in (2 ** i for i in range(2, max_k + 1)) if k <= n]
        if len(lags) < 2:
            return 0.5

        rs_values: List[float] = []
        valid_lags: List[int] = []
        for lag in lags:
            n_chunks = n // lag
            if n_chunks < 1:
                continue
            rs_chunk: List[float] = []
            for i in range(n_chunks):
                chunk = log_ret[i * lag:(i + 1) * lag]
                mean = np.mean(chunk)
                dev = np.cumsum(chunk - mean)
                r_range = float(np.max(dev) - np.min(dev))
                s = float(np.std(chunk, ddof=0))
                if s > 0:
                    rs_chunk.append(r_range / s)
            if rs_chunk:
                rs_values.append(float(np.mean(rs_chunk)))
                valid_lags.append(lag)

        if len(rs_values) < 2:
            return 0.5

        log_lags = np.log(valid_lags)
        log_rs = np.log(rs_values)
        # slope of log(R/S) vs log(lag) via least-squares — the Hurst exponent.
        slope, _intercept = np.polyfit(log_lags, log_rs, 1)
        h = float(slope)
        # Clamp to the theoretically meaningful [0, 1] band; noisy short
        # samples can occasionally push the OLS fit slightly outside it.
        return max(0.0, min(1.0, h))

    # ---- Wolfram Language query generators (mirrors wolfram_bridge.py) -----
    @staticmethod
    def _wl_list(xs: Sequence[float]) -> str:
        return "{" + ", ".join(repr(float(x)) for x in xs) + "}"

    def wl_sortino(self, series: Series, target: float = 0.0) -> str:
        """WL source computing the annualized Sortino ratio for this series'
        returns. An orchestrating agent can run this through the Wolfram MCP
        (WolframLanguageEvaluator) for a verified/symbolic cross-check."""
        r = series.returns()
        return (
            f"With[{{r = {self._wl_list(r)}, t = {target!r}, ppy = {self.ppy}}}, "
            f"Module[{{downside}}, "
            f"downside = Min[# - t, 0] & /@ r; "
            f"With[{{dd = Sqrt[Mean[downside^2]]*Sqrt[ppy]}}, "
            f"If[dd == 0, 0, ((Mean[r]*ppy) - t*ppy)/dd]]]]"
        )

    def wl_beta(self, series: Series, benchmark: Series) -> str:
        """WL source computing OLS beta of `series` returns vs `benchmark`
        returns, aligned to the shortest common length."""
        r = series.returns()
        b = benchmark.returns()
        return (
            f"Module[{{r = {self._wl_list(r)}, b = {self._wl_list(b)}, m}}, "
            f"m = Min[Length[r], Length[b]]; "
            f"With[{{ra = Take[r, -m], ba = Take[b, -m]}}, "
            f"If[Variance[ba] == 0, 0, Covariance[ra, ba]/Variance[ba]]]]"
        )
