"""
Alpha strategy library — three novel, fully causal systematic strategies.

Same contract as finscope.engines.strategies: each strategy is a dataclass with
`name`, `params`, and `weights(series) -> np.ndarray` of len(series) target
weights in [-1, 1]. Every signal is computed with information through bar t and
then passed through `_shift1`, so weights[t] only ever uses data up to t-1
(no lookahead). numpy-only; safe on short series (>= 3 bars) without raising.

Registry: `ALPHA_STRATEGIES` (name -> class).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1, _sma


# ---- internal causal helpers -------------------------------------------------

def _finalize(sig: np.ndarray) -> np.ndarray:
    """Sanitize, clip and shift a raw signal into a causal weight vector."""
    return np.clip(np.nan_to_num(_shift1(np.nan_to_num(sig))), -1.0, 1.0)


def _simple_returns(c: np.ndarray) -> np.ndarray:
    """Per-bar simple returns aligned with the close vector (rets[0] = 0)."""
    r = np.zeros_like(c)
    if c.size > 1:
        prev = c[:-1]
        ok = np.abs(prev) > 1e-12
        safe = np.where(ok, prev, 1.0)  # dummy denominator where prev ~ 0
        r[1:] = np.where(ok, (c[1:] - prev) / safe, 0.0)
    return r


def _efficiency_ratio(c: np.ndarray, i: int, window: int) -> float:
    """Kaufman Efficiency Ratio at bar i over up to `window` bars: |net move| /
    sum |bar moves|. 1.0 = perfectly trending path, ~0 = pure noise/chop."""
    w = min(window, i)
    if w < 2:
        return 0.0
    seg = c[i - w:i + 1]
    path = float(np.sum(np.abs(np.diff(seg))))
    if path <= 1e-12:
        return 0.0
    return float(min(1.0, abs(seg[-1] - seg[0]) / path))


def _rolling_std(x: np.ndarray, i: int, window: int) -> float:
    w = min(window, i + 1)
    if w < 2:
        return 0.0
    return float(np.std(x[i - w + 1:i + 1]))


def _sample_skew(x: np.ndarray) -> float:
    """Fisher skewness (m3 / sd^3) with small-sample guards; 0 when undefined."""
    if x.size < 3:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 3))


# ---- 1. regime-adaptive momentum / mean-reversion blend ----------------------

@dataclass
class RegimeAdaptiveStrategy:
    """Efficiency-ratio-gated blend of trend-following and mean-reversion.

    Quant rationale: single-asset return series alternate between trending
    regimes (positive short-horizon autocorrelation, momentum profits) and
    choppy/range-bound regimes (negative autocorrelation, reversal profits).
    Kaufman's Efficiency Ratio (ER = |net price move| / path length) is a cheap,
    robust regime proxy: ER near 1 means the path was directional, ER near 0
    means it was noise. Rather than a hard switch, we hold a continuous convex
    blend: weight = ER * sign(trend) + (1 - ER) * (-z), so the book rotates
    smoothly from trend-following in efficient tapes to fading z-score
    stretches in inefficient ones — capturing both sides of the
    autocorrelation cycle instead of bleeding in the wrong regime.

    Reference: P. Kaufman, "Trading Systems and Methods" (efficiency ratio);
    A. Lo, "The Adaptive Markets Hypothesis" (JPM, 2004) for regime rotation.
    """
    er_window: int = 8
    z_cap: float = 2.0
    name: str = "regime_adaptive"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"regime_adaptive(er={self.er_window})"
        self.params = {"er_window": self.er_window, "z_cap": self.z_cap}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        sig = np.zeros(n)
        for i in range(2, n):
            w = min(self.er_window, i)
            er = _efficiency_ratio(c, i, self.er_window)
            # trend leg over the same horizon the ER was measured on
            trend = float(np.sign(c[i] - c[i - w]))
            # mean-reversion leg: z-score of close vs its own recent window
            win = c[i - w:i + 1]
            mu = float(np.mean(win))
            sd = float(np.std(win))
            z = (c[i] - mu) / sd if sd > 1e-12 else 0.0
            rev = -float(np.clip(z, -self.z_cap, self.z_cap)) / self.z_cap
            sig[i] = er * trend + (1.0 - er) * rev
        return _finalize(sig)


# ---- 2. vol-scaled time-series momentum with downside-skew tilt --------------

@dataclass
class SkewTiltTSMomStrategy:
    """Volatility-targeted time-series momentum, tilted by realized skewness.

    Quant rationale: time-series momentum (sign of the trailing return) is
    robust across assets but suffers violent "momentum crashes" when the
    position's adverse tail fattens. Two overlays address this: (1) inverse-vol
    scaling to a constant risk target — the classic TSMOM construction, which
    alone improves Sharpe by equalizing risk through time; (2) a realized-skew
    tilt — a long position is hurt by negative return skew, a short by positive
    skew, so we scale exposure by 1 + gain * sign(position) * skew, sizing up
    when the recent return distribution's tail favors the position and cutting
    when it points at a crash. The tilt is bounded so it modulates rather than
    dominates the trend signal.

    References: Moskowitz, Ooi & Pedersen, "Time Series Momentum" (JFE, 2012);
    Daniel & Moskowitz, "Momentum Crashes" (JFE, 2016); Amaya, Christoffersen,
    Jacobs & Vasquez, "Does Realized Skewness Predict the Cross-Section of
    Equity Returns?" (JFE, 2015).
    """
    lookback: int = 6
    vol_window: int = 8
    skew_window: int = 10
    target_vol: float = 0.50   # annualized
    skew_gain: float = 0.5
    ann_factor: float = 365.0  # daily crypto bars
    name: str = "skew_tilt_tsmom"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"skew_tilt_tsmom({self.lookback},tv={self.target_vol})"
        self.params = {
            "lookback": self.lookback, "vol_window": self.vol_window,
            "skew_window": self.skew_window, "target_vol": self.target_vol,
            "skew_gain": self.skew_gain, "ann_factor": self.ann_factor,
        }

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        rets = _simple_returns(c)
        sig = np.zeros(n)
        for i in range(2, n):
            lb = min(self.lookback, i)
            base = c[i - lb]
            mom = float(np.sign(c[i] - base)) if abs(base) > 1e-12 else 0.0
            if mom == 0.0:
                continue
            # inverse-vol scaling to the annualized risk target
            rv = _rolling_std(rets, i, self.vol_window) * np.sqrt(self.ann_factor)
            scale = float(np.clip(self.target_vol / max(rv, 1e-9), 0.0, 1.5)) \
                if rv > 1e-9 else 1.0
            # realized-skew tilt: reward tail alignment, punish crash exposure
            sw = min(self.skew_window, i)
            skew = _sample_skew(rets[i - sw + 1:i + 1])
            tilt = float(np.clip(1.0 + self.skew_gain * mom * skew, 0.25, 1.5))
            sig[i] = mom * scale * tilt
        return _finalize(sig)


# ---- 3. KAMA adaptive trend filter with volatility-adaptive deadband ---------

@dataclass
class KamaTrendStrategy:
    """Kaufman Adaptive Moving Average trend filter with an adaptive deadband.

    Quant rationale: fixed-length moving averages face an unresolvable
    trade-off — fast averages whipsaw in chop, slow ones lag the trend. KAMA
    resolves it by making the smoothing constant itself a function of the
    Efficiency Ratio: the average speeds up (toward the `fast` EMA constant)
    when price moves efficiently and grinds to a halt (toward `slow`) in noise,
    so the filter is fast exactly when speed pays and slow when it would churn.
    Positions are taken only when price clears the KAMA by a volatility-scaled
    deadband (a multiple of the rolling std of KAMA increments — Kaufman's own
    filter rule), with hysteresis inside the band: hold the prior stance rather
    than flip-flop. This converts an adaptive smoother into a low-turnover,
    whipsaw-resistant trend system.

    Reference: P. Kaufman, "Smarter Trading" (1995) — KAMA and the
    filtered-entry rule; also "Trading Systems and Methods", ch. on adaptive
    techniques.
    """
    er_window: int = 6
    fast: int = 2
    slow: int = 12
    filter_window: int = 8
    filter_mult: float = 1.0
    name: str = "kama_trend"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"kama_trend(er={self.er_window},{self.fast},{self.slow})"
        self.params = {
            "er_window": self.er_window, "fast": self.fast, "slow": self.slow,
            "filter_window": self.filter_window, "filter_mult": self.filter_mult,
        }

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n == 0:
            return np.zeros(0)
        sc_fast = 2.0 / (self.fast + 1.0)
        sc_slow = 2.0 / (self.slow + 1.0)
        kama = np.zeros(n)
        kama[0] = c[0]
        for i in range(1, n):
            er = _efficiency_ratio(c, i, self.er_window)
            sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
            kama[i] = kama[i - 1] + sc * (c[i] - kama[i - 1])
        dk = np.abs(np.diff(kama, prepend=kama[0]))
        sig = np.zeros(n)
        for i in range(2, n):
            band = self.filter_mult * _rolling_std(dk, i, self.filter_window)
            if c[i] > kama[i] + band:
                sig[i] = 1.0
            elif c[i] < kama[i] - band:
                sig[i] = -1.0
            else:
                sig[i] = sig[i - 1]  # hysteresis: hold stance inside the band
        return _finalize(sig)


# ---- registry ----------------------------------------------------------------

ALPHA_STRATEGIES = {
    "regime_adaptive": RegimeAdaptiveStrategy,
    "skew_tilt_tsmom": SkewTiltTSMomStrategy,
    "kama_trend": KamaTrendStrategy,
}
