"""
Alpha strategy library, batch 2 — four novel, fully causal systematic strategies.

Same contract as finscope.engines.strategies: each strategy is a dataclass with
`name`, `params`, and `weights(series) -> np.ndarray` of len(series) target
weights in [-1, 1]. Every signal is computed with information through bar t and
then passed through `_shift1`, so weights[t] only ever uses data up to t-1
(no lookahead). numpy-only; safe on short series (>= 3 bars) without raising.

Families here are deliberately distinct from batch 1 (regime-adaptive ER blend,
skew-tilted TSMOM, KAMA trend): an Ornstein-Uhlenbeck half-life stat-arb, a
Connors RSI(2) short-horizon reversal, a MACD-histogram momentum system with
volatility-scaled sizing, and a Bollinger squeeze volatility-compression
breakout.

Registry: `ALPHA2_STRATEGIES` (name -> class).
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


def _safe_log(c: np.ndarray) -> np.ndarray:
    """Elementwise log with non-positive prices mapped to 0 (neutral)."""
    return np.where(c > 1e-12, np.log(np.where(c > 1e-12, c, 1.0)), 0.0)


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Standard recursive EMA with alpha = 2 / (span + 1); seeded at x[0]."""
    n = x.size
    out = np.zeros(n)
    if n == 0:
        return out
    a = 2.0 / (max(span, 1) + 1.0)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def _rolling_std(x: np.ndarray, i: int, window: int) -> float:
    w = min(window, i + 1)
    if w < 2:
        return 0.0
    return float(np.std(x[i - w + 1:i + 1]))


def _rolling_mean(x: np.ndarray, i: int, window: int) -> float:
    w = min(window, i + 1)
    if w < 1:
        return 0.0
    return float(np.mean(x[i - w + 1:i + 1]))


# ---- 1. Ornstein-Uhlenbeck half-life mean-reversion --------------------------

@dataclass
class OUHalfLifeStrategy:
    """AR(1)/Ornstein-Uhlenbeck mean-reversion, sized inversely to half-life.

    Quant rationale: model the log-price as an OU process
    dx = -kappa * (x - mu) dt + sigma dW. A rolling AR(1) fit of the log-price
    increment on its lagged level (dx_t = a + b * x_{t-1}) estimates the
    reversion speed: kappa = -ln(1 + b), half-life = ln(2) / kappa. Trade only
    when the fitted slope is actually negative (statistical evidence of
    reversion in the current window — the strategy self-disables in trending
    tapes rather than fading a drift), fade the z-score of the log-price
    against its rolling mean, and size *inversely to the estimated half-life*:
    fast reversion earns full conviction because the expected holding period
    is short and the alpha decays quickly, while slow reversion (long
    half-life) gets a small weight because the same entry z carries far more
    inventory risk per unit of expected P&L. This is the classic stat-arb
    sizing discipline applied to a single series.

    References: E. Chan, "Algorithmic Trading: Winning Strategies and Their
    Rationale" (Wiley, 2013), ch. 2-3 (half-life of mean reversion, OU fit);
    Uhlenbeck & Ornstein, "On the Theory of the Brownian Motion" (Phys. Rev.,
    1930).
    """
    window: int = 10
    z_cap: float = 2.0
    hl_ref: float = 5.0  # half-life (bars) at/below which conviction is full
    name: str = "ou_half_life"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"ou_half_life(w={self.window},hl={self.hl_ref})"
        self.params = {"window": self.window, "z_cap": self.z_cap,
                       "hl_ref": self.hl_ref}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n == 0:
            return np.zeros(0)
        x = _safe_log(c)
        sig = np.zeros(n)
        for i in range(3, n):
            w = min(self.window, i)
            if w < 3:
                continue
            seg = x[i - w:i + 1]
            lag, dx = seg[:-1], np.diff(seg)
            lm = lag - lag.mean()
            var = float(np.sum(lm * lm))
            if var <= 1e-18:
                continue
            b = float(np.sum(lm * (dx - dx.mean())) / var)
            if b >= 0.0 or (1.0 + b) <= 1e-12:
                continue  # no measured mean reversion -> stay flat
            kappa = -np.log(1.0 + b)
            half_life = float(np.log(2.0) / max(kappa, 1e-12))
            conviction = float(np.clip(self.hl_ref / max(half_life, 1e-9),
                                       0.0, 1.0))
            mu, sd = float(np.mean(seg)), float(np.std(seg))
            if sd <= 1e-12:
                continue
            z = (x[i] - mu) / sd
            sig[i] = -float(np.clip(z, -self.z_cap, self.z_cap)) \
                / self.z_cap * conviction
        return _finalize(sig)


# ---- 2. Connors RSI(2) short-term reversal -----------------------------------

@dataclass
class ConnorsRSI2Strategy:
    """RSI(2) deep-oversold buy / deep-overbought short with trend filter.

    Quant rationale: at very short horizons equity-like series exhibit strong
    negative autocorrelation — sharp 1-3 bar selloffs tend to snap back. The
    Connors RSI(2) system operationalizes this: a 2-period Wilder RSI collapses
    to an extreme (<10) only after consecutive down closes, marking a
    statistically washed-out state; buying there (and symmetrically shorting
    RSI(2) > 90 spikes) harvests the reversal. A trend filter (price vs. a
    longer SMA) restricts longs to uptrends and shorts to downtrends, which in
    Connors' testing is what separates buying dips in bull regimes from
    catching falling knives. Exits are at mean-reachievement: the position is
    held until RSI(2) crosses back through 50, then flattened — a short
    expected holding period consistent with the horizon of the anomaly.

    References: L. Connors & C. Alvarez, "Short Term Trading Strategies That
    Work" (TradingMarkets, 2009), ch. on the 2-period RSI; J. W. Wilder,
    "New Concepts in Technical Trading Systems" (1978) for the RSI itself.
    """
    rsi_period: int = 2
    lower: float = 10.0
    upper: float = 90.0
    trend_window: int = 10
    name: str = "connors_rsi2"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"connors_rsi2({self.rsi_period},{self.lower:g}/{self.upper:g})"
        self.params = {"rsi_period": self.rsi_period, "lower": self.lower,
                       "upper": self.upper, "trend_window": self.trend_window}

    def _rsi(self, c: np.ndarray) -> np.ndarray:
        """Wilder-smoothed RSI; neutral 50 where undefined."""
        n = c.size
        rsi = np.full(n, 50.0)
        if n < 2:
            return rsi
        delta = np.diff(c)
        p = max(self.rsi_period, 1)
        ag = al = 0.0
        for i in range(1, n):
            g = max(delta[i - 1], 0.0)
            l = max(-delta[i - 1], 0.0)
            if i == 1:
                ag, al = g, l
            else:
                ag = (ag * (p - 1) + g) / p
                al = (al * (p - 1) + l) / p
            denom = ag + al
            rsi[i] = 50.0 if denom <= 1e-12 else 100.0 * ag / denom
        return rsi

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n == 0:
            return np.zeros(0)
        rsi = self._rsi(c)
        trend_ma = np.nan_to_num(_sma(c, self.trend_window), nan=0.0)
        sig = np.zeros(n)
        for i in range(1, n):
            trend = float(np.sign(c[i] - trend_ma[i])) if trend_ma[i] > 0 else 0.0
            prev = sig[i - 1]
            if rsi[i] < self.lower and trend >= 0.0:
                sig[i] = 1.0                      # buy the washed-out dip
            elif rsi[i] > self.upper and trend <= 0.0:
                sig[i] = -1.0                     # short the exhaustion spike
            elif prev > 0.0 and rsi[i] < 50.0:
                sig[i] = prev                     # hold long until RSI mean-reverts
            elif prev < 0.0 and rsi[i] > 50.0:
                sig[i] = prev                     # hold short until RSI mean-reverts
            else:
                sig[i] = 0.0                      # exit at RSI 50 cross
        return _finalize(sig)


# ---- 3. MACD-histogram momentum with vol-scaled sizing -----------------------

@dataclass
class MACDHistogramStrategy:
    """MACD histogram direction, signal-line confirmed, volatility-normalized.

    Quant rationale: the MACD histogram (MACD line minus its signal-line EMA)
    is the second derivative of the smoothed price path — its sign flags
    whether trend momentum is building or fading *before* the slower
    moving-average cross fires. Direction is taken from the histogram's sign
    (equivalently, MACD above/below its signal line); size comes from the
    histogram's magnitude normalized by a rolling ATR proxy (mean absolute
    close-to-close move), so a given histogram reading in a quiet tape carries
    more weight than the same reading in a violent one — this makes the
    position risk-comparable across volatility regimes instead of maximal
    exactly when noise is highest. A fading histogram (shrinking toward zero
    against the position) halves the weight, front-running the cross-back
    rather than waiting for it.

    References: G. Appel, "Technical Analysis: Power Tools for Active
    Investors" (FT Press, 2005) — MACD and histogram; J. W. Wilder, "New
    Concepts in Technical Trading Systems" (1978) — ATR normalization.
    """
    fast: int = 6
    slow: int = 13
    signal: int = 5
    vol_window: int = 8
    vol_mult: float = 3.0  # histogram of vol_mult * ATR-proxy => full size
    name: str = "macd_histogram"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"macd_hist({self.fast},{self.slow},{self.signal})"
        self.params = {"fast": self.fast, "slow": self.slow,
                       "signal": self.signal, "vol_window": self.vol_window,
                       "vol_mult": self.vol_mult}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n == 0:
            return np.zeros(0)
        macd = _ema(c, self.fast) - _ema(c, self.slow)
        hist = macd - _ema(macd, self.signal)
        # ATR proxy from closes (bars here may be o=h=l=c): mean |close move|
        moves = np.abs(np.diff(c, prepend=c[0]))
        sig = np.zeros(n)
        for i in range(2, n):
            direction = float(np.sign(hist[i]))
            if direction == 0.0:
                continue
            atr = _rolling_mean(moves, i, self.vol_window)
            if atr <= 1e-12:
                continue
            magnitude = float(np.clip(abs(hist[i]) / (self.vol_mult * atr),
                                      0.0, 1.0))
            # confirmation: histogram still expanding in the trade direction;
            # a fading histogram front-runs the signal cross -> half size
            expanding = (hist[i] - hist[i - 1]) * direction > 0.0
            sig[i] = direction * magnitude * (1.0 if expanding else 0.5)
        return _finalize(sig)


# ---- 4. Bollinger squeeze volatility-compression breakout --------------------

@dataclass
class SqueezeBreakoutStrategy:
    """Flat inside a volatility squeeze; trade the direction of the release.

    Quant rationale: volatility is cyclical — periods of unusually narrow
    Bollinger bandwidth ("the Squeeze") mark energy-storing consolidations
    that statistically precede directional expansions. The system stays flat
    while relative bandwidth (2k*sigma / MA) sits below a fraction of its own
    recent average (no edge, and breakout direction is unknowable inside the
    coil), then on the release bar — bandwidth expanding back above the
    threshold — enters in the direction price sits relative to the middle
    band, riding the expansion. The position is held while price remains on
    its side of the middle band and cut when it crosses back (trend
    exhaustion), and no re-entry occurs until a fresh squeeze re-forms. Unlike
    a Donchian price-channel breakout, the trigger is a *volatility-regime*
    event, not a price-level event — it keys off band compression, which
    filters out the many channel breaks that occur in already-expanded,
    exhausted tapes.

    References: J. Bollinger, "Bollinger on Bollinger Bands" (McGraw-Hill,
    2001) — the Squeeze and bandwidth; J. Carter, "Mastering the Trade"
    (McGraw-Hill, 2005) — the TTM Squeeze variant.
    """
    window: int = 8
    n_std: float = 2.0
    squeeze_frac: float = 0.75  # bw below this fraction of its avg = squeeze
    bw_lookback: int = 10
    name: str = "squeeze_breakout"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"squeeze_breakout(w={self.window},f={self.squeeze_frac})"
        self.params = {"window": self.window, "n_std": self.n_std,
                       "squeeze_frac": self.squeeze_frac,
                       "bw_lookback": self.bw_lookback}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n == 0:
            return np.zeros(0)
        ma = np.zeros(n)
        bw = np.zeros(n)  # relative Bollinger bandwidth
        for i in range(n):
            w = min(self.window, i + 1)
            seg = c[i - w + 1:i + 1]
            m = float(np.mean(seg))
            s = float(np.std(seg)) if w >= 2 else 0.0
            ma[i] = m
            bw[i] = 2.0 * self.n_std * s / m if abs(m) > 1e-12 else 0.0
        sig = np.zeros(n)
        in_squeeze = False
        for i in range(2, n):
            avg_bw = _rolling_mean(bw, i - 1, self.bw_lookback)  # excl. today
            squeezed_now = avg_bw > 1e-12 and bw[i] < self.squeeze_frac * avg_bw
            prev = sig[i - 1]
            if squeezed_now:
                in_squeeze = True
                sig[i] = 0.0  # flat inside the coil: direction unknowable
            elif in_squeeze:
                # release bar: bandwidth expanding out of a squeeze -> enter
                # in the direction price sits relative to the middle band
                in_squeeze = False
                sig[i] = float(np.sign(c[i] - ma[i]))
            elif prev != 0.0:
                # ride the expansion while price holds its side of the mid-band
                sig[i] = prev if (c[i] - ma[i]) * prev > 0.0 else 0.0
            else:
                sig[i] = 0.0  # no position, no squeeze armed: wait
        return _finalize(sig)


# ---- registry ----------------------------------------------------------------

ALPHA2_STRATEGIES = {
    "ou_half_life": OUHalfLifeStrategy,
    "connors_rsi2": ConnorsRSI2Strategy,
    "macd_histogram": MACDHistogramStrategy,
    "squeeze_breakout": SqueezeBreakoutStrategy,
}
