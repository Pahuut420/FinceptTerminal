"""
Alpha strategy library, batch 3 — three novel, fully causal systematic strategies
drawn from information theory, fractal scaling analysis, and spectral estimation.

Same contract as finscope.engines.strategies: each strategy is a dataclass with
`name`, `params`, and `weights(series) -> np.ndarray` of len(series) target
weights in [-1, 1]. Every signal is computed with information through bar t and
then passed through `_shift1`, so weights[t] only ever uses data up to t-1
(no lookahead). numpy-only (entropy, DFA and the DFT cycle fit are implemented
from scratch); safe on short series (>= 3 bars) without raising.

Families here are deliberately distinct from batch 1 (regime-adaptive ER blend,
skew-tilted TSMOM, KAMA trend) and batch 2 (OU half-life, Connors RSI(2),
MACD histogram, Bollinger squeeze):

1. perm_entropy_trend — ordinal-pattern (permutation) entropy regime filter:
   trade trend only when the return stream's symbolic complexity is low.
2. dfa_hurst          — detrended-fluctuation-analysis Hurst exponent:
   follow persistent tapes, fade anti-persistent ones, flat at H ~ 0.5.
3. spectral_cycle     — dominant-cycle phase from a windowed DFT:
   position equals the one-step-ahead slope of the dominant sinusoid, gated
   by spectral concentration so pure-noise spectra produce no position.

Registry: `ALPHA3_STRATEGIES` (name -> class).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1, _sma  # noqa: F401  (_sma kept for parity with sibling modules)


# ---- internal causal helpers -------------------------------------------------

def _finalize(sig: np.ndarray) -> np.ndarray:
    """Sanitize, clip and shift a raw signal into a causal weight vector."""
    return np.clip(np.nan_to_num(_shift1(np.nan_to_num(sig))), -1.0, 1.0)


def _aligned_returns(c: np.ndarray) -> np.ndarray:
    """Per-bar simple returns aligned with the close vector (rets[0] = 0)."""
    r = np.zeros_like(c)
    if c.size > 1:
        prev = c[:-1]
        ok = np.abs(prev) > 1e-12
        safe = np.where(ok, prev, 1.0)
        r[1:] = np.where(ok, (c[1:] - prev) / safe, 0.0)
    return r


def _lin_detrend(y: np.ndarray) -> np.ndarray:
    """Residuals of an OLS straight-line fit (closed form, no polyfit warnings)."""
    n = y.size
    if n < 2:
        return np.zeros_like(y)
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    den = float(np.sum(tc * tc))
    if den <= 1e-18:
        return y - float(np.mean(y))
    slope = float(np.sum(tc * (y - y.mean())) / den)
    intercept = float(y.mean() - slope * t.mean())
    return y - (intercept + slope * t)


def _perm_entropy(x: np.ndarray, m: int) -> float:
    """Normalized Bandt-Pompe permutation entropy of order m in [0, 1].

    Symbolizes each length-m subsequence by the permutation that sorts it and
    measures the Shannon entropy of the resulting ordinal-pattern histogram,
    normalized by log(m!). 1.0 = patterns uniformly random (white noise),
    low values = a few ordinal motifs dominate (structured / predictable).
    Returns 1.0 (maximally random -> no trade) when too short to estimate.
    """
    k = x.size - m + 1
    if m < 2 or k < 2:
        return 1.0
    counts: dict = {}
    for j in range(k):
        pat = tuple(np.argsort(x[j:j + m], kind="stable").tolist())
        counts[pat] = counts.get(pat, 0) + 1
    p = np.asarray(list(counts.values()), dtype=float) / float(k)
    h = -float(np.sum(p * np.log(p)))
    h_max = float(np.log(np.prod(np.arange(1, m + 1, dtype=float))))
    return float(np.clip(h / h_max, 0.0, 1.0)) if h_max > 1e-12 else 1.0


def _dfa_alpha(r: np.ndarray) -> float:
    """DFA-1 scaling exponent of a return segment; nan when not estimable.

    Integrates the mean-centered returns into a profile, computes the RMS
    fluctuation F(s) around per-box linear fits across box sizes s, and
    returns the slope of log F(s) vs log s. For stationary increments
    alpha ~ H: 0.5 = uncorrelated, >0.5 persistent, <0.5 anti-persistent.
    """
    w = r.size
    if w < 8:
        return float("nan")
    profile = np.cumsum(r - r.mean())
    scales, flucts = [], []
    for s in range(4, w // 2 + 1, 2):
        nb = w // s
        if nb < 2:
            continue
        f2 = 0.0
        for b in range(nb):
            box = profile[b * s:(b + 1) * s]
            resid = _lin_detrend(box)
            f2 += float(np.mean(resid * resid))
        f = np.sqrt(f2 / nb)
        if f > 1e-12:
            scales.append(float(s))
            flucts.append(float(f))
    if len(scales) < 2:
        return float("nan")
    ls, lf = np.log(np.asarray(scales)), np.log(np.asarray(flucts))
    lc = ls - ls.mean()
    den = float(np.sum(lc * lc))
    if den <= 1e-18:
        return float("nan")
    return float(np.sum(lc * (lf - lf.mean())) / den)


# ---- 1. permutation-entropy regime-gated trend -------------------------------

@dataclass
class PermEntropyTrendStrategy:
    """Trend-following gated and sized by ordinal-pattern (permutation) entropy.

    Novel thesis: the *ordinal structure* of recent returns — which of the m!
    possible up/down orderings actually occur, and how evenly — is a direct,
    model-free measure of how forecastable the tape currently is. Bandt-Pompe
    permutation entropy is maximal for i.i.d. noise (every ordinal pattern
    equally likely) and drops sharply when a few motifs (e.g. monotone runs)
    dominate; empirically, "forbidden" or under-represented ordinal patterns
    are the signature of inefficient, autocorrelated market phases. So instead
    of running a trend signal unconditionally — bleeding in random regimes —
    we trade the sign of the window's net drift ONLY when normalized
    permutation entropy is low, with conviction ramping linearly from zero at
    `h_hi` (fully random, flat book) to full size at `h_lo` (strongly
    structured). The entropy gate is symbol-based, so it is robust to
    volatility scale and outliers in a way variance-based filters are not.

    References: C. Bandt & B. Pompe, "Permutation Entropy: A Natural
    Complexity Measure for Time Series" (Phys. Rev. Lett. 88, 174102, 2002);
    L. Zunino et al., "Forbidden patterns, permutation entropy and stock
    market inefficiency" (Physica A 388, 2009).
    """
    window: int = 10
    order: int = 3
    h_lo: float = 0.55
    h_hi: float = 0.95
    name: str = "perm_entropy_trend"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"perm_entropy_trend(w={self.window},m={self.order})"
        self.params = {"window": self.window, "order": self.order,
                       "h_lo": self.h_lo, "h_hi": self.h_hi}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n == 0:
            return np.zeros(0)
        r = _aligned_returns(c)
        span = max(self.h_hi - self.h_lo, 1e-9)
        sig = np.zeros(n)
        for i in range(1, n):
            w = min(self.window, i)  # r[i-w+1 : i+1] stays clear of r[0] = 0
            if w - self.order + 1 < 5:
                continue  # too few ordinal patterns for a usable histogram
            seg = r[i - w + 1:i + 1]
            h = _perm_entropy(seg, self.order)
            conviction = float(np.clip((self.h_hi - h) / span, 0.0, 1.0))
            if conviction <= 0.0:
                continue  # random regime: stay flat
            base = c[i - w]
            trend = float(np.sign(c[i] - base)) if abs(base) > 1e-12 else 0.0
            sig[i] = trend * conviction
        return _finalize(sig)


# ---- 2. DFA / Hurst multiscale persistence -----------------------------------

@dataclass
class DFAHurstStrategy:
    """Long-memory regime trader: DFA Hurst exponent picks trend vs. fade.

    Novel thesis: fractional-Brownian scaling of the return stream tells you
    *which* strategy class is currently paid, before you pick a signal. The
    detrended-fluctuation-analysis exponent alpha (~ Hurst H) measures how the
    RMS fluctuation of the integrated return profile grows with scale after
    removing local linear trends: H > 0.5 means increments reinforce across
    scales (persistent, long-memory tape — trend-following has positive
    expectancy), H < 0.5 means increments alternate (anti-persistent, rough
    tape — last moves tend to be given back), H ~ 0.5 is a martingale where
    neither works. We estimate alpha on a rolling window via per-box linear
    detrending across multiple box sizes (a genuinely multiscale statistic —
    robust to slow drifts that corrupt naive R/S), then: follow the window's
    drift sign when alpha is materially above 0.5, fade the latest move when
    materially below, and hold zero in the martingale band, sizing by
    |alpha - 0.5| so conviction tracks the measured distance from randomness.

    References: C.-K. Peng et al., "Mosaic organization of DNA nucleotides"
    (Phys. Rev. E 49, 1994 — DFA); H. E. Hurst, "Long-term storage capacity
    of reservoirs" (Trans. ASCE, 1951); B. Mandelbrot & J. van Ness,
    "Fractional Brownian Motions, Fractional Noises and Applications" (SIAM
    Rev., 1968); D. Cajueiro & B. Tabak, "The Hurst exponent over time"
    (Physica A 336, 2004) for time-varying market efficiency.
    """
    window: int = 16
    dev_min: float = 0.05  # dead band around H = 0.5 (martingale: no trade)
    dev_ref: float = 0.30  # |H - 0.5| at which conviction saturates at 1.0
    name: str = "dfa_hurst"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"dfa_hurst(w={self.window})"
        self.params = {"window": self.window, "dev_min": self.dev_min,
                       "dev_ref": self.dev_ref}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n == 0:
            return np.zeros(0)
        r = _aligned_returns(c)
        sig = np.zeros(n)
        for i in range(1, n):
            w = min(self.window, i)  # r[i-w+1 : i+1] stays clear of r[0] = 0
            if w < 12:
                continue  # need >= 2 DFA box sizes for a scaling slope
            alpha = _dfa_alpha(r[i - w + 1:i + 1])
            if not np.isfinite(alpha):
                continue
            dev = float(np.clip(alpha - 0.5, -0.5, 0.5))
            if abs(dev) < self.dev_min:
                continue  # martingale band: no measurable memory, stay flat
            magnitude = float(np.clip(abs(dev) / self.dev_ref, 0.0, 1.0))
            if dev > 0.0:
                base = c[i - w]
                direction = float(np.sign(c[i] - base)) if abs(base) > 1e-12 else 0.0
            else:
                direction = -float(np.sign(r[i]))  # anti-persistent: fade last move
            sig[i] = direction * magnitude
        return _finalize(sig)


# ---- 3. spectral dominant-cycle phase ----------------------------------------

@dataclass
class SpectralCyclePhaseStrategy:
    """Position from the phase of the dominant DFT cycle, concentration-gated.

    Novel thesis: after removing the local linear trend, what remains of a
    price window is (noise plus) an oscillatory component; if a single
    frequency genuinely dominates the residual spectrum, the *phase* of that
    component tells you where you sit inside the swing and hence the expected
    sign and size of the next increment — buy into the trough-to-peak upswing
    of the cycle, sell into the rollover, independent of any lagging
    moving-average machinery. Concretely: OLS-detrend a rolling window, take
    a real DFT, pick the highest-amplitude non-DC bin, and extrapolate that
    fitted sinusoid A*cos(2*pi*k*t/w + phi) one step past the window's end;
    the raw signal is the predicted one-step change normalized by the
    sinusoid's maximum one-step change (an exact bound, 2*A*sin(pi*k/w)), so
    it lives in [-1, 1] and peaks mid-swing where the cycle's slope is
    steepest. Crucially, the position is *gated by spectral concentration*
    (dominant-bin share of non-DC power): a flat, noise-like spectrum has no
    meaningful phase and produces no position, so the system only trades when
    an actual cycle is measurably present.

    References: J. F. Ehlers, "Cycle Analytics for Traders" (Wiley, 2013) —
    dominant-cycle measurement and cycle-mode trading; C. W. J. Granger &
    M. Hatanaka, "Spectral Analysis of Economic Time Series" (Princeton,
    1964).
    """
    window: int = 12
    min_conc: float = 0.25   # below this dominant-bin power share: no trade
    full_conc: float = 0.60  # at/above this share the gate is fully open
    name: str = "spectral_cycle"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"spectral_cycle(w={self.window})"
        self.params = {"window": self.window, "min_conc": self.min_conc,
                       "full_conc": self.full_conc}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n == 0:
            return np.zeros(0)
        gate_span = max(self.full_conc - self.min_conc, 1e-9)
        sig = np.zeros(n)
        for i in range(1, n):
            w = min(self.window, i + 1)
            if w < 8:
                continue  # too short for a meaningful spectrum
            resid = _lin_detrend(c[i - w + 1:i + 1])
            spec = np.fft.rfft(resid)
            mags = np.abs(spec)
            if mags.size < 3:
                continue
            power = mags * mags
            tot = float(np.sum(power[1:]))  # non-DC power only
            if tot <= 1e-18:
                continue
            k = int(np.argmax(mags[1:])) + 1
            conc = float(power[k] / tot)
            gate = float(np.clip((conc - self.min_conc) / gate_span, 0.0, 1.0))
            if gate <= 0.0:
                continue  # noise-like spectrum: phase is meaningless
            # amplitude of the k-th cosine component (Nyquist bin: no doubling)
            amp = mags[k] / w if (w % 2 == 0 and k == w // 2) else 2.0 * mags[k] / w
            phi = float(np.angle(spec[k]))
            omega = 2.0 * np.pi * k / w
            # predicted one-step change of the cycle just past the window end
            d = amp * (np.cos(omega * w + phi) - np.cos(omega * (w - 1) + phi))
            d_max = 2.0 * amp * np.sin(np.pi * k / w)  # exact per-step bound
            if d_max <= 1e-12:
                continue
            sig[i] = float(np.clip(d / d_max, -1.0, 1.0)) * gate
        return _finalize(sig)


# ---- registry ----------------------------------------------------------------

ALPHA3_STRATEGIES = {
    "perm_entropy_trend": PermEntropyTrendStrategy,
    "dfa_hurst": DFAHurstStrategy,
    "spectral_cycle": SpectralCyclePhaseStrategy,
}
