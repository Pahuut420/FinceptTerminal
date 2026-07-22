"""
rul_15_irreducibility — Computational-irreducibility gate via LZ76 (Ruliology #15).

Concept
-------
Wolfram's computational irreducibility: for many processes there is no shortcut —
the only way to know the outcome is to run every step. A *reducible* process is
compressible: a short program (pattern) reproduces it, so its next step is
predictable. An *irreducible* process is incompressible: its shortest description
is itself, so forecasting is hopeless and the correct position is flat.

The practical proxy for irreducibility used here is Lempel–Ziv 1976 complexity
(Kaspar–Schuster exhaustive-history phrase count) of the recent return-sign bit
string. Few phrases => compressible => reducible => trade the prevailing trend.
Phrase count at (or above) the random-sequence level => irreducible => flat.

Mapping (market -> symbols)
---------------------------
1. ENCODE  — bit[j] = 1 iff close[j+1] > close[j], else 0 (return-sign bits).
   bit[j] is known at bar j+1.
2. MEASURE — over each trailing window of `window` bits compute LZ76 phrase
   count C, normalized to c = C * log2(window) / window (the asymptotic random
   level is ~1, but small windows bias it upward — see 3).
3. CALIBRATE — the *same* estimator is run over a fixed, seeded pseudo-random
   bit stream (RandomState(30) — a nod to Rule 30), giving an empirical null of
   window complexities. Using an identical window/estimator cancels the
   finite-sample bias, so "as complex as random" is measured fairly. The null
   is deterministic (fixed seed) and uses no market data.
4. TRADE   — direction = majority vote of the last `trend` bits (Rule-232 nod).
   * gate variant : all-or-nothing. Trade only when the window's complexity
     undercuts the q-quantile of the random null — i.e. the market is more
     compressible than random almost never is by chance.
   * scale variant: continuous. Position magnitude = reducibility, the window
     complexity mapped linearly from the random-null mean (reducibility 0,
     flat) down to the constant-sequence floor (reducibility 1, full size) —
     an anchored form of "1 - normalized complexity".

Causality
---------
sig[t] only reads bits with index <= t-1 (i.e. closes through bar t), and every
signal is passed through `_shift1`, so weights()[t] depends exclusively on
information available at the close of bar t-1. The null is a constant.

numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1


# --------------------------------------------------------------------------- #
# LZ76 machinery                                                              #
# --------------------------------------------------------------------------- #

def _lz76(s: bytes) -> int:
    """Lempel–Ziv 1976 complexity (Kaspar–Schuster): number of phrases in the
    exhaustive production history of the symbol string `s`."""
    n = len(s)
    if n == 0:
        return 0
    if n == 1:
        return 1
    c, l, i, k, k_max = 1, 1, 0, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c


def _rolling_norm_lz76(bits: np.ndarray, window: int) -> np.ndarray:
    """Normalized LZ76 complexity for every trailing `window`-bit window.

    Returns an array aligned with `bits`: out[e] = C(bits[e-window+1 .. e])
    * log2(window) / window; NaN where the window is incomplete. Strictly
    trailing => causal by construction."""
    bits = np.asarray(bits, dtype=np.uint8)
    m = bits.size
    out = np.full(m, np.nan)
    if window < 2 or m < window:
        return out
    scale = float(np.log2(window)) / float(window)
    buf = bits.tobytes()  # byte-indexing is fast ints
    for e in range(window - 1, m):
        out[e] = _lz76(buf[e - window + 1:e + 1]) * scale
    return out


_NULL_CACHE: Dict[int, np.ndarray] = {}


def _lz76_null(window: int, stream_len: int = 4096, seed: int = 30) -> np.ndarray:
    """Null distribution of normalized window LZ76 complexities measured on a
    fixed seeded pseudo-random bit stream with the *same* estimator. Cached per
    window; deterministic (RandomState is version-stable); no market data."""
    key = int(window)
    cached = _NULL_CACHE.get(key)
    if cached is None:
        rng = np.random.RandomState(seed)
        stream = rng.randint(0, 2, int(stream_len)).astype(np.uint8)
        vals = _rolling_norm_lz76(stream, key)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:  # degenerate params — treat everything as random
            vals = np.array([1.0])
        cached = vals
        _NULL_CACHE[key] = cached
    return cached


def _c_floor(window: int) -> float:
    """Normalized LZ76 of a constant window (phrase count 2): the reducibility
    ceiling / complexity floor for this window length."""
    return 2.0 * float(np.log2(window)) / float(window) if window >= 2 else 0.0


# --------------------------------------------------------------------------- #
# shared market state                                                         #
# --------------------------------------------------------------------------- #

def _return_bits(closes: np.ndarray) -> np.ndarray:
    """bit[j] = 1 iff close[j+1] > close[j] (flat/down = 0); known at bar j+1."""
    if closes.size < 2:
        return np.zeros(0, dtype=np.uint8)
    diffs = np.diff(np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0))
    return (diffs > 0.0).astype(np.uint8)


def _majority_direction(bits: np.ndarray, trend: int) -> np.ndarray:
    """Majority vote over the last `trend` bits: d[e] in {-1, 0, +1}, 0 while
    the window is incomplete (Rule-232 majority nod)."""
    m = bits.size
    d = np.zeros(m)
    if trend < 1 or m < trend:
        return d
    cs = np.concatenate(([0.0], np.cumsum(bits.astype(np.float64))))
    frac = (cs[trend:] - cs[:-trend]) / trend
    d[trend - 1:] = np.sign(frac - 0.5)
    return d


def _state(series: Series, window: int, trend: int):
    """(n, normalized-complexity per bit-end, majority direction per bit-end).
    Bar t sees bit index e = t - 1."""
    c = _closes(series)
    n = c.size
    bits = _return_bits(c)
    cn = _rolling_norm_lz76(bits, window)
    d = _majority_direction(bits, trend)
    return n, cn, d


# --------------------------------------------------------------------------- #
# Strategies                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class LZ76IrreducibilityGateStrategy:
    """All-or-nothing irreducibility gate. Trade the majority trend at +/-1
    ONLY when the return-bit window's normalized LZ76 complexity undercuts the
    q-quantile of the seeded random null — i.e. the sequence is demonstrably
    compressible (reducible => predictable). Otherwise it is statistically
    indistinguishable from an irreducible random stream -> flat."""
    window: int = 12
    trend: int = 6
    q: float = 0.10
    name: str = "lz76_irreducibility_gate"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(int(self.window), 4)
        self.trend = max(int(self.trend), 1)
        self.q = float(min(max(self.q, 0.0), 1.0))
        self.name = f"lz76_irreducibility_gate(w={self.window},t={self.trend},q={self.q})"
        self.params = {"window": self.window, "trend": self.trend, "q": self.q}

    def weights(self, series: Series) -> np.ndarray:
        n, cn, d = _state(series, self.window, self.trend)
        if n == 0:
            return np.zeros(0)
        gate = float(np.quantile(_lz76_null(self.window), self.q))
        sig = np.zeros(n)
        if n > 1:
            e = np.arange(n - 1)          # bit index seen by bar t = e + 1
            ok = np.isfinite(cn[e]) & (cn[e] < gate)
            sig[1:] = np.where(ok, d[e], 0.0)
        sig = np.clip(np.nan_to_num(sig), -1.0, 1.0)
        return _shift1(sig)


@dataclass
class LZ76ReducibilityScaleStrategy:
    """Continuous sibling: position = majority-trend direction sized by
    *reducibility* — the window's normalized LZ76 complexity mapped linearly
    from the random-null mean (reducibility 0 => flat: irreducible, nothing to
    shortcut) down to the constant-sequence floor (reducibility 1 => full
    size). An anchored, finite-sample-fair "1 - normalized complexity"."""
    window: int = 12
    trend: int = 6
    name: str = "lz76_reducibility_scale"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(int(self.window), 4)
        self.trend = max(int(self.trend), 1)
        self.name = f"lz76_reducibility_scale(w={self.window},t={self.trend})"
        self.params = {"window": self.window, "trend": self.trend}

    def weights(self, series: Series) -> np.ndarray:
        n, cn, d = _state(series, self.window, self.trend)
        if n == 0:
            return np.zeros(0)
        anchor = float(np.mean(_lz76_null(self.window)))  # random level ~ irreducible
        floor = _c_floor(self.window)                     # constant-sequence level
        denom = anchor - floor
        denom = denom if denom > 1e-9 else 1.0
        sig = np.zeros(n)
        if n > 1:
            e = np.arange(n - 1)
            # NaN complexity (incomplete window) -> anchor -> reducibility 0
            red = (anchor - np.nan_to_num(cn[e], nan=anchor)) / denom
            sig[1:] = d[e] * np.clip(red, 0.0, 1.0)
        sig = np.clip(np.nan_to_num(sig), -1.0, 1.0)
        return _shift1(sig)


STRATEGIES = {
    "lz76_irreducibility_gate": LZ76IrreducibilityGateStrategy,
    "lz76_reducibility_scale": LZ76ReducibilityScaleStrategy,
}
