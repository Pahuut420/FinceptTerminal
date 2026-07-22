"""
rul_08 — 3-color totalistic cellular automata on quantized returns.

Ruliology brief
---------------
A *totalistic* CA updates each cell from the SUM (total) of its neighborhood,
not the exact arrangement. With k=3 colors and radius 1 the neighborhood sum
ranges 0..6, so a rule is 7 base-3 digits — a code in [0, 3^7 = 2187). Wolfram
used exactly this family in NKS to exhibit class-4 (edge-of-chaos) complexity;
code 1599 is the canonical class-4 example and is the default here.

Trading translation
-------------------
1. ENCODE — quantize each bar's return into down/flat/up (0/1/2) using a
   volatility-scaled dead band: |r| <= k * rolling_sigma reads as "flat".
   The band adapts to regime, so the symbol stream stays informative in both
   quiet and violent tape.
2. EVOLVE — the last `window` symbols form a ring (periodic boundary) and the
   totalistic rule is iterated `gens` steps. The CA is a nonlinear, shift-
   invariant filter over the recent return pattern: class-4 dynamics let local
   structures (runs, alternations) propagate and interact instead of being
   averaged away as a linear filter would.
3. READ OUT — two graded readouts, one per strategy:
   * center commitment: the trajectory of the central cell over the final
     generations. A center pinned at 2 (up) is a full-strength long; a
     flickering center is a small position. Direction and conviction come
     from the same statistic.
   * mean-state drift: the ring's polarization (n_up - n_down) / width,
     averaged over the final generations. How far the whole automaton drifts
     from the flat state measures how strongly the dynamics commit to a level.

Causality: every per-bar signal uses closes up to and including that bar only
(rolling sigma is trailing; the CA row ends at the current bar), and the final
vector goes through `_shift1`, so weights[t] is decided at the close of t-1.

numpy + finscope only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1

_K = 3            # colors
_NSUMS = 7        # radius-1 neighborhood sums: 0..6
_CODE_SPACE = _K ** _NSUMS  # 2187 possible totalistic rules


def _rule_digits(code: int) -> np.ndarray:
    """Base-3 digits of a totalistic rule code: digits[s] = new color for
    neighborhood sum s (Wolfram's totalistic-code convention)."""
    code = int(code) % _CODE_SPACE
    return np.array([(code // _K ** s) % _K for s in range(_NSUMS)], dtype=np.int64)


def _returns(c: np.ndarray) -> np.ndarray:
    """Simple returns, first element 0, safe against zero closes."""
    r = np.zeros_like(c)
    if c.size > 1:
        prev = c[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            r[1:] = np.where(prev != 0.0, (c[1:] - prev) / prev, 0.0)
    return np.nan_to_num(r)


def _trailing_std(x: np.ndarray, w: int) -> np.ndarray:
    """Std of the trailing (<= w)-bar window ending at each index. Causal:
    index t sees x[max(0, t-w+1) : t+1] only."""
    n = x.size
    out = np.zeros(n)
    c1 = np.cumsum(np.insert(x, 0, 0.0))
    c2 = np.cumsum(np.insert(x * x, 0, 0.0))
    for t in range(n):
        lo = max(0, t - w + 1)
        cnt = t + 1 - lo
        m = (c1[t + 1] - c1[lo]) / cnt
        v = (c2[t + 1] - c2[lo]) / cnt - m * m
        out[t] = np.sqrt(v) if v > 0.0 else 0.0
    return out


def _quantize(rets: np.ndarray, sigma: np.ndarray, band_k: float) -> np.ndarray:
    """Ternary encoding: 0 = down, 1 = flat, 2 = up, dead band = band_k * sigma."""
    theta = np.maximum(band_k * sigma, 1e-12)
    return (1 + (rets > theta).astype(np.int64) - (rets < -theta).astype(np.int64))


def _evolve(row: np.ndarray, digits: np.ndarray, gens: int) -> np.ndarray:
    """Iterate the totalistic rule on a ring; returns (gens+1, width) history."""
    hist = np.empty((gens + 1, row.size), dtype=np.int64)
    hist[0] = row
    for g in range(gens):
        r = hist[g]
        s = np.roll(r, 1) + r + np.roll(r, -1)  # periodic boundary
        hist[g + 1] = digits[s]
    return hist


@dataclass
class TotalisticCenterStrategy:
    """3-color totalistic CA; signal = central-cell commitment.

    The last `window` ternary return symbols seed a ring; rule `code` runs for
    `gens` steps; the central cell's state over the final `tail` generations,
    mapped {0: -1, 1: 0, 2: +1} and averaged, is the position. A center that
    locks onto a level gives full size; a flickering center gives little.
    """
    code: int = 1599        # NKS canonical class-4 3-color totalistic rule
    window: int = 9         # ring width (bars of encoded returns)
    gens: int = 9           # CA generations per bar
    vol_lb: int = 10        # trailing-sigma lookback for the dead band
    band_k: float = 0.5     # dead-band half-width in sigmas
    gain: float = 1.5       # signal gain before clipping
    name: str = "ca3_totalistic_center"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"ca3_center(code={self.code},w={self.window},g={self.gens})"
        self.params = {"code": self.code, "window": self.window, "gens": self.gens,
                       "vol_lb": self.vol_lb, "band_k": self.band_k, "gain": self.gain}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n < 3:
            return np.zeros(n)
        digits = _rule_digits(self.code)
        rets = _returns(c)
        sym = _quantize(rets, _trailing_std(rets, self.vol_lb), self.band_k)
        w = max(3, int(self.window))
        gens = max(1, int(self.gens))
        tail = max(2, gens // 3)
        center = w // 2
        sig = np.zeros(n)
        for t in range(w - 1, n):
            hist = _evolve(sym[t - w + 1: t + 1], digits, gens)
            # mean of {-1, 0, +1} center states over the final generations:
            # direction AND conviction in one graded statistic.
            sig[t] = float(np.mean(hist[-tail:, center] - 1))
        sig = np.clip(self.gain * np.nan_to_num(sig), -1.0, 1.0)
        return _shift1(sig)


@dataclass
class TotalisticDriftStrategy:
    """3-color totalistic CA; signal = mean-state polarization drift.

    Same encoding and evolution, but the readout is global: the ring's
    polarization mean(state - 1) = (n_up - n_down) / width, averaged over the
    final `tail` generations. Measures how strongly the whole automaton
    commits to a level after the dynamics have digested the recent tape.
    """
    code: int = 1599
    window: int = 11
    gens: int = 11
    vol_lb: int = 10
    band_k: float = 0.5
    gain: float = 2.5       # polarization rarely saturates for class-4 rules
    name: str = "ca3_totalistic_drift"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"ca3_drift(code={self.code},w={self.window},g={self.gens})"
        self.params = {"code": self.code, "window": self.window, "gens": self.gens,
                       "vol_lb": self.vol_lb, "band_k": self.band_k, "gain": self.gain}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n < 3:
            return np.zeros(n)
        digits = _rule_digits(self.code)
        rets = _returns(c)
        sym = _quantize(rets, _trailing_std(rets, self.vol_lb), self.band_k)
        w = max(3, int(self.window))
        gens = max(1, int(self.gens))
        tail = max(2, gens // 3)
        sig = np.zeros(n)
        for t in range(w - 1, n):
            hist = _evolve(sym[t - w + 1: t + 1], digits, gens)
            # ring polarization averaged over the final generations, in [-1, 1]
            sig[t] = float(np.mean(hist[-tail:] - 1))
        sig = np.clip(self.gain * np.nan_to_num(sig), -1.0, 1.0)
        return _shift1(sig)


STRATEGIES = {
    "ca3_totalistic_center": TotalisticCenterStrategy,
    "ca3_totalistic_drift": TotalisticDriftStrategy,
}
