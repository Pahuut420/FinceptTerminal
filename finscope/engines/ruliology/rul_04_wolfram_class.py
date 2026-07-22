"""
rul_04 — Wolfram-class regime detector (ruliology strategy family).

Thesis
------
Encode the trailing window of return signs as the initial row of a 1-D binary
elementary cellular automaton (ECA) and evolve it under a fixed probe rule.
The resulting spacetime orbit is then classified into Wolfram's four behavior
classes with simple, deterministic statistics:

  class 1  fixed point  — orbit dies to a uniform / frozen row,
  class 2  periodic     — tail rows repeat with a short period,
  class 3  chaotic      — high block entropy, no periodicity, no spatially
                          localized structure (rule-30-like turbulence),
  class 4  complex      — intermediate entropy with localized propagating
                          structures (rule-110-like gliders).

Market reading: a sign tape whose CA orbit collapses (class 1) or locks into a
cycle (class 2) is *computationally reducible* — the recent tape is predictable,
so ride its drift at full/near-full size. A class-3 orbit means the tape is
irreducibly random — stand flat, there is nothing to compress. Class 4 is the
edge of chaos: structure exists but is fragile — trade the drift small,
scaled by how localized the structures are.

Causality: the signal at bar t is computed from returns r[t-window+1 .. t]
(closes up to t only) and then passed through `_shift1`, so weights[t] uses
information through bar t-1 exclusively. Perturbing the final close changes
no emitted weight.

numpy + finscope only; safe on series of any length (>= 0 bars).

Registry: `STRATEGIES` (name -> class).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1


# ---- ECA machinery -----------------------------------------------------------

def _eca_step(row: np.ndarray, rule: int) -> np.ndarray:
    """One synchronous ECA update with periodic (wrap-around) boundaries."""
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right          # neighborhood code 0..7
    return (rule >> idx) & 1                         # rule bit lookup


def _evolve(row: np.ndarray, rule: int, steps: int) -> np.ndarray:
    """Evolve `row` for `steps` updates; returns (steps+1, width) 0/1 array."""
    rows = np.empty((steps + 1, row.size), dtype=np.int64)
    rows[0] = row
    for t in range(steps):
        rows[t + 1] = _eca_step(rows[t], rule)
    return rows


# ---- orbit statistics --------------------------------------------------------

def _block_entropy(rows: np.ndarray, k: int = 3) -> float:
    """Normalized Shannon entropy of overlapping k-blocks over the orbit tail.

    Discards the first half of the orbit as transient. Returns a value in
    [0, 1]: 0 = perfectly ordered, 1 = maximal (i.i.d.-uniform) block entropy.
    """
    tail = rows[rows.shape[0] // 2:]
    width = tail.shape[1]
    if width < k or tail.size == 0:
        return 0.0
    # vectorized sliding k-block codes across every tail row
    blocks = sum(
        tail[:, j:width - k + 1 + j] << (k - 1 - j) for j in range(k)
    )
    counts = np.bincount(blocks.ravel(), minlength=2 ** k).astype(float)
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum() / k)


def _tail_period(rows: np.ndarray) -> int:
    """Smallest p >= 2 such that the last 2p rows repeat with period p (0 if none)."""
    n = rows.shape[0]
    for p in range(2, n // 2 + 1):
        if np.array_equal(rows[n - p:], rows[n - 2 * p:n - p]):
            return p
    return 0


def _classify(rows: np.ndarray, chaos_entropy: float) -> Tuple[int, float]:
    """Map a spacetime orbit to (wolfram_class, confidence in [0, 1])."""
    n_steps = rows.shape[0] - 1
    if n_steps < 2:
        return 3, 1.0  # degenerate orbit: treat as unclassifiable -> flat

    # -- class 1: frozen (uniform final row, or reached a fixed point) --------
    final = rows[-1]
    frozen = bool(np.all(final == final[0])) or bool(np.array_equal(rows[-1], rows[-2]))
    if frozen:
        # confidence grows with how early the orbit froze
        conv = n_steps - 1
        for t in range(n_steps):
            if np.array_equal(rows[t], rows[t + 1]) or np.all(rows[t] == rows[t][0]):
                conv = t
                break
        conf = 1.0 - conv / max(1, n_steps - 1)
        return 1, float(np.clip(conf, 0.25, 1.0))

    # -- class 2: short-period cycle in the tail ------------------------------
    p = _tail_period(rows)
    if p:
        conf = 1.0 - (p - 2) / max(1, n_steps // 2)
        return 2, float(np.clip(conf, 0.30, 1.0))

    # -- class 3 vs 4: entropy + spatial localization of activity -------------
    h = _block_entropy(rows)
    flips = (rows[1:] != rows[:-1])
    col_activity = flips.mean(axis=0)          # per-column flip rate
    localization = float(col_activity.std())   # gliders => uneven activity

    if h >= chaos_entropy and localization < 0.18:
        # deep, spatially homogeneous turbulence: irreducible randomness
        return 3, 1.0
    # complex / edge-of-chaos: confidence from structure localization
    conf = (localization - 0.10) / 0.25
    return 4, float(np.clip(conf, 0.15, 1.0))


# ---- shared signal core ------------------------------------------------------

# position sizing per Wolfram class (before confidence scaling)
_CLASS_BASE = {1: 1.0, 2: 0.8, 3: 0.0, 4: 0.35}


def _class_signal(closes: np.ndarray, rule: int, window: int, steps: int,
                  chaos_entropy: float) -> np.ndarray:
    """Raw (unshifted) signal: weight[t] from the sign tape ending at bar t."""
    n = closes.size
    sig = np.zeros(n, dtype=float)
    if n < window + 1:
        return sig
    rets = np.diff(closes) / np.where(closes[:-1] != 0.0, closes[:-1], 1.0)
    bits = (rets > 0.0).astype(np.int64)       # up-tick = 1, else 0
    for t in range(window, n):
        tape = bits[t - window:t]              # signs of returns .. through bar t
        drift = float(np.sum(rets[t - window:t]))
        direction = float(np.sign(drift))
        if direction == 0.0:
            continue
        orbit = _evolve(tape, rule, steps)
        wclass, conf = _classify(orbit, chaos_entropy)
        sig[t] = direction * _CLASS_BASE[wclass] * conf
    return sig


def _finalize(sig: np.ndarray) -> np.ndarray:
    """Sanitize, clip and shift a raw signal into a causal weight vector."""
    return np.clip(np.nan_to_num(_shift1(np.nan_to_num(sig))), -1.0, 1.0)


# ---- strategies --------------------------------------------------------------

@dataclass
class WolframClassRegimeStrategy:
    """Single-probe Wolfram-class regime detector.

    Evolves the trailing return-sign tape under one ECA rule (default 110, the
    canonical class-4 universal rule — a sensitive 'computational microscope':
    ordered tapes collapse or cycle under it quickly, random tapes sustain
    turbulence) and positions by the detected class: full trend in class 1,
    strong trend in class 2, flat in class 3, small edge-play in class 4 —
    each scaled by classification confidence.
    """
    rule: int = 110
    window: int = 12
    steps: int = 12
    chaos_entropy: float = 0.72
    name: str = "wolfram_class_regime"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rule = int(self.rule) & 0xFF
        self.name = f"wolfram_class(r{self.rule},w={self.window})"
        self.params = {"rule": self.rule, "window": self.window,
                       "steps": self.steps, "chaos_entropy": self.chaos_entropy}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        sig = _class_signal(c, self.rule, self.window, self.steps, self.chaos_entropy)
        return _finalize(sig)


@dataclass
class WolframClassEnsembleStrategy:
    """Multi-probe Wolfram-class consensus.

    Classifies the same sign tape under three probe rules with different
    intrinsic dynamics — 110 (class 4, structure-revealing), 90 (class 2,
    additive/nested: linear, so orbit entropy mirrors input structure) and
    184 (particle-conserving 'traffic' rule: majority/drift revealing) —
    and averages the resulting class-mapped weights. Agreement across probes
    concentrates the position; disagreement (mixed regime evidence)
    automatically shrinks it toward flat.
    """
    rules: Tuple[int, ...] = (110, 90, 184)
    window: int = 12
    steps: int = 12
    chaos_entropy: float = 0.72
    name: str = "wolfram_class_ensemble"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rules = tuple(int(r) & 0xFF for r in self.rules)
        self.name = f"wolfram_ensemble({','.join(str(r) for r in self.rules)},w={self.window})"
        self.params = {"rules": list(self.rules), "window": self.window,
                       "steps": self.steps, "chaos_entropy": self.chaos_entropy}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        if c.size == 0:
            return np.zeros(0, dtype=float)
        acc = np.zeros(c.size, dtype=float)
        for rule in self.rules:
            acc += _class_signal(c, rule, self.window, self.steps, self.chaos_entropy)
        acc /= max(1, len(self.rules))
        return _finalize(acc)


STRATEGIES = {
    "wolfram_class_regime": WolframClassRegimeStrategy,
    "wolfram_class_ensemble": WolframClassEnsembleStrategy,
}
