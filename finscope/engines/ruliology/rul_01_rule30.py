"""
rul_01_rule30 — Rule 30 as a market-randomness benchmark (Ruliology signal #1).

Concept
-------
Wolfram's Rule 30 is the canonical class-3 elementary cellular automaton: from a
single black cell it generates a center column so chaotic it served as
Mathematica's pseudo-random generator. Computational irreducibility says that
stream has no exploitable shortcut — it is the *null hypothesis* for "this bit
sequence is untradeable noise".

Mapping (market -> CA symbols)
------------------------------
1. ENCODE  — each bar's return sign becomes a bit: up (close[t] > close[t-1]) = 1,
   down-or-flat = 0. The recent market is a binary string, exactly the alphabet
   an ECA row lives in.
2. MEASURE — over a trailing window of `window` bits, compute the normalized
   block (Shannon) entropy of overlapping `block`-bit patterns, h in [0, 1]
   bits/symbol. Low h = structured/predictable; h ~ 1 = coin-flip.
3. BENCHMARK — evolve Rule 30 (new = left XOR (center OR right)) on a cyclic
   lattice, harvest its center column, and run the *same* finite-window
   estimator over it. This yields Rule 30's own null distribution of window
   entropies — using an identical estimator/window cancels the finite-sample
   downward bias, so the comparison is fair.
4. TRADE   — only when the market window is measurably MORE predictable than
   Rule-30 chaos (its entropy falls below a low quantile / below the Rule-30
   mean) do we trade a short trend, read as the majority vote of the last
   `trend` bits (a Rule-232 majority nod). When the market is as random as
   Rule 30, there is nothing to compute forward — stay flat.

Causality
---------
The bit known at bar t uses closes up to t only; every signal is passed through
`_shift1`, so weights()[t] depends exclusively on information through bar t-1.
The Rule 30 benchmark is a deterministic constant (no market data).

Strategies
----------
- rule30_entropy_gate   : all-or-nothing. Trade the majority trend at +/-1 only
  when window entropy < the q-quantile of Rule 30's window-entropy null.
- rule30_entropy_margin : continuous. Position = trend direction scaled by the
  predictability margin (Rule-30 mean entropy - market entropy), 0 when the
  market is at least as random as Rule 30.

numpy only; CA evolved with integer/bit ops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1


# --------------------------------------------------------------------------- #
# CA + entropy machinery                                                      #
# --------------------------------------------------------------------------- #

def _rule30_center_column(n_bits: int, width: int = 257, warmup: int = 64) -> np.ndarray:
    """Center column of Rule 30 grown from a single 1 on a cyclic lattice.

    Rule 30: new = left XOR (center OR right). The center column is the
    classic Wolfram pseudo-random bit stream.
    """
    state = np.zeros(width, dtype=np.uint8)
    mid = width // 2
    state[mid] = 1
    total = int(n_bits) + int(warmup)
    col = np.empty(total, dtype=np.uint8)
    for i in range(total):
        col[i] = state[mid]
        state = np.roll(state, 1) ^ (state | np.roll(state, -1))
    return col[warmup:]


def _sliding_block_entropy(bits: np.ndarray, window: int, block: int) -> np.ndarray:
    """Normalized block entropy for every trailing window of `window` bits.

    Returns an array h aligned with `bits`: h[e] is the entropy (bits/symbol,
    in [0, 1]) of bits[e-window+1 .. e]; NaN where the window is incomplete.
    Overlapping `block`-bit patterns are counted with cumulative sums, so the
    whole thing is vectorized numpy.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    m = bits.size
    h = np.full(m, np.nan)
    n_blocks = m - block + 1
    per_win = window - block + 1  # overlapping blocks per window
    if n_blocks <= 0 or per_win <= 0 or m < window:
        return h

    # integer code of each overlapping block (LSB-first)
    codes = np.zeros(n_blocks, dtype=np.int64)
    for j in range(block):
        codes |= bits[j:j + n_blocks].astype(np.int64) << j

    n_windows = n_blocks - per_win + 1
    n_codes = 1 << block
    probs = np.empty((n_codes, n_windows))
    for cval in range(n_codes):
        ind = (codes == cval).astype(np.float64)
        cs = np.concatenate(([0.0], np.cumsum(ind)))
        probs[cval] = (cs[per_win:] - cs[:-per_win]) / per_win

    safe = np.where(probs > 0.0, probs, 1.0)  # log2(1) = 0 -> no warnings
    ent = -(probs * np.log2(safe)).sum(axis=0) / block  # in [0, 1]

    # window starting at block-index i ends at bit-index i + window - 1
    ends = np.arange(n_windows) + window - 1
    h[ends] = ent
    return h


_BENCH_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def _rule30_entropy_null(window: int, block: int, stream_len: int = 2048) -> np.ndarray:
    """Null distribution of window entropies measured on Rule 30's own stream
    with the *same* estimator (same window/block). Cached per parameter pair."""
    key = (int(window), int(block))
    cached = _BENCH_CACHE.get(key)
    if cached is None:
        stream = _rule30_center_column(stream_len)
        h = _sliding_block_entropy(stream, window, block)
        vals = h[np.isfinite(h)]
        if vals.size == 0:  # degenerate params — fall back to max entropy
            vals = np.array([1.0])
        cached = vals
        _BENCH_CACHE[key] = cached
    return cached


def _return_bits(closes: np.ndarray) -> np.ndarray:
    """Up/down encoding: bit[j] = 1 iff close[j+1] > close[j] (flat counts as 0).
    bit[j] is known at bar j+1."""
    if closes.size < 2:
        return np.zeros(0, dtype=np.uint8)
    diffs = np.diff(np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0))
    return (diffs > 0.0).astype(np.uint8)


def _majority_direction(bits: np.ndarray, trend: int) -> np.ndarray:
    """Rule-232-style majority vote over the last `trend` bits.

    d[e] = sign(mean(bits[e-trend+1 .. e]) - 0.5) in {-1, 0, +1}; NaN-free,
    0 where the window is incomplete."""
    m = bits.size
    d = np.zeros(m)
    if m < trend or trend < 1:
        return d
    cs = np.concatenate(([0.0], np.cumsum(bits.astype(np.float64))))
    frac = (cs[trend:] - cs[:-trend]) / trend
    d[trend - 1:] = np.sign(frac - 0.5)
    return d


def _entropy_state(series: Series, window: int, block: int, trend: int):
    """Shared per-bar state: (n, h_per_bit_end, majority_dir). Returns arrays
    aligned with the bit index e; bar t sees bit index e = t - 1."""
    c = _closes(series)
    n = c.size
    bits = _return_bits(c)
    h = _sliding_block_entropy(bits, window, block)
    d = _majority_direction(bits, trend)
    return n, h, d


# --------------------------------------------------------------------------- #
# Strategies                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class Rule30EntropyGateStrategy:
    """Trade the short trend ONLY when the market's return-bit window is more
    predictable than Rule 30's chaos: window entropy must undercut the
    q-quantile of Rule 30's own window-entropy null distribution. Otherwise
    the market is Rule-30-random -> flat (irreducible, don't trade)."""
    window: int = 12
    block: int = 2
    trend: int = 6
    q: float = 0.10
    name: str = "rule30_entropy_gate"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(int(self.window), max(int(self.block) + 1, 2))
        self.block = max(int(self.block), 1)
        self.trend = max(int(self.trend), 1)
        self.q = float(min(max(self.q, 0.0), 1.0))
        self.name = f"rule30_entropy_gate(w={self.window},k={self.block},q={self.q})"
        self.params = {"window": self.window, "block": self.block,
                       "trend": self.trend, "q": self.q}

    def weights(self, series: Series) -> np.ndarray:
        n, h, d = _entropy_state(series, self.window, self.block, self.trend)
        if n == 0:
            return np.zeros(0)
        null = _rule30_entropy_null(self.window, self.block)
        gate = float(np.quantile(null, self.q))
        sig = np.zeros(n)
        # bar t knows bits up to index t-1
        e = np.arange(n - 1)                      # bit index seen by bar t = e + 1
        ok = np.isfinite(h[e]) & (h[e] < gate)
        sig[1:] = np.where(ok, d[e], 0.0)
        sig = np.clip(np.nan_to_num(sig), -1.0, 1.0)
        return _shift1(sig)


@dataclass
class Rule30EntropyMarginStrategy:
    """Continuous sibling: position = majority-trend direction scaled by the
    predictability margin (Rule-30 mean window entropy - market window
    entropy), normalized to [0, 1]. Size decays to zero exactly as the market
    approaches Rule-30 randomness, and is zero beyond it."""
    window: int = 12
    block: int = 2
    trend: int = 6
    name: str = "rule30_entropy_margin"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(int(self.window), max(int(self.block) + 1, 2))
        self.block = max(int(self.block), 1)
        self.trend = max(int(self.trend), 1)
        self.name = f"rule30_entropy_margin(w={self.window},k={self.block})"
        self.params = {"window": self.window, "block": self.block, "trend": self.trend}

    def weights(self, series: Series) -> np.ndarray:
        n, h, d = _entropy_state(series, self.window, self.block, self.trend)
        if n == 0:
            return np.zeros(0)
        null = _rule30_entropy_null(self.window, self.block)
        anchor = float(np.mean(null))             # Rule 30's typical window entropy
        anchor = anchor if anchor > 1e-9 else 1.0
        sig = np.zeros(n)
        e = np.arange(n - 1)
        margin = (anchor - np.nan_to_num(h[e], nan=anchor)) / anchor
        size = np.clip(margin, 0.0, 1.0)
        sig[1:] = d[e] * size
        sig = np.clip(np.nan_to_num(sig), -1.0, 1.0)
        return _shift1(sig)


STRATEGIES = {
    "rule30_entropy_gate": Rule30EntropyGateStrategy,
    "rule30_entropy_margin": Rule30EntropyMarginStrategy,
}
