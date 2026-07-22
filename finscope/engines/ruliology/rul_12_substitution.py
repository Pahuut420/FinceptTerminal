"""
rul_12_substitution — substitution-system (L-system) self-similarity as a
causal trading signal (NKS ch. 3, "Substitution Systems").

A substitution system rewrites every symbol in parallel each step by a fixed
block (e.g. 0 -> 01, 1 -> 10). Iterating from a single seed generates the
fixed-point words of the system — sequences that are *exactly* self-similar:
nested, deterministic, scale-invariant. Two archetypes are used here:

  * Thue-Morse word   (0 -> 01, 1 -> 10): the balanced nested word. Equal
    symbol density, no cubes (no block ever repeats three times), and exact
    self-similarity t[2n] = t[n], t[2n+1] = 1 - t[n]. It is the canonical
    "structured alternation at every scale" pattern.
  * Fibonacci word    (1 -> 10, 0 -> 1): the canonical Sturmian word. Golden-
    ratio symbol density (~61.8% majority symbol), isolated minority symbols
    with self-similar spacing — a "fractal trend with structured pullbacks".

Encoding: the recent return-sign sequence ('1' = up bar, '0' = down/flat bar)
is treated as a candidate window of a substitution fixed point. Every trailing
window is aligned against *all phases* (offsets) of the substitution word, in
both polarities (the complemented word covers the mirrored regime, e.g. a
down-drift Fibonacci regime). The best alignment yields:

  * match  in [0.5, 1] — how strongly the market's recent up/down sequence
    exhibits the nested self-similar structure of the substitution system;
  * phase  — the offset of the best alignment, which *deterministically*
    prescribes the next symbol of the word.

Signal logic (per the ruliology brief):

  * strong self-similarity (match above `threshold`)  -> a structured fractal
    regime: trade in the direction of the next symbol of the current
    substitution phase, sized by the excess match;
  * broken self-similarity (match at/below threshold) -> regime change:
    exit — weight goes to 0 rather than guessing.

Causality: the bit at index i is the sign of the return from close[i] to
close[i+1], known only at the close of bar i+1. The alignment signal built
from bits[..i] is therefore stamped at series index i+1 and then shifted one
bar via `_shift1`, so weights()[t] only uses information available at the
close of bar t-1 (same convention as finscope.engines.strategies).

Strategies
----------
thue_morse_phase    : alignment against the Thue-Morse word — detects nested
                      balanced alternation; direction = next Thue-Morse symbol
                      of the best-matching phase.
fibonacci_word_flow : alignment against the Fibonacci word — detects golden-
                      ratio self-similar trend structure; direction = next
                      Fibonacci-word symbol of the best-matching phase
                      (majority symbol most of the time => rides the
                      structured trend, steps aside at its self-similar
                      pullback slots).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1


# ---------------------------------------------------------------------------
# substitution-system fixed-point words
# ---------------------------------------------------------------------------

def _thue_morse_word(n: int) -> np.ndarray:
    """First n symbols of the Thue-Morse word (0 -> 01, 1 -> 10 fixed point).

    Uses the closed form t[i] = popcount(i) mod 2, which is exactly the
    substitution fixed point started from 0.
    """
    if n <= 0:
        return np.zeros(0, dtype=float)
    v = np.arange(n, dtype=np.int64)
    cnt = np.zeros(n, dtype=np.int64)
    while np.any(v):
        cnt += v & 1
        v >>= 1
    return (cnt & 1).astype(float)


def _fibonacci_word(n: int) -> np.ndarray:
    """First n symbols of the Fibonacci word (1 -> 10, 0 -> 1 fixed point).

    Built via the concatenation recurrence F(k+1) = F(k) + F(k-1) with
    F(1) = "1", F(2) = "10", which equals iterating the substitution.
    """
    if n <= 0:
        return np.zeros(0, dtype=float)
    a, b = [1], [1, 0]
    while len(b) < n:
        a, b = b, b + a
    return np.asarray(b[:n], dtype=float)


# ---------------------------------------------------------------------------
# phase alignment machinery
# ---------------------------------------------------------------------------

def _phase_forecast(
    bits: np.ndarray,
    word: np.ndarray,
    window: int,
    phases: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align each trailing `window` of `bits` against all `phases` offsets of
    `word`, in both polarities.

    Returns (pred, match), both of length len(bits):
      pred[i]  in {-1, 0, +1} — next symbol prescribed by the best-matching
                substitution phase for the window ending at bit i (0 during
                warmup, i.e. i < window - 1);
      match[i] in [0, 1]      — agreement fraction of that best alignment
                (0 during warmup).
    """
    t = bits.size
    pred = np.zeros(t)
    match = np.zeros(t)
    if t < window or window < 1 or phases < 1:
        return pred, match
    if word.size < phases + window:  # need offset o+window for o < phases
        raise ValueError("substitution word too short for phases+window")

    # (t - window + 1, window): row j = bits window ending at index j+window-1
    b = np.lib.stride_tricks.sliding_window_view(bits, window).astype(float)
    # (phases, window): row o = word[o : o+window]
    s = np.lib.stride_tricks.sliding_window_view(
        word[: phases + window - 1], window
    ).astype(float)
    nxt = word[window: window + phases]  # next symbol for each offset

    # agreement counts via one matmul: matches on 1s plus matches on 0s
    agree = b @ s.T + (1.0 - b) @ (1.0 - s.T)          # (rows, phases)
    m_id = agree / float(window)                        # identity polarity
    # complement polarity: bits vs complemented word => match flips
    m_all = np.concatenate([m_id, 1.0 - m_id], axis=1)  # (rows, 2*phases)
    nxt_all = np.concatenate([nxt, 1.0 - nxt])          # (2*phases,)

    best = np.argmax(m_all, axis=1)
    rows = np.arange(m_all.shape[0])
    best_match = m_all[rows, best]
    best_next = nxt_all[best]

    match[window - 1:] = best_match
    pred[window - 1:] = 2.0 * best_next - 1.0
    return pred, match


def _substitution_weights(
    series: Series,
    word_fn,
    window: int,
    phases: int,
    threshold: float,
) -> np.ndarray:
    """Shared weights pipeline for both substitution strategies."""
    c = _closes(series)
    n = c.size
    if n < 3:
        return np.zeros(n)

    bits = (np.diff(c) > 0).astype(float)            # bits[i]: close[i]->close[i+1]
    w_eff = min(window, bits.size)
    if w_eff < 4:                                    # too short to mean anything
        return np.zeros(n)

    word = word_fn(phases + w_eff)
    pred, match = _phase_forecast(bits, word, w_eff, phases)

    # self-similarity gate: below threshold the nested pattern is broken -> 0
    denom = max(1e-9, 1.0 - threshold)
    edge = np.clip((match - threshold) / denom, 0.0, 1.0)
    sig_bits = pred * edge

    sig = np.zeros(n)
    sig[1:] = sig_bits                               # bit i known at close of bar i+1
    sig = np.clip(np.nan_to_num(sig), -1.0, 1.0)
    return _shift1(sig)


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------

@dataclass
class ThueMorsePhaseStrategy:
    """Trade the next symbol of the best-matching Thue-Morse phase.

    High match = the up/down sequence shows Thue-Morse-like nested balanced
    alternation (structured, cube-free, self-similar). The substitution phase
    then deterministically prescribes the next symbol; a broken pattern
    (match <= threshold) means regime change -> flat.
    """
    window: int = 16
    phases: int = 64
    threshold: float = 0.75
    name: str = "thue_morse_phase"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"thue_morse_phase(w={self.window},k={self.phases})"
        self.params = {"window": self.window, "phases": self.phases,
                       "threshold": self.threshold}

    def weights(self, series: Series) -> np.ndarray:
        return _substitution_weights(
            series, _thue_morse_word, self.window, self.phases, self.threshold
        )


@dataclass
class FibonacciWordFlowStrategy:
    """Trade the next symbol of the best-matching Fibonacci-word phase.

    The Fibonacci word is the archetypal Sturmian self-similar sequence:
    ~61.8% majority symbol with golden-ratio-spaced isolated minority symbols.
    A strong match (either polarity) = a structured fractal trend with
    self-similar pullback spacing: ride the prescribed next symbol (majority
    direction most of the time, stepping aside exactly at the pattern's
    pullback slots). Broken self-similarity -> flat.

    Defaults use Fibonacci numbers for window (13) and phases (55).
    """
    window: int = 13
    phases: int = 55
    threshold: float = 0.75
    name: str = "fibonacci_word_flow"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"fibonacci_word_flow(w={self.window},k={self.phases})"
        self.params = {"window": self.window, "phases": self.phases,
                       "threshold": self.threshold}

    def weights(self, series: Series) -> np.ndarray:
        return _substitution_weights(
            series, _fibonacci_word, self.window, self.phases, self.threshold
        )


STRATEGIES: Dict[str, type] = {
    "thue_morse_phase": ThueMorsePhaseStrategy,
    "fibonacci_word_flow": FibonacciWordFlowStrategy,
}
