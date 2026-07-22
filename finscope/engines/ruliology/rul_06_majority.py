"""
Rule 232 — local MAJORITY vote — consensus/coarsening strategies.

Ruliology: ECA rule 232 maps each cell to the majority of its (left, self,
right) neighborhood. It is the canonical *consensus* rule: isolated dissenting
cells are absorbed, same-value domains of length >= 2 are stable and grow, and
the lattice coarsens rapidly toward a fixed point dominated by whichever side
held more ground. Class-1/2 dynamics — the interesting quantity is not the
trajectory but *which side wins* and *how decisively*.

Encoding: the recent window of return-sign bits (1 = up bar, 0 = down bar) is
the initial CA state. Evolving it under rule 232 performs a noise-robust,
locally-weighted vote over recent price direction: single contrarian bars are
voted away, while genuine directional runs (domains) survive and expand.

Two readouts of the same coarsening process:

* ``rul232_consensus``  — sign from the surviving majority, strength from the
  *dominant domain size* (largest run of the winning bit / window). A window
  that coarsens into one solid domain trades at full size; a fragmented
  stalemate trades small.
* ``rul232_coarsen``    — sign and magnitude from net consensus (2*mean - 1),
  damped by *time-to-consensus*: a window already at (or one step from) its
  fixed point formed consensus decisively; one that needs many majority rounds
  — or never settles under the step cap — is indecisive and gets discounted.

Causality: the signal at bar t is computed only from closes[.. t] and then
shifted forward one bar via ``_shift1``, so weights[t] is decided at the close
of t-1. numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1


# ---- rule-232 core ---------------------------------------------------------

def _rule232_step(cells: np.ndarray) -> np.ndarray:
    """One synchronous update of ECA rule 232 (3-cell majority vote).

    Clamped boundaries: edge cells see themselves duplicated outward, so
    consensus grows inward from stable edges instead of wrapping information
    around the window.
    """
    left = np.concatenate((cells[:1], cells[:-1]))
    right = np.concatenate((cells[1:], cells[-1:]))
    return ((left + cells + right) >= 2).astype(np.int8)


def _evolve_majority(cells: np.ndarray, max_steps: int) -> Tuple[np.ndarray, int, bool]:
    """Evolve under rule 232 until fixed point, capped at ``max_steps``.

    Returns (final_cells, steps_taken, converged). ``steps_taken`` counts
    state-changing updates only; an initially unanimous window converges in 0.
    """
    steps = 0
    for _ in range(max_steps):
        nxt = _rule232_step(cells)
        if np.array_equal(nxt, cells):
            return cells, steps, True
        cells = nxt
        steps += 1
    return cells, steps, bool(np.array_equal(_rule232_step(cells), cells))


def _sign_bits(closes: np.ndarray) -> np.ndarray:
    """Return-sign cells: bits[t] is 1 if bar t closed above bar t-1, else 0.

    bits[0] is a pad (no return exists there). Flat bars carry no directional
    information, so they inherit the previous bit (seeded down-neutral at 0).
    """
    n = closes.size
    bits = np.zeros(n, dtype=np.int8)
    prev = 0
    for i in range(1, n):
        d = closes[i] - closes[i - 1]
        if d > 0:
            prev = 1
        elif d < 0:
            prev = 0
        bits[i] = prev
    return bits


def _longest_run(cells: np.ndarray, value: int) -> int:
    """Length of the longest contiguous run of ``value`` in ``cells``."""
    best = cur = 0
    for c in cells:
        if c == value:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


# ---- strategies ------------------------------------------------------------

@dataclass
class MajorityConsensusStrategy:
    """Rule-232 consensus vote; size the trade by the winning domain's size.

    At each bar, the last ``window`` return-sign bits are coarsened to their
    rule-232 fixed point. Direction = the surviving majority side; strength =
    largest contiguous domain of the winning bit / window length, so a window
    that coarsens into one solid block trades at |w| ~ 1 while a fragmented
    stalemate trades near flat. Exact ties are flat.
    """
    window: int = 8
    min_bits: int = 3
    name: str = "rul232_consensus"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(3, int(self.window))
        self.min_bits = max(3, int(self.min_bits))
        self.name = f"rul232_consensus({self.window})"
        self.params = {"window": self.window, "min_bits": self.min_bits}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n < 3:
            return np.zeros(n, dtype=float)
        bits = _sign_bits(c)
        sig = np.zeros(n, dtype=float)
        for t in range(1, n):
            lo = max(1, t - self.window + 1)
            win = bits[lo:t + 1]
            if win.size < self.min_bits:
                continue
            final, _, _ = _evolve_majority(win.copy(), int(win.size))
            ones = int(final.sum())
            if 2 * ones == final.size:  # exact stalemate
                continue
            dominant = 1 if 2 * ones > final.size else 0
            direction = 1.0 if dominant == 1 else -1.0
            sig[t] = direction * (_longest_run(final, dominant) / float(final.size))
        return _shift1(np.clip(np.nan_to_num(sig), -1.0, 1.0))


@dataclass
class MajorityCoarseningStrategy:
    """Rule-232 net consensus, damped by how fast the consensus formed.

    Base signal = 2*mean(fixed-point cells) - 1 in [-1, 1] (net share of the
    window held by the winning side after coarsening). It is scaled by a
    time-to-consensus factor: 0 majority rounds needed -> full conviction;
    each extra round needed to settle linearly discounts it, and a window that
    never settles within the step cap is discounted a further 50%. Slowly
    formed consensus = contested tape = trade small.
    """
    window: int = 10
    min_bits: int = 3
    name: str = "rul232_coarsen"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(3, int(self.window))
        self.min_bits = max(3, int(self.min_bits))
        self.name = f"rul232_coarsen({self.window})"
        self.params = {"window": self.window, "min_bits": self.min_bits}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n < 3:
            return np.zeros(n, dtype=float)
        bits = _sign_bits(c)
        sig = np.zeros(n, dtype=float)
        for t in range(1, n):
            lo = max(1, t - self.window + 1)
            win = bits[lo:t + 1]
            if win.size < self.min_bits:
                continue
            cap = int(win.size)
            final, steps, converged = _evolve_majority(win.copy(), cap)
            base = 2.0 * float(final.mean()) - 1.0
            decisiveness = max(0.0, 1.0 - steps / (cap + 1.0))
            if not converged:
                decisiveness *= 0.5
            sig[t] = base * decisiveness
        return _shift1(np.clip(np.nan_to_num(sig), -1.0, 1.0))


STRATEGIES = {
    "rul232_consensus": MajorityConsensusStrategy,
    "rul232_coarsen": MajorityCoarseningStrategy,
}
