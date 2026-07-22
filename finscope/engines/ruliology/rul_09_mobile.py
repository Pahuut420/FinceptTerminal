"""
rul_09_mobile — Mobile-automaton pointer-walk signals (NKS ch. 3).

Unlike a cellular automaton (all cells update in lockstep), a *mobile automaton*
has a single ACTIVE cell: at each step it updates that one cell from its local
neighborhood and then MOVES left or right per the rule — a wandering pointer
whose trajectory is a localized, history-dependent computation over the tape.

Encoding: the tape is the recent up/down return-sign bit-string (1 = up bar,
0 = down/flat bar). The active cell's walk is driven by the local bits, so the
walk trajectory is a nonlinear functional of the recent return-sign *sequence*
(order matters), not just its sum:

* On a persistent trend tape (runs of equal bits) the pointer drifts steadily
  in one direction — large, coherent net displacement.
* On a choppy/alternating tape the pointer oscillates around its origin —
  near-zero net displacement.

Sign of the signal = drift direction; magnitude = drift coherence. Two rules:

* ``mobile_drift``  — majority-smoothing update; readout = net displacement
  per step over a fixed step budget (drift-coherence observable).
* ``mobile_escape`` — bit-consuming (flip) update; readout = which side of the
  window the pointer escapes and how fast (escape-time observable). Trends let
  the walker burn through the tape and exit early; chop traps it inside.

Causality: the signal at bar t is computed from closes[.. t] only, then shifted
one bar via ``_shift1`` so weights[t] uses information through the close of
t-1. numpy + finscope only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1


def _sign_bits(c: np.ndarray) -> np.ndarray:
    """Return-sign tape: bits[i] = 1 if close[i+1] > close[i] else 0.

    bits[i] is known at the close of bar i+1 (uses closes i and i+1 only).
    """
    d = np.diff(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))
    return (d > 0.0).astype(np.int8)


def _drift_walk(tape: np.ndarray, steps: int) -> float:
    """Majority-update mobile automaton; returns net displacement / steps.

    Rule per step: read active cell c and neighbors l, r (edge cells use c as
    the missing neighbor); rewrite the active cell with majority(l, c, r);
    move right if c == 1 else left. Position clamps at the window edges but
    displacement keeps accumulating, so a saturated trend scores coherence 1.
    """
    w = int(tape.size)
    if w < 2 or steps <= 0:
        return 0.0
    t = tape.copy()
    pos = w // 2
    disp = 0
    for _ in range(steps):
        c = int(t[pos])
        left = int(t[pos - 1]) if pos > 0 else c
        right = int(t[pos + 1]) if pos < w - 1 else c
        t[pos] = 1 if (left + c + right) >= 2 else 0  # update active cell
        move = 1 if c == 1 else -1                     # then move the pointer
        disp += move
        pos = min(max(pos + move, 0), w - 1)
    return float(disp) / float(steps)


def _escape_walk(tape: np.ndarray, steps: int) -> float:
    """Bit-consuming mobile automaton; returns escape-side × escape-speed.

    Rule per step: read active cell c; flip it (the walker consumes the bit);
    move right if c == 1 else left. A trending tape is fuel the walker burns
    through to exit the window early; alternation traps it in a bounded
    oscillation. Readout: +/- (1 - k/steps) if it exits right/left at step k,
    0.0 if the budget expires with the walker still inside.
    """
    w = int(tape.size)
    if w < 2 or steps <= 0:
        return 0.0
    t = tape.copy()
    pos = w // 2
    for k in range(1, steps + 1):
        c = int(t[pos])
        t[pos] = 1 - c                      # consume the bit
        pos += 1 if c == 1 else -1          # then move the pointer
        if pos >= w:
            return 1.0 - k / float(steps)
        if pos < 0:
            return -(1.0 - k / float(steps))
    return 0.0


def _walk_signal(series: Series, window: int, steps: int, walker) -> np.ndarray:
    """Run `walker` on the trailing `window`-bit tape at every bar, causally."""
    c = _closes(series)
    n = c.size
    sig = np.zeros(n, dtype=float)
    if n < 3:
        return sig
    bits = _sign_bits(c)  # bits[i] known at close of bar i+1
    for i in range(window, n):
        tape = bits[i - window:i]  # last `window` return signs through bar i
        sig[i] = walker(tape, steps)
    sig = np.clip(np.nan_to_num(sig), -1.0, 1.0)
    return np.clip(np.nan_to_num(_shift1(sig)), -1.0, 1.0)


@dataclass
class MobileDriftStrategy:
    """Mobile-automaton drift walker: majority-update pointer over the
    return-sign tape; long when the walk drifts right (persistent up-bits),
    short when it drifts left, near-flat when the walk oscillates (chop).
    Magnitude = drift coherence (net displacement per step)."""
    window: int = 8
    steps: int = 16
    name: str = "mobile_drift"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(int(self.window), 2)
        self.steps = max(int(self.steps), 1)
        self.name = f"mobile_drift({self.window},{self.steps})"
        self.params = {"window": self.window, "steps": self.steps}

    def weights(self, series: Series) -> np.ndarray:
        return _walk_signal(series, self.window, self.steps, _drift_walk)


@dataclass
class MobileEscapeStrategy:
    """Mobile-automaton escape walker: bit-consuming pointer over the
    return-sign tape; a trend is fuel the walker burns to exit the window
    early (sign = exit side, magnitude = exit speed), while chop traps the
    walker inside the window until the step budget expires (flat)."""
    window: int = 8
    steps: int = 24
    name: str = "mobile_escape"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(int(self.window), 2)
        self.steps = max(int(self.steps), 1)
        self.name = f"mobile_escape({self.window},{self.steps})"
        self.params = {"window": self.window, "steps": self.steps}

    def weights(self, series: Series) -> np.ndarray:
        return _walk_signal(series, self.window, self.steps, _escape_walk)


STRATEGIES = {
    "mobile_drift": MobileDriftStrategy,
    "mobile_escape": MobileEscapeStrategy,
}
