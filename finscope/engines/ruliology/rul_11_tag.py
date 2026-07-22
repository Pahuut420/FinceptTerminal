"""
rul_11_tag — tag-system dynamics as a regime-predictability signal (NKS ch. 3).

A tag system is a string-rewriting system: repeatedly delete the first m symbols
of a string and, depending on what the *first* deleted symbol was, append a
fixed production word to the end. Despite this triviality, m=2 tag systems are
Turing-complete and exhibit the full NKS behavioural spectrum: fast halting
(class 1), short cycles (class 2), and long, complex, seemingly irreducible
transients (class 3/4) before settling.

Encoding: the recent return-sign sequence ('1' = up bar, '0' = down bar) is the
initial tag string. We run an m=2 tag system on it and classify the orbit:

  * halts fast / falls into a short cycle  -> compressible, predictable regime
                                              -> trade the majority direction.
  * long complex transient (budget spent)  -> computationally irreducible
                                              -> no edge, stay flat (or fade).

Default productions ('1' -> "10", '0' -> "0") are contractive-on-down /
measure-preserving-on-up, so the string length never grows past the window:
all-down strings halt in ~w/1 steps, all-up strings lock into short cycles, and
mixed strings wander through long transients whose length measures regime
complexity. Growth productions (e.g. '1' -> "110", Post-style) can be passed as
params for expanding, class-4 dynamics; orbits that outgrow `max_len` are
treated as non-compressible (weight decays to 0).

Causality: signal at bar t is computed from return signs up to and including
bar t, then shifted one bar via `_shift1` — weights()[t] only uses information
available at the close of bar t-1.

Strategies
----------
tag2_halt_trend      : direction = evolving-string majority, size = orbit
                       predictability (1 near fast halt / short cycle, 0 for an
                       irreducible transient).
tag2_transient_fade  : complement — fades the majority exactly when the orbit
                       is irreducible (complex transient = crowded, overshot
                       tape), flat when the tag system is predictable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1


# ---------------------------------------------------------------------------
# core tag-system machinery
# ---------------------------------------------------------------------------

def _run_tag_system(
    initial: str,
    prod1: str,
    prod0: str,
    m: int,
    max_steps: int,
    max_len: int,
) -> Tuple[float, float]:
    """Run the m-tag system from `initial` and score the orbit.

    Returns (direction, predictability):
      direction      in {-1, 0, +1} — majority symbol over the evolving orbit
                     ('1' majority -> +1, '0' majority -> -1, tie -> 0).
      predictability in [0, 1] — 1 for an immediate halt / tight cycle,
                     decaying to 0 as the transient consumes the step budget;
                     exactly 0 when the budget is spent or the string diverges.
    """
    s = initial
    seen: Dict[str, int] = {}

    halted = False
    cycle_at = -1
    cycle_len = 0
    step = 0

    while step < max_steps:
        if len(s) < m:                      # halting condition
            halted = True
            break
        if s in seen:                       # exact state recurrence -> cycle
            cycle_at = seen[s]
            cycle_len = step - seen[s]
            break
        seen[s] = step
        head = s[0]
        s = s[m:] + (prod1 if head == "1" else prod0)
        if len(s) > max_len:                # diverging orbit: not compressible
            step = max_steps
            break
        step += 1

    # Majority symbol of the evolving string: score the initial tape plus the
    # surviving (final) tape — what the dynamics kept alive dominates.
    ones_score = initial.count("1") + s.count("1")
    zeros_score = (len(initial) - initial.count("1")) + (len(s) - s.count("1"))
    if ones_score > zeros_score:
        direction = 1.0
    elif zeros_score > ones_score:
        direction = -1.0
    else:
        direction = 0.0

    if halted:
        predictability = max(0.0, 1.0 - step / float(max_steps))
    elif cycle_at >= 0:
        # discount by transient length before the cycle plus the cycle period
        predictability = max(0.0, 1.0 - (cycle_at + cycle_len) / float(max_steps))
    else:                                   # budget exhausted or diverged
        predictability = 0.0
    return direction, predictability


def _signs_to_string(window_signs: np.ndarray) -> str:
    return "".join("1" if v > 0 else "0" for v in window_signs)


def _tag_signal_series(
    c: np.ndarray,
    window: int,
    prod1: str,
    prod0: str,
    m: int,
    max_steps: int,
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-bar (direction, predictability) arrays, computed causally at each
    bar t from the return signs of bars t-window+1 .. t."""
    n = c.size
    direction = np.zeros(n, dtype=float)
    predict = np.zeros(n, dtype=float)
    if n < 2:
        return direction, predict

    diffs = np.diff(c)                      # diffs[i] = c[i+1] - c[i]
    signs = (diffs > 0).astype(float)       # sign of return for bars 1..n-1
    cache: Dict[str, Tuple[float, float]] = {}

    for t in range(window, n):
        # signs for bars t-window+1 .. t  <=>  diffs indices t-window .. t-1
        bits = _signs_to_string(signs[t - window:t])
        hit = cache.get(bits)
        if hit is None:
            hit = _run_tag_system(bits, prod1, prod0, m, max_steps, max_len)
            cache[bits] = hit
        direction[t], predict[t] = hit
    return direction, predict


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------

@dataclass
class TagHaltTrendStrategy:
    """Trade the tape's majority direction, sized by tag-orbit predictability.

    Fast halt / short cycle of the m=2 tag system on the return-sign string
    means the recent regime is computationally compressible -> full-size trend
    position in the majority direction. A transient that exhausts the step
    budget (or diverges) is treated as irreducible -> flat.
    """
    window: int = 8
    prod1: str = "10"
    prod0: str = "0"
    m: int = 2
    max_steps: int = 40
    name: str = "tag2_halt_trend"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(2, int(self.window))
        self.m = max(1, int(self.m))
        self.max_steps = max(1, int(self.max_steps))
        self.name = f"tag2_halt_trend(w={self.window},p1={self.prod1},p0={self.prod0})"
        self.params = {
            "window": self.window, "prod1": self.prod1, "prod0": self.prod0,
            "m": self.m, "max_steps": self.max_steps,
        }

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        if c.size == 0:
            return np.zeros(0)
        direction, predict = _tag_signal_series(
            c, self.window, self.prod1, self.prod0,
            self.m, self.max_steps, max_len=4 * self.window,
        )
        sig = np.clip(direction * predict, -1.0, 1.0)
        return np.nan_to_num(_shift1(sig))


@dataclass
class TagTransientFadeStrategy:
    """Fade the majority when the tag orbit is irreducible; flat when simple.

    The complement of `tag2_halt_trend`: a long complex transient means the
    recent tape has no short description — choppy, crowded, overshoot-prone —
    so lean *against* the majority symbol, scaled by (1 - predictability).
    When the orbit halts fast or cycles tightly the regime is trending and this
    strategy stands aside.
    """
    window: int = 8
    prod1: str = "10"
    prod0: str = "0"
    m: int = 2
    max_steps: int = 40
    fade_scale: float = 0.75
    name: str = "tag2_transient_fade"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(2, int(self.window))
        self.m = max(1, int(self.m))
        self.max_steps = max(1, int(self.max_steps))
        self.fade_scale = float(np.clip(self.fade_scale, 0.0, 1.0))
        self.name = (
            f"tag2_transient_fade(w={self.window},p1={self.prod1},p0={self.prod0})"
        )
        self.params = {
            "window": self.window, "prod1": self.prod1, "prod0": self.prod0,
            "m": self.m, "max_steps": self.max_steps, "fade_scale": self.fade_scale,
        }

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        if c.size == 0:
            return np.zeros(0)
        direction, predict = _tag_signal_series(
            c, self.window, self.prod1, self.prod0,
            self.m, self.max_steps, max_len=4 * self.window,
        )
        sig = np.clip(-direction * (1.0 - predict) * self.fade_scale, -1.0, 1.0)
        return np.nan_to_num(_shift1(sig))


STRATEGIES = {
    "tag2_halt_trend": TagHaltTrendStrategy,
    "tag2_transient_fade": TagTransientFadeStrategy,
}
