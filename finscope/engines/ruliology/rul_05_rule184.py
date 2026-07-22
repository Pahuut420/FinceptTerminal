"""
Rule 184 — traffic-flow / particle-hopping strategies (order-flow analog).

Elementary CA Rule 184 (0b10111000) is the canonical 1-D traffic model: a car
(1) moves one cell right iff the cell ahead is empty (0); density is conserved
and the system separates into a free-flow phase (low density: every car moves
each step) and a jammed phase (high density: gridlock clusters form and the
flux collapses). The fundamental diagram is J = min(rho, 1 - rho), and in any
configuration the number of ``10`` pairs equals the number of cars that will
move (equivalently, holes hopping left).

Market mapping: the road is the recent return-sign bit string (1 = up-move),
so density = fraction of up-moves. We evolve the road with *actual* Rule 184
steps — the transient does real computational work dissolving spurious jams —
then read the phase off the evolved configuration:

* free flow of the dominant move species (mobility high) -> the tape is
  absorbing that direction; momentum continues -> trade WITH the imbalance;
* gridlock of the dominant species (mobility low, i.e. one-way clusters that
  survive evolution) -> congestion / overextension -> fade the imbalance.

A literal one-sided reading of "density<0.5 => momentum, density>0.5 =>
reversal" is short in both regimes (down-trend momentum and up-trend fade);
Rule 184 is invariant under the particle-hole + mirror symmetry, so we apply
the same traffic logic to whichever species dominates, which restores
long/short symmetry.

Causality: for bar t the road uses return signs known at the close of t; the
signal is then shifted one bar (`_shift1`) so weights()[t] only uses
information available at the close of t-1. numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1

RULE = 184
# Output for neighborhood index (l<<2 | c<<1 | r): bit n of the rule number.
_TABLE = np.array([(RULE >> n) & 1 for n in range(8)], dtype=np.int8)


def _evolve_rule184(bits: np.ndarray, steps: int) -> np.ndarray:
    """Evolve a binary road `steps` generations under Rule 184 (periodic BC)."""
    b = bits.astype(np.int8)
    for _ in range(max(0, steps)):
        left = np.roll(b, 1)
        right = np.roll(b, -1)
        b = _TABLE[(left << 2) | (b << 1) | right]
    return b


def _return_bits(c: np.ndarray) -> np.ndarray:
    """Return-sign road: 1 for an up-move, 0 for down-or-flat. Length n-1;
    bit j is known at the close of bar j+1."""
    if c.size < 2:
        return np.zeros(0, dtype=np.int8)
    return (np.diff(c) > 0).astype(np.int8)


def _moving_pairs(b: np.ndarray) -> int:
    """Number of `10` pairs on the circular road = cars that move this step
    (= holes that move left). The Rule 184 flux numerator."""
    return int(np.sum(b & (1 - np.roll(b, -1))))


def _max_run_circular(x: np.ndarray) -> int:
    """Longest circular run of 1s in a binary array (a traffic jam length)."""
    n = x.size
    if n == 0 or not x.any():
        return 0
    if x.all():
        return n
    y = np.roll(x, -int(np.argmin(x)))  # rotate so the road starts at a hole
    best = cur = 0
    for v in y:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


@dataclass
class Rule184FlowStrategy:
    """Dominant-species mobility: trend while traffic flows, fade gridlock.

    Per bar, the last `window` return signs seed the road; `steps` Rule 184
    generations are applied. Let rho be the up-move density, dir = sign of the
    imbalance, and m = (moving `10` pairs) / (dominant species count) — the
    fraction of the dominant species free to move in the evolved state
    (steady-state m = min(rho,1-rho)/max(rho,1-rho)). Signal = dir*(2m - 1):
    a moderate imbalance still flowing (m > 1/2, i.e. dominant density < 2/3)
    continues -> trade with it; a one-way road whose jams survive evolution
    (m < 1/2) is congested -> fade it.
    """
    window: int = 12
    steps: int = 6
    name: str = "rule184_flow"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(4, int(self.window))
        self.steps = max(1, int(self.steps))
        self.name = f"rule184_flow({self.window},{self.steps})"
        self.params = {"window": self.window, "steps": self.steps}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        sig = np.zeros(n, dtype=float)
        if n < 3:
            return sig
        bits = _return_bits(c)  # bits[j] known at close of bar j+1
        w = self.window
        for t in range(n):
            if t < w:  # not enough history for a full road
                continue
            road = bits[t - w:t]  # last w return signs, known at close of t
            n_up = int(road.sum())
            imbalance = 2.0 * n_up / w - 1.0
            direction = float(np.sign(imbalance))
            if direction == 0.0:
                continue  # balanced road: no dominant species, stay flat
            final = _evolve_rule184(road, self.steps)
            dominant = max(n_up, w - n_up)
            mobility = _moving_pairs(final) / dominant  # in [0, 1]
            sig[t] = direction * (2.0 * mobility - 1.0)
        sig = np.clip(np.nan_to_num(sig), -1.0, 1.0)
        return _shift1(sig)


@dataclass
class Rule184JamStrategy:
    """Congestion reversal: fade the direction whose jam survives evolution.

    Transient clusters dissolve under Rule 184 when the species is below
    critical density, so any long run remaining after `steps` generations is a
    genuine jam. Signal = gain * (down-jam - up-jam) / window: a persistent
    up-car pile-up (one-way buying) means gridlock -> short the reversal; a
    persistent hole convoy (one-way selling, a jam on the mirror road) ->
    long. Purely contrarian, sized by relative jam length.
    """
    window: int = 16
    steps: int = 8
    gain: float = 1.6
    name: str = "rule184_jam"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(4, int(self.window))
        self.steps = max(1, int(self.steps))
        self.gain = float(self.gain)
        self.name = f"rule184_jam({self.window},{self.steps})"
        self.params = {"window": self.window, "steps": self.steps,
                       "gain": self.gain}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        sig = np.zeros(n, dtype=float)
        if n < 3:
            return sig
        bits = _return_bits(c)
        w = self.window
        for t in range(n):
            if t < w:
                continue
            road = bits[t - w:t]
            final = _evolve_rule184(road, self.steps)
            jam_up = _max_run_circular(final) / w          # surviving car jam
            jam_dn = _max_run_circular(1 - final) / w      # surviving hole jam
            sig[t] = self.gain * (jam_dn - jam_up)
        sig = np.clip(np.nan_to_num(sig), -1.0, 1.0)
        return _shift1(sig)


STRATEGIES = {
    "rule184_flow": Rule184FlowStrategy,
    "rule184_jam": Rule184JamStrategy,
}
