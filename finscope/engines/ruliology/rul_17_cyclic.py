"""
Cyclic Cellular Automaton (Griffeath) — rotational phase-timing strategies.

Ruliology: in the cyclic CA, states live on a cycle 0 -> 1 -> ... -> k-1 -> 0
and a cell advances to its successor state iff a neighbor already holds that
successor. From noise, the rule self-organizes into traveling waves and spiral
cores with an emergent, self-sustained periodicity: phase fronts "eat" their
predecessors and rotation, once established, persists. The interesting
observables are the *aligned phase* of the wave and whether a *front is about
to arrive* at a given site.

Encoding: each bar is assigned one of k cyclic phase-states from the recent
return structure. Detrend price against a trailing SMA (x = close - sma) and
take the oscillator's causal velocity (dx). The pair (x, dx) traces a rotation
in phase space over any price cycle; theta = atan2(-dx, x) increases
monotonically through an idealized oscillation, so quantizing theta into k
bins yields states that a genuine price cycle traverses in CA order
0 -> 1 -> ... -> k-1 -> 0. The trailing window of these states is the CCA
lattice; evolving it lets locally-consistent rotation synchronize and noise
stay incoherent.

Two readouts of the same rotation process:

* ``cca_phase``     — evolve the lattice, then take the circular mean of the
  cell phases. Direction = whether the aligned phase sits in the rising half
  of the cycle (-sin of the mean angle); strength = wave coherence, the
  circular resultant length R in [0, 1]. Incoherent (no wave) windows trade
  near flat automatically since sig = -Im(mean unit phasor) = -R*sin(theta).
* ``cca_wavefront`` — the newest bar is the lattice's *phase leader*, and the
  CCA's successor-eats-predecessor rule makes the leading front propagate
  backward through the window, converting predecessor cells to the leader's
  phase. After evolution, the contiguous leader domain at the recent edge is
  the wavefront's spatial support: a wide front = a well-backed cycle leg.
  Position from the leader's phase read with a consistency-proportional
  half-bin lead (anticipating the next leg); strength = wavefront extent x
  forward-rotation consistency of the raw lattice. Retrograde or churning
  windows produce no trade.

Causality: every per-bar quantity (SMA, oscillator, normalization, phase
state) uses only closes[.. t]; the signal at bar t is then shifted forward one
bar via ``_shift1``, so weights[t] is decided at the close of t-1. numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1

_TWO_PI = 2.0 * np.pi


# ---- cyclic-CA core --------------------------------------------------------

def _cca_step(cells: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """One synchronous Griffeath cyclic-CA update on a 1D lattice.

    A cell in state s advances to (s + 1) % k iff its left or right neighbor
    already holds (s + 1) % k. Clamped boundaries: edge cells see themselves
    duplicated outward, so waves propagate from the interior instead of
    wrapping information around the window. Returns (next_cells, advanced
    mask).
    """
    left = np.concatenate((cells[:1], cells[:-1]))
    right = np.concatenate((cells[1:], cells[-1:]))
    successor = (cells + 1) % k
    advanced = (left == successor) | (right == successor)
    return np.where(advanced, successor, cells), advanced


def _causal_sma(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing SMA that falls back to an expanding mean before w bars."""
    out = np.empty_like(x, dtype=float)
    csum = np.cumsum(x)
    for i in range(x.size):
        lo = max(0, i - w + 1)
        total = csum[i] - (csum[lo - 1] if lo > 0 else 0.0)
        out[i] = total / (i - lo + 1)
    return out


def _phase_states(c: np.ndarray, k: int, ma_window: int, norm_lb: int) -> np.ndarray:
    """Map each bar to a cyclic phase-state in {0, .., k-1} — causally.

    x = close - trailing SMA (the detrended oscillator), dx = its one-bar
    velocity. Both are scaled by their own trailing standard deviation over
    ``norm_lb`` bars so the (x, dx) orbit is roughly circular, then
    theta = atan2(-dx, x) (increasing through a price cycle) is quantized
    into k equal bins. Bars with no usable variance land in state 0.
    """
    n = c.size
    x = c - _causal_sma(c, ma_window)
    dx = np.zeros(n, dtype=float)
    if n > 1:
        dx[1:] = x[1:] - x[:-1]
    states = np.zeros(n, dtype=np.int64)
    for t in range(1, n):
        lo = max(0, t - norm_lb + 1)
        sx = float(np.std(x[lo:t + 1]))
        sdx = float(np.std(dx[lo:t + 1]))
        xh = x[t] / sx if sx > 1e-12 else 0.0
        dxh = dx[t] / sdx if sdx > 1e-12 else 0.0
        theta = float(np.arctan2(-dxh, xh)) % _TWO_PI
        states[t] = int(theta / _TWO_PI * k) % k
    return states


def _state_phasors(cells: np.ndarray, k: int) -> np.ndarray:
    """Unit phasors at each cell's bin-center angle 2*pi*(s + 0.5)/k."""
    ang = _TWO_PI * (cells.astype(float) + 0.5) / k
    return np.exp(1j * ang)


def _rotation_consistency(cells: np.ndarray, k: int) -> float:
    """Net forward-rotation share of the raw lattice, in [0, 1].

    Consecutive-bar state steps are mapped to the shortest signed cyclic
    difference in [-k/2, k/2); consistency = max(0, net / total) so that only
    coherent *forward* rotation (the direction the CCA sustains) scores, while
    retrograde or churning windows score 0.
    """
    if cells.size < 2:
        return 0.0
    d = np.diff(cells)
    d = (d + k // 2) % k - k // 2
    total = float(np.abs(d).sum())
    if total <= 0.0:
        return 0.0
    return max(0.0, float(d.sum()) / total)


# ---- strategies ------------------------------------------------------------

@dataclass
class CyclicPhaseStrategy:
    """Griffeath-CCA aligned phase; size the trade by wave coherence.

    At each bar, the last ``window`` phase-states form the CCA lattice and are
    evolved ``steps`` sweeps so locally-consistent rotation synchronizes into
    a wave. The signal is -Im(mean unit phasor) of the evolved lattice =
    R * (-sin(mean angle)): long when the aligned phase sits in the rising
    half of the cycle, short in the falling half, and automatically near flat
    when phases are incoherent (R ~ 0, i.e. no emergent wave).
    """
    k: int = 6
    window: int = 12
    ma_window: int = 5
    steps: int = 4
    norm_lb: int = 10
    name: str = "cca_phase"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.k = max(3, int(self.k))
        self.window = max(4, int(self.window))
        self.ma_window = max(2, int(self.ma_window))
        self.steps = max(1, int(self.steps))
        self.norm_lb = max(3, int(self.norm_lb))
        self.name = f"cca_phase(k={self.k},{self.window})"
        self.params = {"k": self.k, "window": self.window,
                       "ma_window": self.ma_window, "steps": self.steps,
                       "norm_lb": self.norm_lb}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n < 3:
            return np.zeros(n, dtype=float)
        states = _phase_states(c, self.k, self.ma_window, self.norm_lb)
        sig = np.zeros(n, dtype=float)
        for t in range(2, n):
            lo = max(1, t - self.window + 1)
            lattice = states[lo:t + 1].copy()
            if lattice.size < 3:
                continue
            for _ in range(self.steps):
                lattice, _adv = _cca_step(lattice, self.k)
            z = _state_phasors(lattice, self.k).mean()
            sig[t] = -z.imag  # = coherence R * (-sin(aligned phase))
        return _shift1(np.clip(np.nan_to_num(sig), -1.0, 1.0))


@dataclass
class CyclicWavefrontStrategy:
    """CCA leading-front extent behind the most recent bar.

    In a forward-rotating window the newest cell holds the most advanced
    phase, so under the CCA the leading front propagates *backward*: the
    leader's successor-pressure converts older predecessor cells to its
    phase. After ``steps`` sweeps, the contiguous run of leader-phase cells
    at the recent edge measures how much of the window the current cycle leg
    has organized — its wavefront extent. Direction = -sin of the leader's
    phase angle advanced by ``0.5 * consistency`` of a bin (a phase lead
    proportional to how reliably the tape rotates forward); strength =
    extent * consistency. Windows with no net forward rotation trade flat.
    """
    k: int = 6
    window: int = 12
    ma_window: int = 5
    steps: int = 5
    norm_lb: int = 10
    name: str = "cca_wavefront"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.k = max(3, int(self.k))
        self.window = max(4, int(self.window))
        self.ma_window = max(2, int(self.ma_window))
        self.steps = max(1, int(self.steps))
        self.norm_lb = max(3, int(self.norm_lb))
        self.name = f"cca_wavefront(k={self.k},{self.window})"
        self.params = {"k": self.k, "window": self.window,
                       "ma_window": self.ma_window, "steps": self.steps,
                       "norm_lb": self.norm_lb}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n < 3:
            return np.zeros(n, dtype=float)
        states = _phase_states(c, self.k, self.ma_window, self.norm_lb)
        sig = np.zeros(n, dtype=float)
        for t in range(2, n):
            lo = max(1, t - self.window + 1)
            raw = states[lo:t + 1]
            if raw.size < 3:
                continue
            consistency = _rotation_consistency(raw, self.k)
            if consistency <= 0.0:
                continue
            lattice = raw.copy()
            for _ in range(self.steps):
                lattice, _adv = _cca_step(lattice, self.k)
            leader = int(lattice[-1])
            run = 1  # contiguous leader-phase cells ending at the recent edge
            for j in range(lattice.size - 2, -1, -1):
                if int(lattice[j]) != leader:
                    break
                run += 1
            extent = run / float(lattice.size)
            angle = _TWO_PI * (leader + 0.5 + 0.5 * consistency) / self.k
            sig[t] = -np.sin(angle) * extent * consistency
        return _shift1(np.clip(np.nan_to_num(sig), -1.0, 1.0))


STRATEGIES = {
    "cca_phase": CyclicPhaseStrategy,
    "cca_wavefront": CyclicWavefrontStrategy,
}
