"""
rul_02_rule110 — Rule 110 localized-structure ("glider") momentum.

Ruliology thesis
----------------
Rule 110 is the canonical Class-4 elementary cellular automaton: Turing-universal,
with localized propagating structures (gliders) moving over a periodic ether.
We encode the recent return-sign window (up=1, down=0) as the initial row of a
Rule-110 CA and evolve it a few steps on a periodic lattice. The space-time
diagram is then read as a structure detector:

* A *live glider* — a coherent diagonal run of minority-phase cells persisting
  across several time steps at velocity -1, 0 or +1 — means the return sequence
  seeds self-propagating structure under Rule 110: the trend is "carrying
  structure", so we take the price momentum of the window.
* Net cell drift — the imbalance between rightward-propagating structures
  (drifting toward the newest bits, i.e. the present) and leftward ones —
  scales conviction: forward-drifting structure confirms the move, receding
  structure damps it.
* A dead / uniform grid (frozen ether, all-up or all-down seed, no coherent
  diagonals) reads as no exploitable structure: stay flat.

Causality: the signal at bar t is computed from returns ending at the close of
bar t only, then shifted one bar via `_shift1` — weights[t] is decided at the
close of bar t-1, so the backtester never peeks.

Only numpy + finscope. The ECA is implemented here from scratch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1

# ---- elementary cellular automaton: Rule 110 ------------------------------------

# Output bit for each 3-cell neighbourhood (left,self,right) read as the integer
# 4*l + 2*s + r.  110 = 0b01101110:  111->0 110->1 101->1 100->0 011->1 010->1
# 001->1 000->0.
_RULE110 = np.array([(110 >> i) & 1 for i in range(8)], dtype=np.int8)

_VELOCITIES: Tuple[int, ...] = (-1, 0, 1)
_MIN_BITS = 4          # minimum return bits before emitting any signal
_DEAD_FLIP_FRAC = 0.05  # below this mean flip rate the grid is considered dead
_EPS = 1e-12


def _evolve_rule110(bits: np.ndarray, steps: int) -> np.ndarray:
    """Evolve a binary row `steps` times under Rule 110 (periodic boundary).

    Returns the full (steps+1, W) space-time grid, initial row first.
    """
    row = bits.astype(np.int8)
    grid = np.empty((steps + 1, row.size), dtype=np.int8)
    grid[0] = row
    for k in range(steps):
        idx = (np.roll(row, 1) << 2) | (row << 1) | np.roll(row, -1)
        row = _RULE110[idx]
        grid[k + 1] = row
    return grid


def _diag_coherence(mask: np.ndarray, min_run: int) -> Dict[int, float]:
    """Coherent-diagonal score of `mask` cells per velocity v in {-1, 0, +1}.

    For each velocity the grid is sheared so a structure moving at v cells/step
    stays in one column; the score is the summed *excess* run length (longest
    vertical run per column minus (min_run - 1), floored at 0). Runs shorter
    than `min_run` — plain noise, not a glider — contribute nothing.
    """
    steps1, width = mask.shape
    m = mask.astype(np.float64)
    out: Dict[int, float] = {}
    for v in _VELOCITIES:
        run = np.zeros(width)
        best = np.zeros(width)
        for k in range(steps1):
            row = np.roll(m[k], -v * k)
            run = (run + 1.0) * row
            best = np.maximum(best, run)
        out[v] = float(np.maximum(0.0, best - (min_run - 1)).sum())
    return out


def _grid_features(bits: np.ndarray, steps: int, min_run: int):
    """Evolve the seed and extract (grid, flip_rate, ones_fraction)."""
    grid = _evolve_rule110(bits, steps)
    flips = float(np.mean(grid[1:] != grid[:-1])) if steps > 0 else 0.0
    ones = float(grid.mean())
    return grid, flips, ones


def _coherence_norm(width: int, steps: int, min_run: int) -> float:
    """Max attainable excess-run mass for one velocity (for normalisation)."""
    return float(width * max(1, steps + 1 - (min_run - 1)))


# ---- strategies -----------------------------------------------------------------

@dataclass
class Rule110GliderMomentum:
    """Momentum gated by Rule-110 glider coherence, tilted by net cell drift.

    Per bar: seed a Rule-110 CA with the last `window` return-sign bits (oldest
    left, newest right), evolve `steps` steps, then detect gliders as coherent
    diagonal runs (length >= min_run) of *minority-phase* cells — structure
    against the ether, so a near-uniform background never masquerades as
    signal. Direction is the sign of the window's net return; conviction is
    glider mass times a drift factor favouring rightward (toward-the-present)
    propagation. Dead, frozen or uniform grids give zero weight.
    """
    window: int = 16
    steps: int = 8
    min_run: int = 3
    gain: float = 4.0
    name: str = "rule110_glider_momentum"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(_MIN_BITS, int(self.window))
        self.steps = max(2, int(self.steps))
        self.min_run = max(2, min(int(self.min_run), self.steps))
        self.name = f"rule110_glider_mom(w={self.window},s={self.steps},r={self.min_run})"
        self.params = {"window": self.window, "steps": self.steps,
                       "min_run": self.min_run, "gain": self.gain}

    def _signal_at(self, rets: np.ndarray) -> float:
        bits = (rets > 0.0).astype(np.int8)
        if bits.min() == bits.max():          # uniform seed -> no structure
            return 0.0
        direction = float(np.sign(rets.sum()))
        if direction == 0.0:
            return 0.0
        grid, flips, _ = _grid_features(bits, self.steps, self.min_run)
        if flips < _DEAD_FLIP_FRAC:           # frozen grid -> dead ether
            return 0.0
        minority = 1 if grid.mean() <= 0.5 else 0
        coh = _diag_coherence(grid == minority, self.min_run)
        total = coh[-1] + coh[0] + coh[1]
        if total <= 0.0:                      # no glider survives min_run
            return 0.0
        drift = (coh[1] - coh[-1]) / (total + _EPS)   # + = toward the present
        strength = self.gain * total / (3.0 * _coherence_norm(bits.size, self.steps, self.min_run))
        conviction = min(1.0, strength) * (0.5 + 0.5 * drift)
        return direction * conviction

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        sig = np.zeros(n)
        if n >= _MIN_BITS + 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                rets = np.diff(c) / np.where(c[:-1] != 0.0, c[:-1], 1.0)
            rets = np.nan_to_num(rets)
            for t in range(_MIN_BITS, n):     # rets[:t] known at close of bar t
                sig[t] = self._signal_at(rets[max(0, t - self.window):t])
        w = _shift1(np.clip(np.nan_to_num(sig), -1.0, 1.0))
        return np.clip(np.nan_to_num(w), -1.0, 1.0)


@dataclass
class Rule110StructurePolarity:
    """Long/short from the polarity of the phase carrying coherent structure.

    Same CA and glider detector, but direction comes from *which bit phase*
    propagates: coherent diagonal runs of up-cells (1s) vs down-cells (0s),
    each counted only while that phase is genuinely structure (its grid share
    below `bg_cap` — a phase that has become the ether is background, not a
    glider). Signal = normalised polarity of up- vs down-structure mass,
    scaled by liveness; uniform seeds and dead grids stay flat.
    """
    window: int = 16
    steps: int = 8
    min_run: int = 3
    bg_cap: float = 0.85
    name: str = "rule110_structure_polarity"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(_MIN_BITS, int(self.window))
        self.steps = max(2, int(self.steps))
        self.min_run = max(2, min(int(self.min_run), self.steps))
        self.bg_cap = float(min(max(self.bg_cap, 0.5), 1.0))
        self.name = f"rule110_polarity(w={self.window},s={self.steps},r={self.min_run})"
        self.params = {"window": self.window, "steps": self.steps,
                       "min_run": self.min_run, "bg_cap": self.bg_cap}

    def _signal_at(self, rets: np.ndarray) -> float:
        bits = (rets > 0.0).astype(np.int8)
        if bits.min() == bits.max():
            return 0.0
        grid, flips, ones = _grid_features(bits, self.steps, self.min_run)
        if flips < _DEAD_FLIP_FRAC:
            return 0.0
        up = 0.0
        if ones < self.bg_cap:                # 1s are structure, not ether
            cu = _diag_coherence(grid == 1, self.min_run)
            up = cu[-1] + cu[0] + cu[1]
        dn = 0.0
        if (1.0 - ones) < self.bg_cap:        # 0s are structure, not ether
            cd = _diag_coherence(grid == 0, self.min_run)
            dn = cd[-1] + cd[0] + cd[1]
        total = up + dn
        if total <= 0.0:
            return 0.0
        liveness = min(1.0, total / (3.0 * _coherence_norm(bits.size, self.steps, self.min_run)) * 4.0)
        return (up - dn) / (total + _EPS) * liveness

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        sig = np.zeros(n)
        if n >= _MIN_BITS + 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                rets = np.diff(c) / np.where(c[:-1] != 0.0, c[:-1], 1.0)
            rets = np.nan_to_num(rets)
            for t in range(_MIN_BITS, n):
                sig[t] = self._signal_at(rets[max(0, t - self.window):t])
        w = _shift1(np.clip(np.nan_to_num(sig), -1.0, 1.0))
        return np.clip(np.nan_to_num(w), -1.0, 1.0)


STRATEGIES = {
    "rule110_glider_momentum": Rule110GliderMomentum,
    "rule110_structure_polarity": Rule110StructurePolarity,
}
