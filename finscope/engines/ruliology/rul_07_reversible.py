"""
rul_07_reversible — second-order (reversible) cellular-automaton memory.

A second-order ECA evolves *two* time layers: ``next = rule(current) XOR previous``.
Because ``previous = rule(current) XOR next``, the dynamics is exactly
time-reversible — the update is a permutation of the (prev, cur) state space, so
every orbit is a pure cycle: no attractors, no dissipation, information is
conserved and every configuration recurs. Wolfram (NKS ch. 9) and Fredkin's
"R"-rules (90R, 150R) are the canonical examples.

Market reading: encode recent return-sign bits as the two CA layers. If price
dynamics carries reversible (information-conserving) memory, shocks do not
dissipate — they echo back after a characteristic recurrence time. That implies
mean-reversion with long memory:

* ``ReversibleEchoFadeStrategy`` — measures the actual recurrence period of the
  second-order orbit seeded by the live bit history and fades the current price
  deviation over a lookback derived from that period (short cycle -> fast echo
  -> tight fade window; long cycle -> slow echo -> wide window).
* ``ReversibleParityEchoStrategy`` — takes the one-step second-order rule-150
  echo prediction of the *next* return-sign bit (XOR parity naturally fades
  streaks: a run of ups predicts a down) and sizes it by the predictor's own
  trailing causal hit-rate, so it only trades when the reversible-memory model
  has recently beaten a coin flip.

Causality: all per-bar computations use data up to and including bar t only,
then the signal is shifted one bar forward via ``_shift1`` — weights()[t] is
decided at the close of bar t-1. Outputs are finite and clipped to [-1, 1];
safe for series as short as 3 bars (returns zeros until warm).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1


# ---- second-order (reversible) ECA machinery -----------------------------------

def _rule_table(rule: int) -> np.ndarray:
    """8-entry lookup table for an elementary CA rule number (Wolfram code)."""
    return np.array([(int(rule) >> i) & 1 for i in range(8)], dtype=np.uint8)


def _eca_step(row: np.ndarray, table: np.ndarray) -> np.ndarray:
    """One first-order ECA step with periodic boundary conditions."""
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right
    return table[idx]


def _second_order_step(prev: np.ndarray, cur: np.ndarray,
                       table: np.ndarray) -> np.ndarray:
    """Reversible update: next = rule(current) XOR previous."""
    return _eca_step(cur, table) ^ prev


def _sign_bits(closes: np.ndarray) -> np.ndarray:
    """Return-sign bits; bits[j] = 1 iff the return ending at bar j+1 is > 0."""
    diff = np.diff(closes)
    prevc = closes[:-1]
    safe = np.abs(prevc) > 1e-12
    rets = np.where(safe, diff / np.where(safe, prevc, 1.0), 0.0)
    return (rets > 0.0).astype(np.uint8)


def _recurrence_period(prev: np.ndarray, cur: np.ndarray, table: np.ndarray,
                       max_period: int) -> int:
    """Steps until the (prev, cur) layer pair first returns to its start.

    The second-order map is a bijection, so the orbit is a cycle and recurrence
    is guaranteed; we cap the search at ``max_period`` for bounded cost.
    """
    p0, c0 = prev.copy(), cur.copy()
    p, c = p0, c0
    for step in range(1, max_period + 1):
        p, c = c, _second_order_step(p, c, table)
        if np.array_equal(p, p0) and np.array_equal(c, c0):
            return step
    return max_period


# ---- strategies ----------------------------------------------------------------

@dataclass
class ReversibleEchoFadeStrategy:
    """Fade price deviation over a lookback set by the reversible recurrence period.

    Seed a second-order ECA (default rule 90R) with the last ``cells`` return-sign
    bits as the current layer and the bits one bar older as the previous layer.
    Measure the orbit's recurrence period P (capped), map it log-scale onto a
    lookback L in [min_lb, max_lb], and fade the z-scored deviation of the close
    from its L-bar mean. Information-conserving recurrence = the shock echoes
    back, so deviations are traded against, with the memory horizon chosen by
    the CA itself rather than a fixed window.
    """
    cells: int = 8
    rule: int = 90
    max_period: int = 64
    min_lb: int = 2
    max_lb: int = 12
    z_cap: float = 2.0
    name: str = "rul07_reversible_echo_fade"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"rul07_echo_fade(r{self.rule},k{self.cells})"
        self.params = {"cells": self.cells, "rule": self.rule,
                       "max_period": self.max_period, "min_lb": self.min_lb,
                       "max_lb": self.max_lb, "z_cap": self.z_cap}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        sig = np.zeros(n)
        k = max(2, int(self.cells))
        if n < k + 2:
            return sig  # not enough bits for two layers — stay flat
        bits = _sign_bits(c)  # length n-1; bits[:t] known at close of bar t
        table = _rule_table(self.rule)
        log_den = np.log2(self.max_period + 1.0)
        for t in range(k + 1, n):
            cur = bits[t - k:t]          # k bits ending at bar t
            prev = bits[t - k - 1:t - 1]  # same window, one bar earlier
            p = _recurrence_period(prev, cur, table, self.max_period)
            frac = min(1.0, np.log2(p + 1.0) / log_den)
            lb = int(round(self.min_lb + frac * (self.max_lb - self.min_lb)))
            lb = max(self.min_lb, min(lb, self.max_lb, t))
            win = c[t - lb + 1:t + 1]
            sd = float(win.std())
            if sd > 1e-12:
                z = (c[t] - float(win.mean())) / sd
                sig[t] = -np.clip(z, -self.z_cap, self.z_cap) / self.z_cap
        return np.nan_to_num(np.clip(_shift1(sig), -1.0, 1.0))


@dataclass
class ReversibleParityEchoStrategy:
    """Trade the second-order rule-150 one-step echo bit, sized by causal hit-rate.

    The reversible update predicts the newest cell's next state as
    ``left XOR center XOR right XOR previous`` — a pure parity echo of the two
    most recent bit layers. Parity fades streaks (three ups with an up history
    predict a down), i.e. mean-reversion with two-layer memory. The direction is
    scaled by ``max(0, 2*hitrate - 1)`` over the predictor's own trailing
    realized outcomes, so the strategy goes flat whenever the reversible-memory
    model stops beating a coin flip. Hit-rate at bar t uses only predictions
    made at t-1 or earlier against returns realized by bar t — strictly causal.
    """
    cells: int = 8
    rule: int = 150
    hit_window: int = 12
    min_obs: int = 5
    name: str = "rul07_reversible_parity_echo"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"rul07_parity_echo(r{self.rule},k{self.cells})"
        self.params = {"cells": self.cells, "rule": self.rule,
                       "hit_window": self.hit_window, "min_obs": self.min_obs}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        sig = np.zeros(n)
        k = max(2, int(self.cells))
        if n < k + 2:
            return sig
        bits = _sign_bits(c)
        table = _rule_table(self.rule)

        # direction[t]: predicted sign of the return ending at bar t+1,
        # decided with bits known at the close of bar t.
        direction = np.zeros(n)
        for t in range(k + 1, n):
            cur = bits[t - k:t]
            prev = bits[t - k - 1:t - 1]
            nxt = _second_order_step(prev, cur, table)
            direction[t] = 1.0 if nxt[-1] else -1.0

        hits: list = []
        for t in range(k + 2, n):
            # score yesterday's prediction against the return realized at bar t
            if direction[t - 1] != 0.0:
                realized = 1.0 if bits[t - 1] else -1.0
                hits.append(1.0 if realized == direction[t - 1] else 0.0)
            if direction[t] != 0.0 and len(hits) >= self.min_obs:
                h = float(np.mean(hits[-self.hit_window:]))
                sig[t] = direction[t] * max(0.0, 2.0 * h - 1.0)
        return np.nan_to_num(np.clip(_shift1(sig), -1.0, 1.0))


STRATEGIES = {
    "rul07_reversible_echo_fade": ReversibleEchoFadeStrategy,
    "rul07_reversible_parity_echo": ReversibleParityEchoStrategy,
}
