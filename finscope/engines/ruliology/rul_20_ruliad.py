"""
rul_20_ruliad — Ruliad ensemble vote (Ruliology signal #20).

Concept
-------
The Ruliad (Wolfram Physics) is the entangled limit of *all* possible
computational rules run on all inputs. An observer never sees the whole thing —
it samples a coherent slice. The practical, tradeable analog: run a *panel* of
diverse elementary cellular automata on the same market initial condition and
let each rule act as one "possible physics" of the tape. Each rule computes the
tape forward and emits a next-bar direction call; an observer (the strategy)
aggregates the panel into a weighted vote where each rule's voice is its
*demonstrated trailing skill*. Diversity across Wolfram classes + skill
weighting is the point — no single rule is the model, the ensemble is.

Mapping (market -> CA symbols)
------------------------------
1. ENCODE  — return-sign bits: bit[t] = 1 iff close[t] > close[t-1], else 0
   (bit[0] is a pad). The recent `window` bits are the shared initial condition
   every rule in the panel runs on.
2. COMPUTE — each ECA rule in the panel (default spans the classes:
   30/54/60/90/110/150/184/225/232 — chaotic, complex, additive, particle,
   consensus) evolves the window `steps` synchronous updates with clamped
   boundaries. The rightmost (newest-information) cell of the evolved state is
   that rule's computed next bit -> a direction call in {-1, +1}.
3. SCORE   — each rule carries a trailing, Laplace-smoothed hit-rate: the call
   it made at bar t-1 is graded against the realized return sign of bar t,
   averaged over the last `skill_window` graded calls. With no evidence the
   prior is exactly 0.5 (coin-flip), so unproven rules start voiceless
   (vote strategy) or uniform (softmax strategy).
4. VOTE    — the position is the skill-weighted net vote of the panel,
   normalized to [-1, 1]. Sign = which way the skilled slice of the
   computational universe leans; magnitude = how strongly it agrees.

Causality
---------
pred[:, t] uses closes[.. t] only. The skill available when forming sig[t]
grades only predictions made at bars <= t-1 against outcomes at bars <= t.
Every signal then passes through ``_shift1``, so weights()[t] — the position
held over bar t — was decided at the close of t-1 from bars < t, including the
skill weights themselves. No lookahead anywhere.

Strategies
----------
- ruliad_vote    : hard skill gate. Only rules whose trailing hit-rate exceeds
  0.5 may vote, each weighted by its excess skill (hit-rate - 0.5). If no rule
  has proven itself, stay flat — an observer with no coherent slice trades
  nothing.
- ruliad_softmax : soft consensus. Every rule always votes, weighted by
  softmax(beta * (hit-rate - 0.5)). With no track record this degrades
  gracefully to the plain panel majority — the raw "consensus of the
  computational universe" — and concentrates onto skilled rules as evidence
  accumulates.

numpy only; CA evolved with integer lookup-table ops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1


# --------------------------------------------------------------------------- #
# ECA machinery                                                               #
# --------------------------------------------------------------------------- #

#: Default panel — a deliberate spread across Wolfram's classes:
#: 30, 225      class 3 chaotic          (irreducible computation)
#: 54, 110      class 4 complex          (localized structures / universality)
#: 60, 90, 150  additive / fractal       (linear rules, XOR physics)
#: 184          particle / traffic flow  (density transport)
#: 232          majority / consensus     (coarsening vote)
_DEFAULT_RULES: Tuple[int, ...] = (30, 54, 60, 90, 110, 150, 184, 225, 232)


def _rule_table(rule: int) -> np.ndarray:
    """8-entry lookup table for an elementary CA rule number.

    table[idx] with idx = 4*left + 2*center + right is the new cell value.
    """
    r = int(rule) & 0xFF
    return np.array([(r >> i) & 1 for i in range(8)], dtype=np.uint8)


def _eca_step(cells: np.ndarray, table: np.ndarray) -> np.ndarray:
    """One synchronous ECA update with clamped boundaries (edges see
    themselves duplicated outward — the past does not wrap onto the present)."""
    left = np.concatenate((cells[:1], cells[:-1])).astype(np.int64)
    right = np.concatenate((cells[1:], cells[-1:])).astype(np.int64)
    idx = (left << 2) | (cells.astype(np.int64) << 1) | right
    return table[idx]


def _predict_next_bit(window_bits: np.ndarray, table: np.ndarray, steps: int) -> int:
    """Evolve the bit window `steps` updates under the rule; the rightmost
    (newest-information) cell of the final state is the rule's computed next
    bit. `steps` deepens the light cone: after k steps that cell has integrated
    roughly the last k+1 observations under this rule's physics."""
    cells = window_bits.astype(np.uint8)
    for _ in range(max(1, int(steps))):
        cells = _eca_step(cells, table)
    return int(cells[-1])


def _sign_bits(closes: np.ndarray) -> np.ndarray:
    """Return-sign encoding: bits[t] = 1 iff close[t] > close[t-1], else 0.
    bits[0] is a pad (no return exists there). Known at the close of bar t."""
    n = closes.size
    bits = np.zeros(n, dtype=np.uint8)
    if n >= 2:
        d = np.diff(np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0))
        bits[1:] = (d > 0.0).astype(np.uint8)
    return bits


# --------------------------------------------------------------------------- #
# Shared panel state                                                          #
# --------------------------------------------------------------------------- #

def _panel_predictions(
    series: Series,
    rules: Tuple[int, ...],
    window: int,
    steps: int,
    min_bits: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Run the whole panel over the tape.

    Returns (n, pred, dir_act):
    - pred[r, t] in {-1, 0, +1}: rule r's next-bar direction call computed at
      the close of bar t from bits[.. t] only (0 = not enough history yet).
    - dir_act[t] in {-1, 0, +1}: the realized return sign of bar t.
    """
    c = _closes(series)
    n = c.size
    n_rules = len(rules)
    pred = np.zeros((n_rules, n), dtype=np.int8)
    dir_act = np.zeros(n)
    if n < 2 or n_rules == 0:
        return n, pred, dir_act

    bits = _sign_bits(c)
    tables: List[np.ndarray] = [_rule_table(r) for r in rules]
    w = max(3, int(window))
    mb = max(2, int(min_bits))

    for t in range(1, n):
        lo = max(1, t - w + 1)          # never include the bits[0] pad
        win = bits[lo:t + 1]
        if win.size < mb:
            continue
        for ri, table in enumerate(tables):
            pred[ri, t] = 1 if _predict_next_bit(win, table, steps) == 1 else -1

    dir_act[1:] = np.where(bits[1:] == 1, 1.0, -1.0)
    return n, pred, dir_act


def _trailing_skill(pred: np.ndarray, dir_act: np.ndarray, skill_window: int) -> np.ndarray:
    """Laplace-smoothed trailing hit-rate per rule, aligned so skill[r, t] is
    computable at the close of bar t: it grades calls made at bars <= t-1
    against outcomes realized at bars <= t, over the last `skill_window` graded
    calls. No evidence -> exactly 0.5 (the coin-flip prior)."""
    n_rules, n = pred.shape
    hits = np.zeros((n_rules, n))
    valid = np.zeros((n_rules, n))
    if n >= 2:
        p_prev = pred[:, :-1].astype(float)          # call made at t-1 ...
        act = dir_act[1:][None, :]                   # ... graded by bar t
        v = (p_prev != 0.0) & (act != 0.0)
        hits[:, 1:] = (v & (p_prev == act)).astype(float)
        valid[:, 1:] = v.astype(float)

    ch = np.cumsum(hits, axis=1)
    cv = np.cumsum(valid, axis=1)
    k = max(1, int(skill_window))
    hs, vs = ch.copy(), cv.copy()
    if n > k:
        hs[:, k:] = ch[:, k:] - ch[:, :-k]
        vs[:, k:] = cv[:, k:] - cv[:, :-k]
    return (hs + 1.0) / (vs + 2.0)                   # Laplace prior -> 0.5


# --------------------------------------------------------------------------- #
# Strategies                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class RuliadSkillVoteStrategy:
    """Skill-gated Ruliad vote: only rules whose trailing hit-rate beats the
    coin flip may vote, each weighted by its excess skill (hit-rate - 0.5).
    Position = normalized net vote in [-1, 1]; magnitude = how strongly the
    proven slice of the rule panel agrees. No proven rule -> flat."""
    window: int = 10
    steps: int = 3
    skill_window: int = 20
    min_bits: int = 3
    rules: Tuple[int, ...] = _DEFAULT_RULES
    name: str = "ruliad_vote"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(3, int(self.window))
        self.steps = max(1, int(self.steps))
        self.skill_window = max(2, int(self.skill_window))
        self.min_bits = max(2, int(self.min_bits))
        self.rules = tuple(int(r) & 0xFF for r in self.rules) or _DEFAULT_RULES
        self.name = f"ruliad_vote(w={self.window},k={self.skill_window},r={len(self.rules)})"
        self.params = {"window": self.window, "steps": self.steps,
                       "skill_window": self.skill_window, "min_bits": self.min_bits,
                       "rules": list(self.rules)}

    def weights(self, series: Series) -> np.ndarray:
        n, pred, dir_act = _panel_predictions(
            series, self.rules, self.window, self.steps, self.min_bits)
        if n == 0:
            return np.zeros(0)
        skill = _trailing_skill(pred, dir_act, self.skill_window)
        excess = np.clip(skill - 0.5, 0.0, None)     # coin-flip rules are voiceless
        active = (pred != 0).astype(float)
        w = excess * active
        denom = w.sum(axis=0)
        num = (w * pred).sum(axis=0)
        sig = np.where(denom > 1e-12, num / np.maximum(denom, 1e-12), 0.0)
        return _shift1(np.clip(np.nan_to_num(sig), -1.0, 1.0))


@dataclass
class RuliadSoftmaxConsensusStrategy:
    """Soft Ruliad consensus: every rule always votes, weighted by
    softmax(beta * (trailing hit-rate - 0.5)). With no track record this is
    the plain panel majority (the raw consensus of the computational
    universe); as evidence accumulates, weight concentrates onto the rules
    whose physics has actually been predicting this tape."""
    window: int = 10
    steps: int = 3
    skill_window: int = 20
    min_bits: int = 3
    beta: float = 8.0
    rules: Tuple[int, ...] = _DEFAULT_RULES
    name: str = "ruliad_softmax"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(3, int(self.window))
        self.steps = max(1, int(self.steps))
        self.skill_window = max(2, int(self.skill_window))
        self.min_bits = max(2, int(self.min_bits))
        self.beta = float(max(self.beta, 0.0))
        self.rules = tuple(int(r) & 0xFF for r in self.rules) or _DEFAULT_RULES
        self.name = f"ruliad_softmax(w={self.window},k={self.skill_window},b={self.beta})"
        self.params = {"window": self.window, "steps": self.steps,
                       "skill_window": self.skill_window, "min_bits": self.min_bits,
                       "beta": self.beta, "rules": list(self.rules)}

    def weights(self, series: Series) -> np.ndarray:
        n, pred, dir_act = _panel_predictions(
            series, self.rules, self.window, self.steps, self.min_bits)
        if n == 0:
            return np.zeros(0)
        skill = _trailing_skill(pred, dir_act, self.skill_window)
        w = np.exp(np.clip(self.beta * (skill - 0.5), -30.0, 30.0))
        w = w * (pred != 0)                          # only rules with a call vote
        denom = w.sum(axis=0)
        num = (w * pred).sum(axis=0)
        sig = np.where(denom > 1e-12, num / np.maximum(denom, 1e-12), 0.0)
        return _shift1(np.clip(np.nan_to_num(sig), -1.0, 1.0))


STRATEGIES = {
    "ruliad_vote": RuliadSkillVoteStrategy,
    "ruliad_softmax": RuliadSoftmaxConsensusStrategy,
}
