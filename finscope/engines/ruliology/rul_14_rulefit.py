"""
rul_14_rulefit — Nearest-ECA-rule predictor (Ruliology signal #14).

Concept
-------
NKS-style rule-space search: among the 256 elementary cellular automata, find
the rule whose one-step dynamics best reproduce the observed binary sequence,
then use THAT rule to predict the next bit. This is program induction over the
(smallest corner of the) computational universe: each rule number R in [0, 256)
is an 8-row truth table mapping a 3-bit neighborhood to an output bit, and we
ask which little program the market has been running lately.

Mapping (market -> CA symbols)
------------------------------
1. ENCODE  — each bar's return sign becomes a bit: up (close[j+1] > close[j])
   = 1, down-or-flat = 0. bit[j] is the direction of bar j+1 and is first
   known at the close of bar j+1.
2. FIT     — read the bit stream *temporally*: the neighborhood for target
   bit[j] is (bit[j-3], bit[j-2], bit[j-1]), Wolfram-coded as
   4*b[j-3] + 2*b[j-2] + b[j-1]. Over a trailing window of `window`
   transitions, score ALL 256 rules by how many observed next-bit transitions
   each reproduces (a single matrix product against the rolling
   (neighborhood, outcome) count table — no per-rule loop).
3. PREDICT — apply the best-fitting rule to the latest known 3-bit
   neighborhood to predict the NEXT bit, i.e. the next bar's direction.
4. TRADE   — position sign = predicted direction; magnitude = the best rule's
   in-window edge over chance, 2*(hit_rate - 0.5) clipped to [0, 1] (hit-rate
   1.0 -> full size). If no rule beats chance (hit_rate <= 0.5), stay flat.
   Note every rule's bitwise complement scores 1 - hit_rate, so the best rule
   always has hit_rate >= 0.5; the flat case is exactly hit_rate == 0.5.

Overfit guards
--------------
- `min_trans`: below this many in-window transitions the fit is vacuous
  (256 programs vs a handful of examples) -> flat.
- Unobserved-neighborhood gate (best-rule strategy): if the current
  neighborhood never occurred inside the fit window, the best rule's output
  row for it is unconstrained by data (a free bit chosen by tie-break), so we
  refuse to trade on it. The ensemble strategy gets this for free: rules
  identical on observed rows but differing on the current unobserved row have
  identical scores and opposite predictions, so their votes cancel *exactly*
  and the ensemble abstains.
- Ties in the argmax are broken toward the LOWEST rule number — a fixed,
  deterministic choice (the "simplest program index" in Wolfram numbering).

Causality
---------
sig[t] uses only bits with target index <= t-1 for the fit (bit[t-1] compares
close[t] with close[t-1], known at the close of bar t) and the neighborhood
(bit[t-3], bit[t-2], bit[t-1]) for the prediction — also fully known at bar t.
The predicted bit is bit[t] = the direction of bar t+1. `_shift1` then makes
weights()[t+1] = sig[t]: the position held over bar t+1 was decided at the
close of bar t. No bar's fit or prediction ever touches its own target.

Strategies
----------
- eca_rulefit          : hard argmax. Trade the single best rule's prediction,
  sized by its in-window edge over chance.
- eca_rulefit_ensemble : soft rule-space vote. Every rule with positive edge
  votes its own prediction, weighted by that edge; position = normalized net
  vote in [-1, 1]. Robust to ties/overfit; abstains exactly on neighborhoods
  never seen in-window.

numpy only; all 256 rules scored per bar via rolling count tables + matmul.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1

# rule truth tables: _RULE_BITS[R, c] = output bit of rule R for neighborhood
# code c (Wolfram convention: c = 4*left + 2*center + 1*right, output = bit c
# of the rule number).
_RULES = np.arange(256, dtype=np.int64)
_RULE_BITS = ((_RULES[:, None] >> np.arange(8)[None, :]) & 1).astype(np.float64)


def _return_bits(closes: np.ndarray) -> np.ndarray:
    """Up/down encoding: bit[j] = 1 iff close[j+1] > close[j] (flat counts as
    0). bit[j] is first known at bar j+1."""
    if closes.size < 2:
        return np.zeros(0, dtype=np.int64)
    diffs = np.diff(np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0))
    return (diffs > 0.0).astype(np.int64)


def _rulefit_state(
    series: Series, window: int
) -> Tuple[int, Optional[Tuple[np.ndarray, ...]]]:
    """Shared per-bar rule-space fit.

    Returns (n, state) where state = (t_arr, hits, n_win, cur_code, obs):
      t_arr    : bar indices t with at least one fully-known transition
      hits     : (len(t_arr), 256) — in-window reproduced-transition count per
                 rule, fit on targets bit[j] with j <= t-1 only
      n_win    : (len(t_arr),) number of transitions in each fit window
      cur_code : (len(t_arr),) neighborhood code (b[t-3], b[t-2], b[t-1]) —
                 the input for predicting bit[t] = bar t+1's direction
      obs      : (len(t_arr),) in-window occurrence count of cur_code
    state is None when the series is too short for a single transition.
    """
    c = _closes(series)
    n = c.size
    b = _return_bits(c)
    m = b.size  # n - 1 bits
    if m < 4:  # need bits 0..3 for the first transition (target j = 3)
        return n, None

    # codes_full[i] = neighborhood code for target index j = i + 3
    codes_full = 4 * b[:-2] + 2 * b[1:-1] + b[2:]          # length m - 2
    T = m - 3                                              # transitions: j = 3..m-1
    pair = 2 * codes_full[:T] + b[3:3 + T]                 # flat (code, y) in [0, 16)
    onehot = np.zeros((T, 16))
    onehot[np.arange(T), pair] = 1.0
    C = np.concatenate([np.zeros((1, 16)), np.cumsum(onehot, axis=0)], axis=0)

    t_arr = np.arange(4, n)          # bar t knows transitions with target j <= t-1
    ie = t_arr - 3                   # count of transitions with target j < t
    is_ = np.maximum(0, ie - int(window))
    W = C[ie] - C[is_]               # (n_t, 16) rolling (code, y) counts
    n_win = (ie - is_).astype(np.float64)

    # score all 256 rules at once: hits[t, R] = sum_c cnt(c,0)*(1-B[R,c])
    #                                          + cnt(c,1)*B[R,c]
    hits = W[:, 0::2] @ (1.0 - _RULE_BITS.T) + W[:, 1::2] @ _RULE_BITS.T

    cur_code = codes_full[t_arr - 3]  # b[t-3..t-1], fully known at bar t
    idx = np.arange(t_arr.size)
    obs = W[idx, 2 * cur_code] + W[idx, 2 * cur_code + 1]
    return n, (t_arr, hits, n_win, cur_code, obs)


# --------------------------------------------------------------------------- #
# Strategies                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class EcaRuleFitStrategy:
    """Hard rule-space argmax: fit all 256 ECA rules on the trailing window of
    return-bit transitions, apply the single best-fitting rule to the latest
    3-bit neighborhood, and trade its predicted direction sized by the rule's
    in-window edge over chance. Flat when no rule beats a coin flip, when the
    window is too thin, or when the current neighborhood was never observed
    in-window (the rule's output there is a data-free tie-break bit)."""
    window: int = 16
    min_trans: int = 8
    name: str = "eca_rulefit"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(int(self.window), 1)
        self.min_trans = max(1, min(int(self.min_trans), self.window))
        self.name = f"eca_rulefit(w={self.window},min={self.min_trans})"
        self.params = {"window": self.window, "min_trans": self.min_trans}

    def weights(self, series: Series) -> np.ndarray:
        n, st = _rulefit_state(series, self.window)
        sig = np.zeros(n)
        if st is not None:
            t_arr, hits, n_win, cur_code, obs = st
            idx = np.arange(t_arr.size)
            best = np.argmax(hits, axis=1)              # ties -> lowest rule number
            p = hits[idx, best] / np.maximum(n_win, 1.0)
            direction = 2.0 * ((best >> cur_code) & 1) - 1.0
            edge = np.clip(2.0 * (p - 0.5), 0.0, 1.0)   # 0 at chance, 1 at perfect
            ok = (n_win >= self.min_trans) & (obs > 0.0)
            sig[t_arr] = np.where(ok, direction * edge, 0.0)
        sig = np.clip(np.nan_to_num(sig), -1.0, 1.0)
        return _shift1(sig)


@dataclass
class EcaRuleFitEnsembleStrategy:
    """Soft rule-space vote: every ECA rule with a positive in-window edge
    votes its own next-bit prediction, weighted by that edge; the position is
    the normalized net vote in [-1, 1]. Complement rules carry the negated
    edge (clipped to 0), so noise self-cancels; on a neighborhood never seen
    in-window, equal-scoring rules split their votes exactly and the ensemble
    abstains without needing an explicit gate."""
    window: int = 16
    min_trans: int = 8
    name: str = "eca_rulefit_ensemble"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.window = max(int(self.window), 1)
        self.min_trans = max(1, min(int(self.min_trans), self.window))
        self.name = f"eca_rulefit_ens(w={self.window},min={self.min_trans})"
        self.params = {"window": self.window, "min_trans": self.min_trans}

    def weights(self, series: Series) -> np.ndarray:
        n, st = _rulefit_state(series, self.window)
        sig = np.zeros(n)
        if st is not None:
            t_arr, hits, n_win, cur_code, _obs = st
            rates = hits / np.maximum(n_win, 1.0)[:, None]
            edges = np.clip(rates - 0.5, 0.0, None)                    # (n_t, 256)
            dirs = 2.0 * ((_RULES[None, :] >> cur_code[:, None]) & 1) - 1.0
            num = (dirs * edges).sum(axis=1)
            den = edges.sum(axis=1)
            vote = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
            sig[t_arr] = np.where(n_win >= self.min_trans, vote, 0.0)
        sig = np.clip(np.nan_to_num(sig), -1.0, 1.0)
        return _shift1(sig)


STRATEGIES = {
    "eca_rulefit": EcaRuleFitStrategy,
    "eca_rulefit_ensemble": EcaRuleFitEnsembleStrategy,
}
