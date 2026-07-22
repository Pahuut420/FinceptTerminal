"""
rul_13_multiway — Multiway-system branching as uncertainty sizing (Ruliology #13).

Concept
-------
In Wolfram's multiway systems every applicable rewrite rule is applied *at
once*: a state does not evolve down one path, it branches into all of its
possible successors, growing a multiway graph. Two observables fall out:

- BRANCH COUNT   — how many distinct successor states exist. Few branches =
  the possible futures converge (low nondeterminism); many branches = the
  futures diverge (high nondeterminism).
- BRANCHIAL SPREAD — how far apart those branches sit in "branchial space".
  Here we measure it as the dispersion of the branches' implied price moves:
  branches can be numerous yet all agree (small spread) or few yet violently
  disagree (large spread).

Mapping (market -> multiway system)
-----------------------------------
1. ENCODE  — each bar's return becomes a ternary symbol U/F/D (+1/0/-1) using
   a *causal* expanding-std deadband: r > eps -> +1, r < -eps -> -1, else 0,
   with eps[j] = eps_frac * std(returns[0..j]). Each symbol depends only on
   data up to its own bar, so the encoding never repaints.
2. RULES   — the rewrite ruleset is mined causally from the symbol stream
   itself: every observed m-gram transition (context of length m = 1..max_m
   -> next symbol) is a rewrite rule, with its observation count as rule
   multiplicity. At bar t only transitions completed by bar t are known.
3. BRANCH  — take the current k-symbol context and apply ALL applicable rules
   simultaneously (true multiway step): the state branches into every next
   symbol supported by at least one matching rule, weighted by rule counts
   pooled across context lengths. Repeat to `depth`, growing the multiway
   cone of near futures. A context with no matching rule branches uniformly
   into all three symbols — maximal nondeterminism, exactly as it should.
4. TRADE   — direction = the branch-weighted mean implied move of the leaves
   (the consensus of the multiway cone, dominated by the heaviest branch).
   Size = confidence from the cone's geometry:
     - multiway_branch    : 1 - (B - 1)/(3^depth - 1), B = leaf branch count.
       One surviving branch -> full conviction; a fully branched cone -> flat.
     - multiway_branchial : 1 / (1 + lam * spread), spread = branch-weighted
       std of leaf implied moves (branchial distance between futures).

Causality
---------
Symbols, rules, and the multiway expansion at bar t use closes[0..t] only
(rules are ingested incrementally, never from the future), and the final
signal is passed through `_shift1`, so weights()[t] depends exclusively on
information available at the close of bar t-1. numpy only.

Strategies
----------
- multiway_branch    : consensus direction sized by branch-count convergence.
- multiway_branchial : consensus direction sized by inverse branchial spread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1

# rule multiplicity table: (context_len, context_tuple) -> counts over next
# symbol, indexed [D, F, U] = [-1, 0, +1]
_Rules = Dict[Tuple[int, Tuple[int, ...]], np.ndarray]


# --------------------------------------------------------------------------- #
# Encoding                                                                    #
# --------------------------------------------------------------------------- #

def _ternary_symbols(closes: np.ndarray, eps_frac: float) -> np.ndarray:
    """Causal U/F/D encoding of returns. symbol[j] (known at bar j+1) is
    +1 / 0 / -1 by comparing return[j] against an expanding-std deadband
    computed from returns[0..j] only — no repainting."""
    c = np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0)
    if c.size < 2:
        return np.zeros(0, dtype=np.int64)
    prev = np.where(np.abs(c[:-1]) > 1e-12, c[:-1], 1.0)
    r = np.diff(c) / prev
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    # expanding (causal) std via cumulative moments
    idx = np.arange(1, r.size + 1, dtype=float)
    mean = np.cumsum(r) / idx
    var = np.maximum(np.cumsum(r * r) / idx - mean * mean, 0.0)
    eps = np.maximum(eps_frac * np.sqrt(var), 1e-12)

    sym = np.zeros(r.size, dtype=np.int64)
    sym[r > eps] = 1
    sym[r < -eps] = -1
    return sym


# --------------------------------------------------------------------------- #
# Multiway expansion                                                          #
# --------------------------------------------------------------------------- #

def _multiway_cone(suffix: Tuple[int, ...], rules: _Rules,
                   max_m: int, depth: int) -> Dict[Tuple[int, ...], float]:
    """Expand the multiway cone of near futures from `suffix`.

    Each step applies ALL applicable rewrite rules at once: the counts of
    every matching m-gram rule (m = 1..max_m) are pooled, and the state
    branches into each supported next symbol with count-proportional weight.
    No applicable rule -> uniform branch into all three symbols. Returns
    {leaf path (tuple of future symbols): branch weight}, weights sum to 1.
    """
    frontier: Dict[Tuple[int, ...], float] = {(): 1.0}
    for _ in range(max(depth, 1)):
        nxt: Dict[Tuple[int, ...], float] = {}
        for path, w in frontier.items():
            ctx = suffix + path
            counts = np.zeros(3)
            for m in range(1, max_m + 1):
                if len(ctx) >= m:
                    vec = rules.get((m, ctx[-m:]))
                    if vec is not None:
                        counts = counts + vec
            total = counts.sum()
            if total <= 0.0:
                counts = np.ones(3)
                total = 3.0
            for si in range(3):
                if counts[si] > 0.0:
                    p = path + (si - 1,)
                    nxt[p] = nxt.get(p, 0.0) + w * (counts[si] / total)
        frontier = nxt
    return frontier


def _cone_stats(frontier: Dict[Tuple[int, ...], float],
                depth: int) -> Tuple[int, float, float]:
    """(branch_count, mean_implied_move, branchial_spread), the last two
    normalized by depth so both live on a per-bar [-1, 1] / [0, 1]-ish scale."""
    if not frontier:
        return 1, 0.0, 0.0
    implied = np.array([float(sum(p)) for p in frontier], dtype=float)
    w = np.array(list(frontier.values()), dtype=float)
    tot = w.sum()
    w = w / tot if tot > 0 else np.full(w.size, 1.0 / w.size)
    mu = float((w * implied).sum())
    spread = float(np.sqrt(max((w * (implied - mu) ** 2).sum(), 0.0)))
    d = float(max(depth, 1))
    return len(frontier), mu / d, spread / d


def _multiway_signal(series: Series, context: int, max_m: int, depth: int,
                     eps_frac: float, mode: str, lam: float,
                     gain: float) -> np.ndarray:
    """Shared engine. sig[t] uses closes[0..t] only; caller shifts."""
    c = _closes(series)
    n = c.size
    sig = np.zeros(n)
    if n < 3:
        return sig

    syms = _ternary_symbols(c, eps_frac)   # syms[j] known at bar j+1
    rules: _Rules = {}
    max_b = 3 ** max(depth, 1)

    for i in range(1, n):
        j = i - 1                          # newest symbol index known at bar i
        # ingest the rewrite rules completed by symbol j (causal, incremental)
        for m in range(1, max_m + 1):
            if j >= m:
                key = (m, tuple(syms[j - m:j]))
                vec = rules.get(key)
                if vec is None:
                    vec = np.zeros(3)
                    rules[key] = vec
                vec[syms[j] + 1] += 1.0

        suffix = tuple(syms[max(0, i - context):i])
        if not suffix:
            continue
        frontier = _multiway_cone(suffix, rules, max_m, depth)
        n_branch, mu, spread = _cone_stats(frontier, depth)

        if mode == "count":
            conf = 1.0 - (n_branch - 1) / max(max_b - 1, 1)
        else:  # "branchial"
            conf = 1.0 / (1.0 + lam * spread)
        sig[i] = gain * mu * conf

    return np.clip(np.nan_to_num(sig), -1.0, 1.0)


# --------------------------------------------------------------------------- #
# Strategies                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class MultiwayBranchStrategy:
    """Multiway branch-count sizing: bet the multiway cone's consensus
    direction, scaled by how few branches survive. One branch (futures
    converge) -> full conviction; a fully branched cone (futures maximally
    nondeterministic) -> flat."""
    context: int = 4
    max_m: int = 3
    depth: int = 2
    eps_frac: float = 0.35
    gain: float = 2.0
    name: str = "multiway_branch"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.context = max(int(self.context), 1)
        self.max_m = max(min(int(self.max_m), self.context), 1)
        self.depth = max(int(self.depth), 1)
        self.eps_frac = float(max(self.eps_frac, 0.0))
        self.gain = float(max(self.gain, 0.0))
        self.name = f"multiway_branch(k={self.context},m={self.max_m},d={self.depth})"
        self.params = {"context": self.context, "max_m": self.max_m,
                       "depth": self.depth, "eps_frac": self.eps_frac,
                       "gain": self.gain}

    def weights(self, series: Series) -> np.ndarray:
        sig = _multiway_signal(series, self.context, self.max_m, self.depth,
                               self.eps_frac, "count", 0.0, self.gain)
        return _shift1(sig)


@dataclass
class MultiwayBranchialStrategy:
    """Branchial-spread sizing: same multiway cone, but confidence comes from
    how far apart the branches sit in branchial space — the branch-weighted
    dispersion of their implied moves. Many branches that all agree still
    trade near full size; few branches that violently disagree get shrunk."""
    context: int = 4
    max_m: int = 2
    depth: int = 3
    eps_frac: float = 0.35
    lam: float = 1.5
    gain: float = 2.0
    name: str = "multiway_branchial"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.context = max(int(self.context), 1)
        self.max_m = max(min(int(self.max_m), self.context), 1)
        self.depth = max(int(self.depth), 1)
        self.eps_frac = float(max(self.eps_frac, 0.0))
        self.lam = float(max(self.lam, 0.0))
        self.gain = float(max(self.gain, 0.0))
        self.name = f"multiway_branchial(k={self.context},m={self.max_m},d={self.depth})"
        self.params = {"context": self.context, "max_m": self.max_m,
                       "depth": self.depth, "eps_frac": self.eps_frac,
                       "lam": self.lam, "gain": self.gain}

    def weights(self, series: Series) -> np.ndarray:
        sig = _multiway_signal(series, self.context, self.max_m, self.depth,
                               self.eps_frac, "branchial", self.lam, self.gain)
        return _shift1(sig)


STRATEGIES = {
    "multiway_branch": MultiwayBranchStrategy,
    "multiway_branchial": MultiwayBranchialStrategy,
}
