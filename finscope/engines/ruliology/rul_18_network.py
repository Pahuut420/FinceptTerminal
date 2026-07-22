"""
rul_18_network — Network-automaton connectivity signal (Ruliology signal #18).

Concept
-------
In Wolfram-model physics, "space" is not a backdrop — it *is* a hypergraph that
rewrites itself by local rules, and geometry (curvature, dimension, coherence)
is an emergent property of its connectivity. We borrow the cheapest faithful
proxy: treat the recent market as a small evolving graph whose nodes are bars
and whose edges are pattern-recurrences, then read the graph's emergent
geometry — density and clustering — as a regime detector.

Mapping (market -> graph automaton)
-----------------------------------
1. ENCODE  — each recent bar i becomes a node carrying its local return-pattern
   p_i = (r[i-m+1], ..., r[i]), an m-dimensional delay embedding of returns.
   The rolling `window` of bars is the automaton's current spatial slice; each
   new bar is one rewrite step (add a node, drop the oldest, re-derive edges).
2. REWRITE — connect nodes i, j iff ||p_i - p_j||_2 <= eps, with eps scaled by
   the window's return volatility (recurrence-plot convention: an edge means
   two bars share structure *beyond the noise scale*, so the graph is not just
   re-measuring vol).
3. MEASURE — two emergent-geometry statistics of the adjacency matrix A:
   - density       = edges / possible edges          (connectivity)
   - transitivity  = trace(A^3) / open-triads        (global clustering:
     "regular geometry" = many closed triangles)
   A dense / highly-clustered graph means the same local pattern keeps
   recurring — a coherent regime. A fragmented graph (few edges, no triangles)
   means the rewrite produced dust — no geometry, nothing to trade.
4. TRADE   — sign comes from the *net directional flow across the graph*: the
   degree-weighted mean of node returns, sum_i(deg_i * r_i) / sum_i(deg_i).
   Well-connected nodes (patterns that recur) dominate the vote; isolated
   nodes (one-off outliers) are ignored. Position = coherence gate (ramp on
   density or transitivity) times tanh of the vol-normalized flow. Fragmented
   graph => gate ~ 0 => flat.

Causality
---------
The graph at bar t is built exclusively from returns r[1..t] (closes up to t).
Every per-bar signal is then passed through `_shift1`, so weights()[t] depends
only on information through bar t-1. eps and the flow normalization use the
same trailing window — no full-series statistics leak in.

Strategies
----------
- net_connectivity : gate = ramp(graph density). Coherent (dense) recurrence
  graph => trade the degree-weighted drift; sparse graph => flat.
- net_clustering   : gate = ramp(graph transitivity). Demands closed triangles
  (regular geometry), a stricter coherence test than raw density.

numpy only; graphs are tiny (k <= window nodes), adjacency algebra is O(k^3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# graph machinery                                                             #
# --------------------------------------------------------------------------- #

def _returns(c: np.ndarray) -> np.ndarray:
    """Simple returns aligned to bars; r[0] = 0 placeholder (never embedded)."""
    r = np.zeros_like(c)
    if c.size > 1:
        prev = c[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            r[1:] = np.where(np.abs(prev) > _EPS, (c[1:] - prev) / prev, 0.0)
    return np.nan_to_num(r)


def _recurrence_graph(
    r: np.ndarray, t: int, window: int, embed: int, eps_scale: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build the recurrence graph over bars (t-window+1 .. t) using data <= t.

    Returns (adjacency, node_returns, vol). Adjacency is empty (0 nodes) when
    too little history exists. Node i's pattern is r[i-embed+1 .. i]; nodes
    need i >= embed so patterns contain only real returns (r[0] is a
    placeholder).
    """
    lo = max(embed, t - window + 1)
    idx = np.arange(lo, t + 1)
    k = idx.size
    if k < 2:
        empty = np.zeros((0, 0))
        return empty, np.zeros(0), 0.0
    # patterns: (k, embed) delay-embedding matrix
    pat = np.stack([r[i - embed + 1: i + 1] for i in idx])
    node_ret = r[idx]
    vol = float(np.std(r[max(1, t - window + 1): t + 1]))
    eps = eps_scale * max(vol, _EPS) * np.sqrt(embed)
    # pairwise Euclidean distances -> vol-scaled recurrence adjacency
    diff = pat[:, None, :] - pat[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    adj = (dist <= eps).astype(float)
    np.fill_diagonal(adj, 0.0)
    return adj, node_ret, vol


def _density(adj: np.ndarray) -> float:
    k = adj.shape[0]
    if k < 2:
        return 0.0
    return float(adj.sum() / (k * (k - 1)))


def _transitivity(adj: np.ndarray) -> float:
    """Global clustering: closed triplets / all triplets (0 when no triads)."""
    if adj.shape[0] < 3:
        return 0.0
    a2 = adj @ adj
    triads = float(a2.sum() - np.trace(a2))  # open+closed paths of length 2
    if triads <= 0.0:
        return 0.0
    closed = float(np.trace(a2 @ adj))       # 6 * number of triangles
    return min(1.0, closed / triads)


def _flow(adj: np.ndarray, node_ret: np.ndarray) -> float:
    """Net directional flow: degree-weighted mean node return."""
    deg = adj.sum(axis=1)
    total = float(deg.sum())
    if total <= 0.0:
        return 0.0
    return float(np.dot(deg, node_ret) / total)


def _ramp(x: float, lo: float, hi: float) -> float:
    """Piecewise-linear gate: 0 below lo, 1 above hi."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _network_signal(
    c: np.ndarray,
    window: int,
    embed: int,
    eps_scale: float,
    gate_lo: float,
    gate_hi: float,
    flow_gain: float,
    min_nodes: int,
    use_clustering: bool,
) -> np.ndarray:
    """Per-bar raw signal (uses info up to and including bar t); caller shifts."""
    n = c.size
    sig = np.zeros(n)
    if n < 3:
        return sig
    r = _returns(c)
    for t in range(embed + min_nodes - 1, n):
        adj, node_ret, vol = _recurrence_graph(r, t, window, embed, eps_scale)
        if adj.shape[0] < min_nodes:
            continue
        stat = _transitivity(adj) if use_clustering else _density(adj)
        gate = _ramp(stat, gate_lo, gate_hi)
        if gate <= 0.0:
            continue
        drift = _flow(adj, node_ret)
        z = drift / max(vol, _EPS)
        sig[t] = gate * float(np.tanh(flow_gain * z))
    return np.clip(np.nan_to_num(sig), -1.0, 1.0)


# --------------------------------------------------------------------------- #
# strategies                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class NetworkConnectivityStrategy:
    """Trade the recurrence graph's degree-weighted drift, gated by density.

    High connectivity (the same local pattern keeps recurring across the
    window) => coherent regime => position along the graph's directional flow.
    Fragmented graph => flat.
    """
    window: int = 12
    embed: int = 3
    eps_scale: float = 1.0
    gate_lo: float = 0.15
    gate_hi: float = 0.60
    flow_gain: float = 2.0
    min_nodes: int = 4
    name: str = "net_connectivity"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"net_connectivity({self.window},{self.embed})"
        self.params = {
            "window": self.window, "embed": self.embed,
            "eps_scale": self.eps_scale, "gate_lo": self.gate_lo,
            "gate_hi": self.gate_hi, "flow_gain": self.flow_gain,
            "min_nodes": self.min_nodes,
        }

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        sig = _network_signal(
            c, self.window, self.embed, self.eps_scale,
            self.gate_lo, self.gate_hi, self.flow_gain,
            self.min_nodes, use_clustering=False,
        )
        return _shift1(sig)


@dataclass
class NetworkClusteringStrategy:
    """Trade the recurrence graph's drift, gated by global clustering.

    Transitivity demands closed triangles — mutually-recurrent bar triples —
    i.e. *regular emergent geometry*, a stricter coherence test than raw edge
    density. Only such a self-consistent graph earns a position.
    """
    window: int = 14
    embed: int = 3
    eps_scale: float = 1.0
    gate_lo: float = 0.20
    gate_hi: float = 0.70
    flow_gain: float = 2.0
    min_nodes: int = 5
    name: str = "net_clustering"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"net_clustering({self.window},{self.embed})"
        self.params = {
            "window": self.window, "embed": self.embed,
            "eps_scale": self.eps_scale, "gate_lo": self.gate_lo,
            "gate_hi": self.gate_hi, "flow_gain": self.flow_gain,
            "min_nodes": self.min_nodes,
        }

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        sig = _network_signal(
            c, self.window, self.embed, self.eps_scale,
            self.gate_lo, self.gate_hi, self.flow_gain,
            self.min_nodes, use_clustering=True,
        )
        return _shift1(sig)


STRATEGIES = {
    "net_connectivity": NetworkConnectivityStrategy,
    "net_clustering": NetworkClusteringStrategy,
}
