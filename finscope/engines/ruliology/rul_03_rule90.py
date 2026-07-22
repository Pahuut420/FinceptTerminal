"""
Ruliology strategy 03 — Rule 90 (Sierpinski / additive-XOR) signals.

Rule 90 maps a binary row to its successor by XOR-ing each cell's two
neighbours: ``c'_i = c_{i-1} XOR c_{i+1}``.  Seeded from a single cell it
draws an exactly self-similar Sierpinski triangle, and — because XOR is
additive over GF(2) — its evolution is scale-invariant: after 2^k steps a
cell equals the XOR of the cells 2^k to its left and right, so the pattern
at dyadic scale 2^k is a nested copy of the pattern at scale 1.

We port both properties to a return-sign bit stream (one bit per bar,
1 = up, 0 = down/flat — the "CA row" encoding):

* ``Rule90FractalTrendStrategy`` — measures whether the sign pattern is
  *nested across dyadic scales* (drift of 1-, 2-, 4-bar aggregated
  returns).  When the per-scale drifts agree, the trend is fractal and
  persistent -> follow it; when the scales disagree the self-similarity is
  broken -> fade the apparent drift.

* ``Rule90XorEchoStrategy`` — exploits Rule 90's additivity directly: the
  causal XOR of two recent sign bits is the additive one-step forecast of
  the next bit.  A rolling hit-rate turns that forecast into a signed edge
  in [-1, 1]; the position is the XOR forecast weighted by its measured
  edge (an anti-predictive bit stream flips the sign automatically).

Both are causal: the signal at bar t is computed from bars <= t only and
then shifted one bar forward via ``_shift1``, matching the library-wide
"decide at close of t-1, hold over bar t" convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1

_EPS = 1e-12


def _returns(series: Series) -> np.ndarray:
    """Per-bar simple returns, length == len(series), with returns[0] == 0."""
    c = _closes(series)
    if c.size < 2:
        return np.zeros(c.size)
    prev = c[:-1]
    ok = np.abs(prev) > _EPS
    r = np.where(ok, (c[1:] - prev) / np.where(ok, prev, 1.0), 0.0)
    return np.nan_to_num(np.concatenate(([0.0], r)))


def _sign_bits(rets: np.ndarray) -> np.ndarray:
    """Return-sign CA row: 1 = up bar, 0 = down/flat bar."""
    return (rets > 0.0).astype(np.int8)


@dataclass
class Rule90FractalTrendStrategy:
    """Follow the drift while the sign pattern is self-similar across
    dyadic scales (Rule 90 / Sierpinski nesting); fade it when broken.

    At each bar t the last ``window`` returns are aggregated at scales
    1, 2, ..., 2^(n_scales-1); each scale's drift is the mean sign of its
    block returns (in [-1, 1]).  Nesting = 1 - (spread of drifts) / 2 maps
    perfect cross-scale agreement to 1 and full disagreement to 0.  The
    signal ``mean_drift * (2 * nesting - 1)`` therefore follows the trend
    when the structure is nested and fades it when self-similarity breaks.
    """

    window: int = 16
    n_scales: int = 3
    name: str = "rule90_fractal_trend"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"rule90_fractal({self.window},{self.n_scales})"
        self.params = {"window": self.window, "n_scales": self.n_scales}

    def weights(self, series: Series) -> np.ndarray:
        rets = _returns(series)
        n = rets.size
        sig = np.zeros(n)
        if n < 3:
            return sig
        scales = [2 ** k for k in range(max(1, self.n_scales))]
        window = max(self.window, scales[-1])
        drifts = np.empty(len(scales))
        for t in range(window, n):
            for j, s in enumerate(scales):
                k = max(1, window // s)
                seg = rets[t - k * s + 1: t + 1]
                drifts[j] = float(np.mean(np.sign(seg.reshape(k, s).sum(axis=1))))
            dbar = float(drifts.mean())
            nesting = 1.0 - (float(drifts.max()) - float(drifts.min())) / 2.0
            sig[t] = dbar * (2.0 * nesting - 1.0)
        return _shift1(np.clip(np.nan_to_num(sig), -1.0, 1.0))


@dataclass
class Rule90XorEchoStrategy:
    """Trade the additive (Rule 90) XOR forecast of the next return-sign
    bit, scaled by its rolling out-of-sample hit-rate edge.

    Forecast for bit t is ``b[t - lag_a] XOR b[t - lag_b]`` — the Rule 90
    neighbour-XOR read causally along the time axis.  The rolling hit-rate
    over ``window`` past forecasts becomes an edge in [-1, 1]; the position
    is ``(2 * next_forecast - 1) * edge``, so a persistently *wrong* XOR
    structure is faded rather than followed.
    """

    lag_a: int = 1
    lag_b: int = 2
    window: int = 10
    name: str = "rule90_xor_echo"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.lag_a < self.lag_b:
            raise ValueError("require 0 < lag_a < lag_b")
        self.name = f"rule90_xor_echo({self.lag_a},{self.lag_b},w={self.window})"
        self.params = {
            "lag_a": self.lag_a,
            "lag_b": self.lag_b,
            "window": self.window,
        }

    def weights(self, series: Series) -> np.ndarray:
        rets = _returns(series)
        n = rets.size
        if n < 3:
            return np.zeros(n)
        b = _sign_bits(rets)
        la, lb, w = self.lag_a, self.lag_b, max(2, self.window)

        # In-sample XOR forecast of bit t from bits t-la and t-lb (t >= lb).
        p = np.zeros(n, dtype=np.int8)
        p[lb:] = b[lb - la: n - la] ^ b[: n - lb]
        hits = (p == b).astype(float)

        # Rolling hit-rate edge over the last w fully-valid forecasts
        # (windows start at t-w+1 >= lb+1, skipping the synthetic bit 0).
        edge = np.zeros(n)
        t0 = lb + w
        if t0 < n:
            csum = np.concatenate(([0.0], np.cumsum(hits)))
            roll = csum[w:] - csum[:-w]  # roll[j] = sum(hits[j : j + w])
            edge[t0:] = 2.0 * roll[lb + 1: n - w + 1] / w - 1.0

        # Causal forecast of the *next* bit, made at the close of bar t:
        # q[t] = b[t+1-la] XOR b[t+1-lb], defined for t >= lb - 1.
        q = np.zeros(n)
        if n >= lb:
            q[lb - 1:] = (b[lb - la: n - la + 1] ^ b[: n - lb + 1]).astype(float)

        sig = (2.0 * q - 1.0) * edge
        return _shift1(np.clip(np.nan_to_num(sig), -1.0, 1.0))


STRATEGIES = {
    "rule90_fractal_trend": Rule90FractalTrendStrategy,
    "rule90_xor_echo": Rule90XorEchoStrategy,
}
