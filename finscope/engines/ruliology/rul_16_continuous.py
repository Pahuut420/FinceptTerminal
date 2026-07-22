"""
rul_16 — Continuous cellular automata / Lenia growth fields on raw returns.

Ruliology brief
---------------
NKS ch.4 generalizes CA from discrete symbols to CONTINUOUS cell values;
Lenia (Chan 2019) pushes further: cells hold reals, the neighborhood is read
through a smooth ring-shaped ("shell") kernel, and the update is a bell-shaped
GROWTH function G applied to the convolution potential U = K * A,

    A  <-  clip(A + dt * G(U)),   G bell-shaped around a growth niche mu.

Growth is positive only when the local potential sits inside the niche and
negative (or zero) outside it — that selectivity is what lets Lenia produce
smooth self-organizing solitons at the edge of chaos instead of the wash-out
of a linear filter.

Trading translation
-------------------
ENCODE — the recent return vector IS the continuous field. Returns are
volatility-standardized (trailing sigma only) and squashed with tanh so cell
values live in (-1, 1) across regimes.

Because returns are signed, the scalar Lenia bell is mirrored into a
DUAL-NICHE, antisymmetric growth function

    G(u) = exp(-(u - mu)^2 / 2 s^2) - exp(-(u + mu)^2 / 2 s^2)

which has a long niche at +mu and a short niche at -mu, is ~0 for a flat tape
(u ~ 0), and DECAYS BACK TO 0 for extreme |u| (a blow-off exits the niche and
stops earning growth). This is the nonlinear part: unlike an EMA, response is
non-monotone in the smoothed return — moderate, persistent drift is rewarded;
noise and overextension are not. |G| < 1 by construction.

Two readouts, one per strategy:

* lenia_growth — single Lenia step AT the newest cell. The newest cell's
  neighborhood is the trailing lags 1..R read through a bell shell kernel
  (peaked mid-lag, vanishing at self and at the rim, Lenia-style). The signal
  is the growth increment G(U_t) itself: positive growth => build long,
  negative growth => short.

* lenia_field — the full self-organizing smoother. A persistent 1-D field of
  the last `field_len` cells is advected one cell per bar (newest return
  injected at index 0), then evolved one Lenia step with a symmetric
  zero-center shell kernel: A <- clip(A + dt * G(K * A), -1, 1). Old cells
  keep evolving under the growth dynamics, so coherent runs reinforce into
  soliton-like persistent structure while incoherent noise decays toward 0.
  The signal is the newest cell after the update — already in [-1, 1].

Causality: every per-bar value uses closes up to and including that bar only
(trailing sigma; field state at bar t is built from returns <= t), and the
final vector goes through `_shift1`, so weights[t] is decided at the close of
bar t-1. Safe from 3 bars: shorter inputs return zeros of the right length.

numpy + finscope only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1

_STD_FLOOR = 1e-8


def _returns(c: np.ndarray) -> np.ndarray:
    """Simple per-bar returns, r[0] = 0, safe against zero closes."""
    r = np.zeros_like(c)
    if c.size > 1:
        prev = c[:-1]
        safe = np.where(np.abs(prev) > 0.0, prev, 1.0)
        r[1:] = np.where(np.abs(prev) > 0.0, (c[1:] - prev) / safe, 0.0)
    return np.nan_to_num(r)


def _causal_std(r: np.ndarray, lookback: int) -> np.ndarray:
    """Trailing rolling std of r (window ends AT each bar — causal),
    expanding while fewer than `lookback` bars exist, floored > 0."""
    out = np.empty_like(r)
    for i in range(r.size):
        w = r[max(0, i - lookback + 1): i + 1]
        s = float(np.std(w)) if w.size >= 2 else 0.0
        out[i] = s if s > _STD_FLOOR else _STD_FLOOR
    return out


def _std_field(c: np.ndarray, vol_lookback: int) -> np.ndarray:
    """Continuous CA field values: vol-standardized, tanh-squashed returns."""
    r = _returns(c)
    return np.tanh(r / _causal_std(r, vol_lookback))


def _shell(x: np.ndarray, mu: float = 0.5, sigma: float = 0.15) -> np.ndarray:
    """Lenia kernel shell: bell over normalized radius x in (0, 1]."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _ring_kernel(radius: int) -> np.ndarray:
    """1-D causal ring kernel over lags 1..radius, unit mass. Peaked mid-lag,
    ~0 at self and at the rim — the 1-D analogue of Lenia's ring kernel."""
    radius = max(1, int(radius))
    x = np.arange(1, radius + 1, dtype=float) / float(radius)
    k = _shell(x)
    s = k.sum()
    return k / s if s > 0 else k


def _sym_kernel(radius: int) -> np.ndarray:
    """Symmetric zero-center shell kernel over offsets -R..R, unit mass.
    Used to evolve the whole field (np.convolve mode='same')."""
    radius = max(1, int(radius))
    x = np.arange(1, radius + 1, dtype=float) / float(radius)
    half = _shell(x)
    k = np.concatenate([half[::-1], [0.0], half])
    s = k.sum()
    return k / s if s > 0 else k


def _dual_growth(u: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Antisymmetric dual-niche Lenia growth: long niche at +mu, short niche
    at -mu, ~0 when flat, decays to 0 for extreme |u|. |G| < 1 always."""
    up = np.exp(-0.5 * ((u - mu) / sigma) ** 2)
    dn = np.exp(-0.5 * ((u + mu) / sigma) ** 2)
    return up - dn


@dataclass
class LeniaGrowthStrategy:
    """Lenia growth increment at the newest cell of the return field.

    U_t = ring-kernel convolution of the trailing standardized returns;
    signal = G(U_t) with the dual-niche bell growth. Long when the local
    potential sits in the +mu niche, short in the -mu niche, flat when the
    tape is flat or blown out past the niche."""
    radius: int = 6
    growth_mu: float = 0.55
    growth_sigma: float = 0.40
    vol_lookback: int = 12
    name: str = "lenia_growth"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"lenia_growth(r={self.radius},mu={self.growth_mu})"
        self.params = {"radius": self.radius, "growth_mu": self.growth_mu,
                       "growth_sigma": self.growth_sigma,
                       "vol_lookback": self.vol_lookback}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n < 3:
            return np.zeros(n)
        a = _std_field(c, self.vol_lookback)
        k = _ring_kernel(self.radius)
        sig = np.zeros(n)
        for i in range(1, n):
            lags = min(int(self.radius), i)   # lags 1..lags available at bar i
            kk = k[:lags]
            km = kk.sum()
            if km <= 0:
                continue
            # neighborhood of the newest cell = trailing lags (a[i-1] .. a[i-lags])
            u = float(np.dot(kk / km, a[i - 1::-1][:lags]))
            sig[i] = float(_dual_growth(np.array([u]),
                                        self.growth_mu, self.growth_sigma)[0])
        w = _shift1(np.clip(sig, -1.0, 1.0))
        return np.clip(np.nan_to_num(w), -1.0, 1.0)


@dataclass
class LeniaFieldStrategy:
    """Persistent self-organizing Lenia field over the recent return vector.

    Each bar: advect (cells age one slot, newest standardized return injected
    at index 0), then one Lenia step A <- clip(A + dt*G(K*A), -1, 1) over the
    whole field. Coherent runs reinforce into soliton-like structure; noise
    decays. Signal = newest cell after the update."""
    field_len: int = 12
    radius: int = 4
    dt: float = 0.35
    growth_mu: float = 0.50
    growth_sigma: float = 0.35
    vol_lookback: int = 12
    name: str = "lenia_field"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"lenia_field(L={self.field_len},r={self.radius},dt={self.dt})"
        self.params = {"field_len": self.field_len, "radius": self.radius,
                       "dt": self.dt, "growth_mu": self.growth_mu,
                       "growth_sigma": self.growth_sigma,
                       "vol_lookback": self.vol_lookback}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        if n < 3:
            return np.zeros(n)
        z = _std_field(c, self.vol_lookback)
        k = _sym_kernel(self.radius)
        length = max(3, int(self.field_len))
        a = np.zeros(length)
        sig = np.zeros(n)
        for i in range(n):
            # advect: cells age one slot, inject the newest return at index 0
            a[1:] = a[:-1]
            a[0] = z[i]
            # one Lenia update step over the whole field
            u = np.convolve(a, k, mode="same")
            a = np.clip(a + self.dt * _dual_growth(u, self.growth_mu,
                                                   self.growth_sigma),
                        -1.0, 1.0)
            sig[i] = a[0]
        w = _shift1(np.clip(sig, -1.0, 1.0))
        return np.clip(np.nan_to_num(w), -1.0, 1.0)


STRATEGIES = {
    "lenia_growth": LeniaGrowthStrategy,
    "lenia_field": LeniaFieldStrategy,
}
