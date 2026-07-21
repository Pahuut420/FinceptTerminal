"""
Strategy library — each maps a price Series to a *causal* target-weight vector.

Convention (no lookahead): weights()[t] is the position to hold over bar t,
decided using information available at the close of bar t-1. Internally we
compute a signal and shift it forward by one bar so the backtester never peeks.

Weights are in [-1, 1]; +1 = full long, -1 = full short, 0 = flat.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finscope.core.contracts import Series


def _closes(series: Series) -> np.ndarray:
    return np.asarray(series.closes(), dtype=float)


def _shift1(sig: np.ndarray) -> np.ndarray:
    """Shift a signal forward one bar (decision at t-1 applies to bar t)."""
    out = np.zeros_like(sig)
    if sig.size > 1:
        out[1:] = sig[:-1]
    return np.nan_to_num(out)


def _sma(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x.copy()
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full_like(x, np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


@dataclass
class MomentumStrategy:
    """Long when fast SMA > slow SMA, short otherwise (trend following)."""
    fast: int = 3
    slow: int = 8
    name: str = "momentum"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"momentum({self.fast},{self.slow})"
        self.params = {"fast": self.fast, "slow": self.slow}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        sig = np.sign(np.nan_to_num(_sma(c, self.fast) - _sma(c, self.slow)))
        return _shift1(sig)


@dataclass
class MeanReversionStrategy:
    """Fade deviations from a moving average (contrarian), z-scored."""
    window: int = 5
    z_cap: float = 2.0
    name: str = "mean_reversion"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"mean_reversion({self.window})"
        self.params = {"window": self.window, "z_cap": self.z_cap}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        ma = _sma(c, self.window)
        dev = c - ma
        std = np.nanstd(dev[~np.isnan(dev)]) or 1.0
        z = np.clip(np.nan_to_num(dev) / std, -self.z_cap, self.z_cap) / self.z_cap
        return _shift1(-z)  # fade: short when above MA


@dataclass
class VolTargetMomentumStrategy:
    """Momentum sign scaled to a target annualised volatility."""
    fast: int = 3
    slow: int = 8
    target_vol: float = 0.60
    lookback: int = 7
    name: str = "vol_target_momentum"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"vol_target_mom({self.fast},{self.slow},tv={self.target_vol})"
        self.params = {"fast": self.fast, "slow": self.slow,
                       "target_vol": self.target_vol, "lookback": self.lookback}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        sign = np.sign(np.nan_to_num(_sma(c, self.fast) - _sma(c, self.slow)))
        rets = np.diff(c) / c[:-1] if c.size > 1 else np.array([0.0])
        rets = np.insert(rets, 0, 0.0)
        rv = np.array([
            np.std(rets[max(0, i - self.lookback):i]) * np.sqrt(365) if i > 2 else self.target_vol
            for i in range(c.size)
        ])
        scale = np.clip(self.target_vol / np.where(rv > 1e-9, rv, 1e-9), 0.0, 1.5)
        return _shift1(np.clip(sign * scale, -1.0, 1.0))


@dataclass
class BreakoutStrategy:
    """Donchian breakout: long on new N-bar high, short on new N-bar low."""
    window: int = 5
    name: str = "breakout"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"breakout({self.window})"
        self.params = {"window": self.window}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        sig = np.zeros_like(c)
        for i in range(self.window, c.size):
            hi = c[i - self.window:i].max()
            lo = c[i - self.window:i].min()
            sig[i] = 1.0 if c[i] >= hi else (-1.0 if c[i] <= lo else sig[i - 1])
        return _shift1(sig)


# registry of named strategy factories (for the agent surface + autoresearch loop)
STRATEGIES = {
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "vol_target_momentum": VolTargetMomentumStrategy,
    "breakout": BreakoutStrategy,
}


def build(name: str, **params):
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r} (have {sorted(STRATEGIES)})")
    return STRATEGIES[name](**params)
