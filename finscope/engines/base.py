"""
Engine ABC + contracts + registry.

Contract summary:
  * A `Strategy` maps a price `Series` to a causal target-weight vector in [-1, 1]
    (weight[t] may only use information available up to bar t-1 — no lookahead).
  * An `Engine` runs a strategy against a Series and returns a `BacktestResult`.
  * `Engine.run(request: dict) -> dict` is the JSON surface an LLM agent drives.

Everything is JSON-serialisable so the dashboard agent can pass results around.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from finscope.core.contracts import Series


# ---- contracts ---------------------------------------------------------------
@dataclass(frozen=True)
class Signal:
    symbol: str
    action: str          # "long" | "short" | "flat"
    size: float          # target weight in [-1, 1]
    ts: float
    reason: str = ""
    engine: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    engine: str
    n_bars: int
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    turnover: float
    oos_sharpe: float          # the AutoResearch metric (out-of-sample)
    params: dict = field(default_factory=dict)
    equity_curve: List[float] = field(default_factory=list)
    verified: bool = True      # computed from real data, not a stub
    notes: str = ""

    def to_dict(self, with_curve: bool = False) -> dict:
        d = asdict(self)
        if not with_curve:
            d.pop("equity_curve", None)
        return d

    @property
    def score(self) -> float:
        """Single number the AutoResearch loop maximises."""
        return self.oos_sharpe


@runtime_checkable
class Strategy(Protocol):
    name: str
    params: dict
    def weights(self, series: Series) -> np.ndarray: ...


# ---- engine ABC --------------------------------------------------------------
class Engine:
    name: str = "base"
    kind: str = "backtest"     # "backtest" | "live" | "agent"
    description: str = ""

    def available(self) -> bool:
        """True if this engine can actually run here (deps/repo present)."""
        return True

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "available": self.available(),
                "description": self.description}

    def backtest(self, strategy: Strategy, series: Series, cost_bps: float = 5.0) -> BacktestResult:
        raise NotImplementedError

    def signal(self, strategy: Strategy, series: Series) -> Signal:
        """Latest target position from the strategy."""
        w = strategy.weights(series)
        last = float(w[-1]) if len(w) else 0.0
        action = "long" if last > 0.05 else ("short" if last < -0.05 else "flat")
        ts = series.candles[-1].ts if series.candles else 0.0
        return Signal(symbol=series.symbol, action=action, size=round(last, 4),
                      ts=ts, reason=getattr(strategy, "name", "strategy"), engine=self.name)

    # ---- JSON surface for agents ----
    def run(self, request: dict) -> dict:
        """Drive the engine with a JSON request. Overridden by adapters that need
        custom dispatch; the default handles {"op":"describe"}."""
        op = request.get("op", "describe")
        if op == "describe":
            return self.describe()
        return {"error": f"{self.name}: unsupported op {op!r}"}


# ---- registry ----------------------------------------------------------------
REGISTRY: Dict[str, Engine] = {}


def register_engine(cls):
    inst = cls()
    REGISTRY[inst.name] = inst
    return cls


def all_engines() -> List[Engine]:
    return list(REGISTRY.values())


def get_engine(name: str) -> Engine:
    if name not in REGISTRY:
        raise KeyError(f"unknown engine: {name} (have {sorted(REGISTRY)})")
    return REGISTRY[name]
