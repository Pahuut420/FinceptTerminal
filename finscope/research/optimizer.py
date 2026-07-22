"""
Deterministic strategy optimiser — the AutoResearch mutator, no LLM.

Given the strategy library and a set of symbols, search each strategy's parameter
space and rank candidates by a robust, out-of-sample-aware score computed by the
$0 native backtester. Two searches:

  * grid_search  — exhaustive over a param grid (fully deterministic).
  * genetic_search — seeded mutation/selection for larger spaces (deterministic
    given the seed; no randomness that would break reproducibility).

Score (higher = better), aggregated across symbols:
    score = mean_oos_sharpe - turnover_penalty*mean_turnover
with a tie-break preferring lower mean max-drawdown. OOS-aware so it does not
reward in-sample overfitting; small samples are flagged, not hidden.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from finscope.core.datahub import DataHub
from finscope.core.contracts import Series
from finscope.engines import get_engine
from finscope.engines import strategies as strat_lib

# default parameter grids per strategy
DEFAULT_GRIDS: Dict[str, Dict[str, Sequence]] = {
    "momentum": {"fast": [2, 3, 5], "slow": [8, 10, 15, 20]},
    "mean_reversion": {"window": [3, 5, 7, 10], "z_cap": [1.5, 2.0, 2.5]},
    "vol_target_momentum": {"fast": [3, 5], "slow": [8, 15], "target_vol": [0.4, 0.6, 0.8]},
    "breakout": {"window": [3, 5, 7, 10]},
    # Fable-authored alpha strategies (finscope/engines/strategies_alpha.py)
    "regime_adaptive": {"er_window": [5, 8, 12], "z_cap": [1.5, 2.0, 2.5]},
    "skew_tilt_tsmom": {"lookback": [4, 6, 10], "target_vol": [0.4, 0.6]},
    "kama_trend": {"er_window": [5, 8], "fast": [2], "slow": [10, 15], "filter_mult": [0.5, 1.0]},
    # Fable alpha batch 2 (finscope/engines/strategies_alpha2.py)
    "ou_half_life": {"window": [7, 10, 14], "z_cap": [1.5, 2.0]},
    "connors_rsi2": {"rsi_period": [2, 3], "lower": [5.0, 10.0], "trend_window": [10, 20]},
    "macd_histogram": {"fast": [6, 12], "slow": [13, 26], "signal": [5]},
    "squeeze_breakout": {"window": [8, 10], "squeeze_frac": [0.6, 0.75]},
    # Fable alpha batch 3 — information-theoretic / multiscale / spectral (default params)
    "perm_entropy_trend": {}, "dfa_hurst": {}, "spectral_cycle": {},
}


@dataclass
class Candidate:
    strategy: str
    params: dict
    score: float
    mean_oos_sharpe: float
    mean_sharpe: float
    mean_return: float
    mean_max_drawdown: float
    mean_turnover: float
    per_symbol: Dict[str, float] = field(default_factory=dict)
    n_bars: int = 0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "params": self.params,
            "score": round(self.score, 4), "mean_oos_sharpe": round(self.mean_oos_sharpe, 4),
            "mean_sharpe": round(self.mean_sharpe, 4), "mean_return": round(self.mean_return, 4),
            "mean_max_drawdown": round(self.mean_max_drawdown, 4),
            "mean_turnover": round(self.mean_turnover, 4),
            "per_symbol": {k: round(v, 4) for k, v in self.per_symbol.items()},
            "n_bars": self.n_bars,
        }


class StrategyOptimizer:
    def __init__(self, symbols: Optional[List[str]] = None, days: int = 365,
                 engine: str = "native", cost_bps: float = 5.0,
                 turnover_penalty: float = 0.05, hub: Optional[DataHub] = None):
        self.hub = hub or DataHub()
        self.symbols = symbols or ["BTC", "ETH", "SOL"]
        self.days = days
        self.engine = get_engine(engine)
        self.cost_bps = cost_bps
        self.turnover_penalty = turnover_penalty
        self._series: Dict[str, Series] = {}

    def _load(self) -> None:
        for s in self.symbols:
            if s not in self._series:
                ser = self.hub.history(s, days=self.days)
                if ser and len(ser) >= 3:
                    self._series[s] = ser

    def evaluate(self, strategy_name: str, params: dict) -> Optional[Candidate]:
        self._load()
        if not self._series:
            return None
        try:
            oos, shp, ret, mdd, tno, per = [], [], [], [], [], {}
            nbars = 0
            for sym, ser in self._series.items():
                strat = strat_lib.build(strategy_name, **params) if strategy_name in strat_lib.STRATEGIES \
                    else _build_any(strategy_name, params)
                r = self.engine.backtest(strat, ser, cost_bps=self.cost_bps)
                oos.append(r.oos_sharpe); shp.append(r.sharpe); ret.append(r.total_return)
                mdd.append(r.max_drawdown); tno.append(r.turnover); per[sym] = r.oos_sharpe
                nbars = max(nbars, r.n_bars)
            m = lambda x: sum(x) / len(x)
            score = m(oos) - self.turnover_penalty * m(tno)
            return Candidate(strategy=strategy_name, params=params, score=score,
                             mean_oos_sharpe=m(oos), mean_sharpe=m(shp), mean_return=m(ret),
                             mean_max_drawdown=m(mdd), mean_turnover=m(tno),
                             per_symbol=per, n_bars=nbars)
        except Exception:
            return None


def _build_any(name: str, params: dict):
    """Build from the core library or the Fable-authored alpha library."""
    if name in strat_lib.STRATEGIES:
        return strat_lib.build(name, **params)
    for mod, attr in (("strategies_alpha", "ALPHA_STRATEGIES"),
                      ("strategies_alpha2", "ALPHA2_STRATEGIES"),
                      ("strategies_alpha3", "ALPHA3_STRATEGIES")):
        try:
            m = __import__(f"finscope.engines.{mod}", fromlist=[attr])
            reg = getattr(m, attr, {})
            if name in reg:
                return reg[name](**params)
        except Exception:
            continue
    raise KeyError(f"unknown strategy {name!r}")


def _param_combos(grid: Dict[str, Sequence]) -> List[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def grid_search(strategy_name: str, grid: Optional[Dict[str, Sequence]] = None,
                opt: Optional[StrategyOptimizer] = None) -> List[Candidate]:
    opt = opt or StrategyOptimizer()
    grid = grid if grid is not None else DEFAULT_GRIDS.get(strategy_name, {})
    cands = [opt.evaluate(strategy_name, p) for p in _param_combos(grid)]
    cands = [c for c in cands if c is not None]
    cands.sort(key=lambda c: (c.score, -c.mean_max_drawdown), reverse=True)
    return cands


def genetic_search(strategy_name: str, grid: Dict[str, Sequence],
                   generations: int = 6, pop: int = 8, seed: int = 7,
                   opt: Optional[StrategyOptimizer] = None) -> List[Candidate]:
    """Deterministic genetic search (LCG PRNG seeded) for larger param spaces."""
    opt = opt or StrategyOptimizer()
    keys = list(grid)
    state = seed & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    def sample() -> dict:
        return {k: grid[k][int(rnd() * len(grid[k])) % len(grid[k])] for k in keys}

    population = [sample() for _ in range(pop)]
    seen: Dict[Tuple, Candidate] = {}

    def evalp(p: dict) -> Optional[Candidate]:
        key = tuple(sorted(p.items()))
        if key not in seen:
            c = opt.evaluate(strategy_name, p)
            if c:
                seen[key] = c
        return seen.get(key)

    for _ in range(generations):
        scored = [c for c in (evalp(p) for p in population) if c]
        scored.sort(key=lambda c: c.score, reverse=True)
        elite = scored[: max(2, pop // 2)]
        population = [e.params for e in elite]
        while len(population) < pop:  # mutate an elite gene
            base = dict(elite[int(rnd() * len(elite)) % len(elite)].params)
            k = keys[int(rnd() * len(keys)) % len(keys)]
            base[k] = grid[k][int(rnd() * len(grid[k])) % len(grid[k])]
            population.append(base)
    out = list(seen.values())
    out.sort(key=lambda c: (c.score, -c.mean_max_drawdown), reverse=True)
    return out


def optimize_all(symbols: Optional[List[str]] = None, days: int = 365,
                 top_k: int = 3) -> dict:
    """Search every core strategy and return the global ranking + per-strategy best."""
    opt = StrategyOptimizer(symbols=symbols, days=days)
    all_cands: List[Candidate] = []
    best_per: Dict[str, dict] = {}
    for name in DEFAULT_GRIDS:
        cands = grid_search(name, opt=opt)
        if cands:
            best_per[name] = cands[0].to_dict()
            all_cands.extend(cands)
    all_cands.sort(key=lambda c: c.score, reverse=True)
    return {
        "symbols": opt.symbols, "days": days,
        "evaluated": len(all_cands),
        "global_top": [c.to_dict() for c in all_cands[:top_k]],
        "best_per_strategy": best_per,
        "note": ("small-sample: scores are illustrative until the DuckDB lake is "
                 "filled with more history" if all_cands and all_cands[0].n_bars < 60 else ""),
    }
