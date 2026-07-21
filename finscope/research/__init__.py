"""
FinScope research — deterministic strategy optimisation (AutoResearch pattern).

No LLM in the loop: the search over strategy parameters is a deterministic grid +
seeded genetic search, and every candidate is scored by the $0 numpy backtester.
The metric maximised is out-of-sample Sharpe aggregated across symbols. This is
the Karpathy AutoResearch loop with a deterministic mutator — pure code, so it
runs for free and is fully reproducible.
"""
from finscope.research.optimizer import (
    StrategyOptimizer, grid_search, genetic_search, optimize_all,
)

__all__ = ["StrategyOptimizer", "grid_search", "genetic_search", "optimize_all"]
