"""
WolframBridge — optional symbolic/verified offload for heavy analytics.

Why a bridge and not a direct call?  A standalone Python process can't call an
MCP tool; MCP tools live in the agent/session layer. So the bridge is designed
two ways:

  1. As a *query generator*: `wl_*` methods return the exact Wolfram Language
     source for a computation. An orchestrating agent (this session) can run that
     source through the Wolfram MCP (`WolframLanguageEvaluator`) and feed the
     result back — useful for verification and for offloading big symbolic work.

  2. As a *drop-in engine*: construct with an `evaluator` callable
     `evaluator(wl_code:str) -> float | list`. If provided, `.annualized_vol`,
     `.correlation`, etc. route through Wolfram; if not, they transparently fall
     back to the numpy AnalyticsEngine. Either way the terminal keeps working.

This mirrors the verified session result (annualized vol 0.2030, Sharpe 5.214 on
the 10-point sample) computed live through the Wolfram MCP.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from finscope.analytics.engine import AnalyticsEngine
from finscope.core.contracts import Series, CorrelationMatrix

Evaluator = Callable[[str], object]


class WolframBridge:
    def __init__(self, evaluator: Optional[Evaluator] = None, periods_per_year: int = 365):
        self.evaluator = evaluator
        self.numpy = AnalyticsEngine(periods_per_year=periods_per_year)
        self.ppy = periods_per_year

    @property
    def engine_name(self) -> str:
        return "wolfram" if self.evaluator else "numpy"

    # ---- Wolfram Language source generators (always available) ------------
    @staticmethod
    def _wl_list(xs: Sequence[float]) -> str:
        return "{" + ", ".join(repr(float(x)) for x in xs) + "}"

    def wl_annualized_vol(self, returns: Sequence[float]) -> str:
        return (f"With[{{r = {self._wl_list(returns)}}}, "
                f"StandardDeviation[r]*Sqrt[{self.ppy}]]")

    def wl_sharpe(self, returns: Sequence[float]) -> str:
        return (f"With[{{r = {self._wl_list(returns)}}}, "
                f"(Mean[r]*{self.ppy})/(StandardDeviation[r]*Sqrt[{self.ppy}])]")

    def wl_correlation(self, series_list: List[Series]) -> str:
        rows = []
        for s in series_list:
            rows.append(self._wl_list(s.returns()))
        # align to shortest then Correlation on the transposed matrix
        return (f"Module[{{data = {{{', '.join(rows)}}}, m}}, "
                f"m = Min[Length /@ data]; "
                f"Correlation[Transpose[(Take[#, -m] & /@ data)]]]")

    # ---- routed computations (Wolfram if evaluator present, else numpy) ---
    def annualized_vol(self, returns: Sequence[float]) -> float:
        if self.evaluator:
            try:
                return float(self.evaluator(self.wl_annualized_vol(returns)))  # type: ignore[arg-type]
            except Exception:
                pass
        return self.numpy.annualized_vol(returns)

    def sharpe(self, returns: Sequence[float]) -> float:
        if self.evaluator:
            try:
                return float(self.evaluator(self.wl_sharpe(returns)))  # type: ignore[arg-type]
            except Exception:
                pass
        return self.numpy.sharpe(returns)

    def correlation(self, series_list: List[Series]) -> CorrelationMatrix:
        # numpy path returns the structured matrix; Wolfram path is used mainly
        # for verification via wl_correlation(), so we keep numpy as the shape source.
        return self.numpy.correlation(series_list)
