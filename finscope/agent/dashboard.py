"""
DashboardAgent — the agent-usable control surface for FinScope.

A single JSON-in / JSON-out object an LLM agent (or a human, or another engine)
can drive to: read the market, inspect providers/engines/strategies, run
backtests, get live signals, and rank strategies. `TOOL_SPEC` is an
OpenAI/Anthropic-style function schema so an agent can discover the ops.

Everything returns plain dicts/lists (JSON-serialisable). No terminal I/O here.
"""
from __future__ import annotations

import json
from typing import List, Optional

from finscope.core.datahub import DataHub
from finscope.core.contracts import AssetClass
from finscope.analytics.engine import AnalyticsEngine
from finscope.engines import all_engines, get_engine
from finscope.engines import strategies as strat_lib


class DashboardAgent:
    def __init__(self, hub: Optional[DataHub] = None):
        self.hub = hub or DataHub()
        self.analytics = AnalyticsEngine()

    # ---- capability discovery ----
    def describe(self) -> dict:
        return {
            "name": "finscope-dashboard",
            "providers": self.hub.provider_names(),
            "engines": [e.describe() for e in all_engines()],
            "strategies": sorted(strat_lib.STRATEGIES),
            "ops": [t["name"] for t in TOOL_SPEC],
            "watchlist": self.hub.watchlist,
        }

    # ---- market ----
    def market_snapshot(self, limit: int = 15) -> dict:
        quotes = self.hub.top(AssetClass.CRYPTO, limit=limit)
        return {"count": len(quotes),
                "quotes": [self._q(q) for q in quotes]}

    def quote(self, symbol: str) -> dict:
        q = self.hub.quote(symbol)
        return self._q(q) if q else {"error": f"no quote for {symbol}"}

    def history(self, symbol: str, days: int = 90) -> dict:
        s = self.hub.history(symbol, days=days)
        if not s or not len(s):
            return {"error": f"no history for {symbol}"}
        return {"symbol": s.symbol, "source": s.source, "n": len(s),
                "closes": s.closes(), "last": s.closes()[-1]}

    def risk(self, symbol: str, days: int = 180) -> dict:
        s = self.hub.history(symbol, days=days)
        if not s or len(s) < 3:
            return {"error": f"insufficient history for {symbol}"}
        return self.analytics.risk_metrics(s).__dict__

    # ---- engines / strategies ----
    def list_engines(self) -> dict:
        return {"engines": [e.describe() for e in all_engines()]}

    def list_strategies(self) -> dict:
        out = {}
        for name, cls in strat_lib.STRATEGIES.items():
            try:
                out[name] = cls().params
            except Exception:
                out[name] = {}
        return {"strategies": out}

    def backtest(self, engine: str = "native", strategy: str = "momentum",
                 symbol: str = "BTC", days: int = 365, params: Optional[dict] = None,
                 cost_bps: float = 5.0, with_curve: bool = False) -> dict:
        eng = get_engine(engine)
        if not eng.available():
            return {"error": f"engine {engine} not available here"}
        s = self.hub.history(symbol, days=days)
        if not s or len(s) < 3:
            return {"error": f"no history for {symbol}"}
        strat = strat_lib.build(strategy, **(params or {}))
        return eng.backtest(strat, s, cost_bps=cost_bps).to_dict(with_curve=with_curve)

    def signal(self, engine: str = "native", strategy: str = "momentum",
               symbol: str = "BTC", days: int = 90, params: Optional[dict] = None) -> dict:
        eng = get_engine(engine)
        s = self.hub.history(symbol, days=days)
        if not s or len(s) < 3:
            return {"error": f"no history for {symbol}"}
        return eng.signal(strat_lib.build(strategy, **(params or {})), s).to_dict()

    def rank_strategies(self, symbol: str = "BTC", days: int = 365,
                        engine: str = "native", cost_bps: float = 5.0) -> dict:
        eng = get_engine(engine)
        s = self.hub.history(symbol, days=days)
        if not s or len(s) < 3:
            return {"error": f"no history for {symbol}"}
        rows = []
        for name in strat_lib.STRATEGIES:
            try:
                r = eng.backtest(strat_lib.build(name), s, cost_bps=cost_bps)
                rows.append({"strategy": r.strategy, "oos_sharpe": r.oos_sharpe,
                             "sharpe": r.sharpe, "total_return": r.total_return,
                             "max_drawdown": r.max_drawdown, "n_trades": r.n_trades})
            except Exception as e:
                rows.append({"strategy": name, "error": str(e)})
        rows.sort(key=lambda x: x.get("oos_sharpe", -99), reverse=True)
        return {"symbol": symbol, "engine": engine, "ranked": rows}

    # ---- single dispatch entrypoint (what an agent calls) ----
    def run(self, request) -> dict:
        if isinstance(request, str):
            request = json.loads(request)
        op = request.get("op", "describe")
        args = {k: v for k, v in request.items() if k != "op"}
        fn = getattr(self, op, None)
        if not callable(fn) or op.startswith("_") or op == "run":
            return {"error": f"unknown op {op!r}", "ops": [t["name"] for t in TOOL_SPEC]}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad args for {op}: {e}"}
        except Exception as e:
            return {"error": f"{op} failed: {e}"}

    @staticmethod
    def _q(q) -> dict:
        return {"symbol": q.symbol, "name": q.name, "price": q.price,
                "change_24h_pct": q.change_24h_pct, "volume_24h": q.volume_24h,
                "market_cap": q.market_cap, "rank": q.rank, "source": q.source}


# OpenAI/Anthropic-style tool schema so an agent can discover the surface.
TOOL_SPEC = [
    {"name": "describe", "description": "List providers, engines, strategies, ops.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "market_snapshot", "description": "Top crypto quotes.",
     "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}}},
    {"name": "quote", "description": "One instrument snapshot.",
     "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"]}},
    {"name": "history", "description": "Close series for a symbol.",
     "parameters": {"type": "object", "properties": {"symbol": {"type": "string"},
                    "days": {"type": "integer"}}, "required": ["symbol"]}},
    {"name": "risk", "description": "Risk metrics (vol/Sharpe/VaR/drawdown).",
     "parameters": {"type": "object", "properties": {"symbol": {"type": "string"},
                    "days": {"type": "integer"}}, "required": ["symbol"]}},
    {"name": "list_engines", "description": "Available backtest/trade engines.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "list_strategies", "description": "Available strategies + params.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "backtest", "description": "Backtest a strategy on a symbol.",
     "parameters": {"type": "object", "properties": {
         "engine": {"type": "string"}, "strategy": {"type": "string"},
         "symbol": {"type": "string"}, "days": {"type": "integer"},
         "params": {"type": "object"}, "cost_bps": {"type": "number"}}}},
    {"name": "signal", "description": "Latest target position from a strategy.",
     "parameters": {"type": "object", "properties": {
         "engine": {"type": "string"}, "strategy": {"type": "string"},
         "symbol": {"type": "string"}}}},
    {"name": "rank_strategies", "description": "Backtest all strategies, rank by OOS Sharpe.",
     "parameters": {"type": "object", "properties": {"symbol": {"type": "string"},
                    "engine": {"type": "string"}}}},
]
