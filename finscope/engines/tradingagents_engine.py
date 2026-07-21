"""
TradingAgents adapter (LLM multi-agent — budget-gated).

TradingAgents (github.com/TauricResearch/TradingAgents) runs a graph of LLM
analysts/researchers/traders that debate and emit a BUY/HOLD/SELL decision via
`TradingAgentsGraph.propagate(ticker, date)`. It costs real model tokens, so this
adapter is OFF by default and only runs when BOTH:

  * `FINSCOPE_ENABLE_TRADINGAGENTS=1`, and
  * a cheap model is configured (env `TRADINGAGENTS_*`, e.g. a local/ollama or
    Haiku backend) — enforced by a soft check so an expensive default can't fire.

Otherwise it degrades to a describe() with enablement instructions. This keeps the
LLM engine from ever surprising the token budget.
"""
from __future__ import annotations

import os
from typing import Optional

from finscope.engines.base import Engine, register_engine
from finscope.engines import _detect

_CHEAP_HINTS = ("haiku", "ollama", "localhost", "127.0.0.1", "mini", "flash", "small")


@register_engine
class TradingAgentsEngine(Engine):
    name = "tradingagents"
    kind = "agent"
    description = "TradingAgents — LLM multi-agent decision graph. Budget-gated (off by default)."

    def _repo(self) -> Optional[str]:
        return _detect.find_repo("tradingagents", "TradingAgents")

    def available(self) -> bool:
        return _detect.module_installed("tradingagents") or bool(self._repo())

    def _enabled(self) -> bool:
        return os.environ.get("FINSCOPE_ENABLE_TRADINGAGENTS") == "1"

    def _cheap_model_set(self) -> bool:
        blob = " ".join(v.lower() for k, v in os.environ.items() if k.startswith("TRADINGAGENTS"))
        return any(h in blob for h in _CHEAP_HINTS)

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "installed_module": _detect.module_installed("tradingagents"),
            "repo_clone": self._repo(),
            "enabled": self._enabled(),
            "cheap_model_configured": self._cheap_model_set(),
            "entrypoint": "tradingagents.graph.trading_graph.TradingAgentsGraph.propagate(ticker, date)",
            "enable": "export FINSCOPE_ENABLE_TRADINGAGENTS=1 and point TRADINGAGENTS_* env "
                      "at a cheap backend (ollama / Haiku), then op=decide.",
        })
        return d

    def run(self, request: dict) -> dict:
        op = request.get("op", "describe")
        if op == "describe":
            return self.describe()
        if op == "decide":
            if not self.available():
                return {"error": "tradingagents not installed", "describe": self.describe()}
            if not self._enabled():
                return {"error": "TradingAgents is off (budget guard). "
                                 "Set FINSCOPE_ENABLE_TRADINGAGENTS=1 to enable.",
                        "describe": self.describe()}
            if not self._cheap_model_set():
                return {"error": "refusing to run on an unconfigured/expensive model. "
                                 "Set TRADINGAGENTS_* to a cheap backend (ollama/Haiku) first."}
            ticker = request.get("ticker", "SPY")
            date = request.get("date")
            try:
                from tradingagents.graph.trading_graph import TradingAgentsGraph  # type: ignore
                from tradingagents.default_config import DEFAULT_CONFIG  # type: ignore
                ta = TradingAgentsGraph(debug=False, config=DEFAULT_CONFIG.copy())
                _, decision = ta.propagate(ticker, date)
                return {"engine": "tradingagents", "ticker": ticker, "date": date,
                        "decision": decision}
            except Exception as e:
                return {"error": f"tradingagents run failed: {e}"}
        return {"error": f"tradingagents: unsupported op {op!r}"}
