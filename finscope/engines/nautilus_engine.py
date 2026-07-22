"""
Nautilus Trader adapter.

Nautilus (github.com/nautechsystems/nautilus_trader) is a high-performance,
event-driven backtest/live platform (Rust core + Cython/Python API). Its
`BacktestEngine` needs a compiled install. This adapter detects it, reports how
to enable, and exposes the uniform Engine surface; live wiring runs on the user's
machine where nautilus is built.
"""
from __future__ import annotations

from typing import Optional

from finscope.engines.base import Engine, register_engine
from finscope.engines import _detect


@register_engine
class NautilusEngine(Engine):
    name = "nautilus"
    kind = "backtest"
    description = "Nautilus Trader — event-driven backtest/live (Rust+Cython). Detect + describe."

    def _repo(self) -> Optional[str]:
        return _detect.find_repo("nautilus_trader")

    def available(self) -> bool:
        return _detect.module_installed("nautilus_trader")

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "installed_module": _detect.module_installed("nautilus_trader"),
            "repo_clone": self._repo(),
            "entrypoint": "nautilus_trader.backtest.engine.BacktestEngine",
            "enable": "pip install nautilus_trader  (prebuilt wheels); then build a "
                      "BacktestEngine, add a venue + data, and a Strategy that maps "
                      "FinScope weights to order intents.",
        })
        return d

    def run(self, request: dict) -> dict:
        op = request.get("op", "describe")
        if op == "describe":
            return self.describe()
        if op == "backtest":
            if not self.available():
                return {"error": "nautilus_trader not installed",
                        "hint": "pip install nautilus_trader", "describe": self.describe()}
            return {"error": "nautilus live wiring runs on a machine with nautilus built; "
                             "use the BacktestEngine entrypoint from describe()."}
        return {"error": f"nautilus: unsupported op {op!r}"}
