"""
FinScope engines — pluggable trading/backtest backends behind one interface.

Each engine (native numpy, freqtrade, nautilus_trader, neural-trader, TradingAgents)
adapts to the `Engine` ABC so they can be tested and driven interchangeably — by a
human at the terminal or by an LLM agent via the JSON `run()` surface.

The `native` engine is always available and $0 (pure numpy); external engines
detect their repo/install and degrade gracefully.
"""
from finscope.engines.base import (
    Engine, Signal, BacktestResult, Strategy, register_engine, all_engines, get_engine,
)

# auto-discover engine adapters (self-register on import)
from finscope.engines import _discover as _d  # noqa: E402
_d.autodiscover()

__all__ = ["Engine", "Signal", "BacktestResult", "Strategy",
           "register_engine", "all_engines", "get_engine"]
