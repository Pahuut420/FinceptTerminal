"""
Freqtrade adapter.

Freqtrade (github.com/freqtrade/freqtrade) is a mature crypto backtesting/live
bot built on the `IStrategy` interface. This adapter:
  * detects freqtrade (pip module or sibling clone at /workspace/freqtrade),
  * exposes a uniform `describe()`/`available()`,
  * `run({"op":"backtest",...})` shells out to `freqtrade backtesting` when a
    user config + strategy are provided,
  * `export_strategy()` emits a freqtrade `IStrategy` scaffold wrapping a FinScope
    strategy's causal weights (long/short entries from weight sign changes).

Nothing is imported from freqtrade at load time, so this file is always safe.
"""
from __future__ import annotations

import subprocess
from typing import Optional

from finscope.engines.base import Engine, register_engine
from finscope.engines import _detect


@register_engine
class FreqtradeEngine(Engine):
    name = "freqtrade"
    kind = "backtest"
    description = "Freqtrade crypto bot (IStrategy) — backtest/live. Detect + shell-out."

    def _repo(self) -> Optional[str]:
        return _detect.find_repo("freqtrade")

    def available(self) -> bool:
        return _detect.module_installed("freqtrade") or bool(_detect.which("freqtrade"))

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "installed_module": _detect.module_installed("freqtrade"),
            "cli": _detect.which("freqtrade"),
            "repo_clone": self._repo(),
            "enable": "pip install freqtrade  (or use the sibling clone) — then "
                      "`freqtrade backtesting --strategy <S> --timerange <T>`",
        })
        return d

    def run(self, request: dict) -> dict:
        op = request.get("op", "describe")
        if op == "describe":
            return self.describe()
        if op == "backtest":
            if not self.available():
                return {"error": "freqtrade not installed",
                        "hint": "pip install freqtrade", "describe": self.describe()}
            args = request.get("args") or ["backtesting"]
            try:
                r = subprocess.run(["freqtrade", *args], capture_output=True,
                                   text=True, timeout=request.get("timeout", 600))
                return {"engine": "freqtrade", "returncode": r.returncode,
                        "stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:]}
            except (subprocess.SubprocessError, OSError) as e:
                return {"error": f"freqtrade run failed: {e}"}
        if op == "export_strategy":
            return {"scaffold": _ISTRATEGY_SCAFFOLD.format(
                name=request.get("name", "FinScopeStrategy"))}
        return {"error": f"freqtrade: unsupported op {op!r}"}


_ISTRATEGY_SCAFFOLD = '''\
# Auto-generated freqtrade IStrategy wrapping a FinScope causal strategy.
# Fill in `finscope_weights()` to import your FinScope strategy's weights().
from freqtrade.strategy import IStrategy
from pandas import DataFrame

class {name}(IStrategy):
    timeframe = "1d"
    minimal_roi = {{"0": 10}}
    stoploss = -0.15

    def populate_indicators(self, df: DataFrame, meta: dict) -> DataFrame:
        # from finscope.engines.strategies import build; from finscope.core.contracts import Series
        # w = build("momentum").weights(Series.from_closes(meta["pair"], df["close"].tolist()))
        # df["fs_weight"] = w
        return df

    def populate_entry_trend(self, df: DataFrame, meta: dict) -> DataFrame:
        df.loc[df.get("fs_weight", 0) > 0.05, "enter_long"] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, meta: dict) -> DataFrame:
        df.loc[df.get("fs_weight", 0) <= 0, "exit_long"] = 1
        return df
'''
