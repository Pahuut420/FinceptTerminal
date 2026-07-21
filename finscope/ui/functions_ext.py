"""
functions_ext — swarm extension hook adding advanced-risk mnemonics.

Auto-registered by finscope.ui.commands (see the "optional extension hook" at
the bottom of that file): exposes EXTRA_COMMANDS = {"CODE": handler} and
EXTRA_HELP = [("CODE", "desc")]. Handlers take (router, args) -> Result, same
contract as CommandRouter.cmd_* methods, but as free functions so this module
never has to import/subclass CommandRouter.

New function codes:
    BETA <SYM> [days]      beta of SYM vs BTC (default 90d)
    SORTINO <SYM> [days]   annualized Sortino ratio (default 90d)
    ES <SYM> [days]        expected shortfall / CVaR 95% (default 90d)
    MC <SYM> [days]        Monte-Carlo VaR 95% via AnalyticsEngine (default 90d)

All handlers degrade gracefully: missing/short history returns a Result with
a red rich.Text error, never a traceback.
"""
from __future__ import annotations

from typing import List, Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from finscope.analytics.engine import AnalyticsEngine
from finscope.analytics.risk import AdvancedRisk
from finscope.core.contracts import Series
from finscope.ui import panels
from finscope.ui.commands import Result

AMBER = panels.AMBER
UP = panels.UP
DOWN = panels.DOWN
DIM = panels.DIM

_MIN_OBS = 3  # same "not enough history" floor commands.py uses for VOL/CORR
_DEFAULT_DAYS = 90
_BENCHMARK = "BTC"


def _int_arg(args: List[str], idx: int, default: int) -> int:
    try:
        return int(args[idx])
    except (IndexError, ValueError):
        return default


def _err(msg: str) -> Result:
    return Result(renderable=Text(msg, style="bold red"))


def _risk(router) -> AdvancedRisk:
    ppy = getattr(router.engine, "ppy", 365)
    return AdvancedRisk(periods_per_year=ppy)


def _fetch(router, sym: str, days: int) -> Optional[Series]:
    try:
        return router.hub.history(sym, days=days)
    except Exception:
        return None


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _grid() -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style=f"bold {AMBER}", justify="right")
    t.add_column(justify="right")
    return t


# ---- BETA -------------------------------------------------------------------
def cmd_beta(router, args: List[str]) -> Result:
    if not args:
        return _err("Usage: BETA <SYMBOL> [days]")
    sym = args[0].upper()
    days = _int_arg(args, 1, _DEFAULT_DAYS)

    series = _fetch(router, sym, days)
    if not series or len(series) < _MIN_OBS:
        return _err(f"Not enough history for {sym}")

    bench = _fetch(router, _BENCHMARK, days)
    if not bench or len(bench) < _MIN_OBS:
        return _err(f"Not enough benchmark history for {_BENCHMARK}")

    rk = _risk(router)
    beta = rk.beta(series, bench)
    n = min(len(series.returns()), len(bench.returns()))

    if beta > 1.2:
        note, style = "More volatile than benchmark", UP if beta >= 0 else DOWN
    elif beta < 0:
        note, style = "Inverse relationship to benchmark", DOWN
    elif beta < 0.8:
        note, style = "Less volatile than benchmark", DIM
    else:
        note, style = "Tracks benchmark closely", "bold white"

    t = _grid()
    t.add_row("Symbol", sym)
    t.add_row("Benchmark", _BENCHMARK)
    t.add_row("Observations", str(n))
    t.add_row("Beta", Text(f"{beta:.3f}", style=style))
    t.add_row("Read", Text(note, style=style))
    return Result(renderable=Panel(t, title=f"[bold {AMBER}]BETA · {sym} vs {_BENCHMARK}[/]",
                                   border_style=AMBER),
                  status=f"{sym} beta={beta:.3f}")


# ---- SORTINO ------------------------------------------------------------------
def cmd_sortino(router, args: List[str]) -> Result:
    if not args:
        return _err("Usage: SORTINO <SYMBOL> [days]")
    sym = args[0].upper()
    days = _int_arg(args, 1, _DEFAULT_DAYS)

    series = _fetch(router, sym, days)
    if not series or len(series) < _MIN_OBS:
        return _err(f"Not enough history for {sym}")

    rk = _risk(router)
    sortino = rk.sortino(series)
    dd = rk.downside_deviation(series)
    rets = series.returns()
    ann_return = (sum(rets) / len(rets)) * rk.ppy if rets else 0.0

    style = UP if sortino >= 0 else DOWN
    t = _grid()
    t.add_row("Symbol", sym)
    t.add_row("Observations", str(len(rets)))
    t.add_row("Annualized return", _pct(ann_return))
    t.add_row("Downside deviation", _pct(dd))
    t.add_row("Sortino", Text(f"{sortino:.3f}", style=style))
    return Result(renderable=Panel(t, title=f"[bold {AMBER}]SORTINO · {sym}[/]",
                                   border_style=AMBER),
                  status=f"{sym} sortino={sortino:.3f}")


# ---- ES / CVaR ----------------------------------------------------------------
def cmd_es(router, args: List[str]) -> Result:
    if not args:
        return _err("Usage: ES <SYMBOL> [days]")
    sym = args[0].upper()
    days = _int_arg(args, 1, _DEFAULT_DAYS)

    series = _fetch(router, sym, days)
    if not series or len(series) < _MIN_OBS:
        return _err(f"Not enough history for {sym}")

    rk = _risk(router)
    es = rk.expected_shortfall(series, confidence=0.95)
    var95 = rk.parametric_var(series, confidence=0.95)
    n = len(series.returns())

    t = _grid()
    t.add_row("Symbol", sym)
    t.add_row("Observations", str(n))
    t.add_row("Confidence", "95%")
    t.add_row("Parametric VaR (1d)", Text(_pct(var95), style=DOWN))
    t.add_row("Expected Shortfall (CVaR)", Text(_pct(es), style=DOWN))
    return Result(renderable=Panel(t, title=f"[bold {AMBER}]ES · {sym} 95% CVaR[/]",
                                   border_style=AMBER),
                  status=f"{sym} es={es:.4f}")


# ---- MC (Monte-Carlo VaR) ------------------------------------------------------
def cmd_mc(router, args: List[str]) -> Result:
    if not args:
        return _err("Usage: MC <SYMBOL> [days]")
    sym = args[0].upper()
    days = _int_arg(args, 1, _DEFAULT_DAYS)

    series = _fetch(router, sym, days)
    if not series or len(series) < _MIN_OBS:
        return _err(f"Not enough history for {sym}")

    engine: AnalyticsEngine = router.engine or AnalyticsEngine()
    mc_var = engine.monte_carlo_var(series, horizon=1, sims=10000, confidence=0.95)
    n = len(series.returns())

    t = _grid()
    t.add_row("Symbol", sym)
    t.add_row("Observations", str(n))
    t.add_row("Simulations", "10,000")
    t.add_row("Horizon", "1d")
    t.add_row("Confidence", "95%")
    t.add_row("Monte-Carlo VaR (1d)", Text(_pct(mc_var), style=DOWN))
    return Result(renderable=Panel(t, title=f"[bold {AMBER}]MC · {sym} Monte-Carlo VaR[/]",
                                   border_style=AMBER),
                  status=f"{sym} mc_var={mc_var:.4f}")


EXTRA_COMMANDS = {
    "BETA": cmd_beta,
    "SORTINO": cmd_sortino,
    "ES": cmd_es,
    "MC": cmd_mc,
}

EXTRA_HELP = [
    ("BETA <SYM> [days]", "Beta of SYM vs BTC (OLS, default 90d)"),
    ("SORTINO <SYM> [days]", "Annualized Sortino ratio (downside deviation)"),
    ("ES <SYM> [days]", "Expected shortfall / CVaR 95% (+ parametric VaR)"),
    ("MC <SYM> [days]", "Monte-Carlo VaR 95% (10,000 sims)"),
]
