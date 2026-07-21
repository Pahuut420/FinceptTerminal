"""
Command router — Bloomberg-style mnemonic functions.

Kept free of terminal I/O: every handler returns a `rich` renderable (or a
`Signal` for control flow like quitting). The app layer owns the Console. This
makes the whole command surface unit-testable headlessly.

Function codes (type HELP in the app):
    HELP / H            list functions
    CRYPTO / WEI        crypto market monitor (top by mkt cap)
    TOP <n>             top n coins
    MOVERS             biggest 24h gainers & losers
    DES <SYM>          security description
    GP <SYM> [days]    price graph (sparkline + range)
    VOL <SYM> [days]   risk/volatility metrics
    CORR [SYMS...]     correlation matrix (watchlist by default)
    WATCH <SYM...>     show/set the watchlist
    API <query>        search the public-apis free-API registry
    PROV               providers + health
    WL <SYM>           show the Wolfram Language for this coin's vol (offload demo)
    Q / QUIT / EXIT    leave
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rich.text import Text
from rich.table import Table
from rich.panel import Panel

from finscope.core.contracts import AssetClass
from finscope.core.datahub import DataHub
from finscope.analytics.engine import AnalyticsEngine
from finscope.analytics.wolfram_bridge import WolframBridge
from finscope.ui import panels

AMBER = panels.AMBER


@dataclass
class Signal:
    kind: str  # "quit" | "clear" | "none"
    message: str = ""


@dataclass
class Result:
    renderable: object = None
    signal: Optional[Signal] = None
    status: str = ""  # short status-line text


class CommandRouter:
    def __init__(self, hub: Optional[DataHub] = None,
                 engine: Optional[AnalyticsEngine] = None,
                 wolfram: Optional[WolframBridge] = None):
        self.hub = hub or DataHub()
        self.engine = engine or AnalyticsEngine()
        self.wolfram = wolfram or WolframBridge()

    # ---- dispatch ----
    def dispatch(self, line: str) -> Result:
        line = (line or "").strip()
        if not line:
            return Result(status="")
        parts = line.split()
        cmd, args = parts[0].upper(), parts[1:]
        handler = _COMMANDS.get(cmd)
        if not handler:
            return Result(renderable=Text(f"Unknown function {cmd!r}. Type HELP.",
                                          style="bold red"))
        try:
            return handler(self, args)
        except Exception as e:  # a bad command must never crash the terminal
            return Result(renderable=Text(f"{cmd} error: {e}", style="bold red"))

    # ---- handlers ----
    def cmd_help(self, args: List[str]) -> Result:
        t = Table(title="FINSCOPE FUNCTIONS", title_style=f"bold {AMBER}",
                  header_style=f"bold {AMBER}", border_style="grey58")
        t.add_column("CODE", style="bold white", no_wrap=True)
        t.add_column("DESCRIPTION")
        for code, desc in _HELP_ROWS:
            t.add_row(code, desc)
        return Result(renderable=t, status="help")

    def cmd_crypto(self, args: List[str]) -> Result:
        n = _int_arg(args, 0, 20)
        quotes = self.hub.top(AssetClass.CRYPTO, limit=n)
        if not quotes:
            return Result(renderable=Text("No data (offline?). Try again.", style="bold red"))
        return Result(renderable=panels.quotes_table(quotes, f"CRYPTO · TOP {n}"),
                      status=f"{len(quotes)} coins")

    def cmd_top(self, args: List[str]) -> Result:
        return self.cmd_crypto(args or ["25"])

    def cmd_movers(self, args: List[str]) -> Result:
        quotes = self.hub.top(AssetClass.CRYPTO, limit=200)
        graded = [q for q in quotes if q.change_24h_pct is not None]
        graded.sort(key=lambda q: q.change_24h_pct or 0.0)
        losers, gainers = graded[:8], list(reversed(graded[-8:]))
        gt = panels.quotes_table(gainers, "TOP GAINERS 24H")
        lt = panels.quotes_table(losers, "TOP LOSERS 24H")
        from rich.console import Group
        return Result(renderable=Group(gt, lt), status="movers")

    def cmd_des(self, args: List[str]) -> Result:
        if not args:
            return Result(renderable=Text("Usage: DES <SYMBOL>", style="bold red"))
        sym = args[0].upper()
        q = self.hub.quote(sym)
        if not q:
            return Result(renderable=Text(f"No quote for {sym}", style="bold red"))
        series = self.hub.history(sym, days=90)
        return Result(renderable=panels.security_desc(q, series), status=sym)

    def cmd_gp(self, args: List[str]) -> Result:
        if not args:
            return Result(renderable=Text("Usage: GP <SYMBOL> [days]", style="bold red"))
        sym = args[0].upper()
        days = _int_arg(args, 1, 90)
        series = self.hub.history(sym, days=days)
        if not series or not len(series):
            return Result(renderable=Text(f"No history for {sym}", style="bold red"))
        q = self.hub.quote(sym)
        return Result(renderable=panels.price_graph(series, q), status=f"{sym} {days}d")

    def cmd_vol(self, args: List[str]) -> Result:
        if not args:
            return Result(renderable=Text("Usage: VOL <SYMBOL> [days]", style="bold red"))
        sym = args[0].upper()
        days = _int_arg(args, 1, 180)
        series = self.hub.history(sym, days=days)
        if not series or len(series) < 3:
            return Result(renderable=Text(f"Not enough history for {sym}", style="bold red"))
        rm = self.engine.risk_metrics(series)
        return Result(renderable=panels.risk_panel(rm), status=f"{sym} risk")

    def cmd_corr(self, args: List[str]) -> Result:
        syms = [a.upper() for a in args] or self.hub.watchlist[:6]
        series = [self.hub.history(s, days=90) for s in syms]
        series = [s for s in series if s and len(s) >= 3]
        if len(series) < 2:
            return Result(renderable=Text("Need >=2 symbols with history for CORR",
                                          style="bold red"))
        cm = self.engine.correlation(series)
        return Result(renderable=panels.correlation_table(cm), status="corr")

    def cmd_watch(self, args: List[str]) -> Result:
        if args:
            self.hub.watchlist = [a.upper() for a in args]
        quotes = self.hub.watchlist_quotes()
        return Result(renderable=panels.quotes_table(quotes, "WATCHLIST"),
                      status=f"watch={','.join(self.hub.watchlist)}")

    def cmd_api(self, args: List[str]) -> Result:
        query = " ".join(args)
        try:
            from finscope.core.providers import public_apis_registry as reg  # type: ignore
        except Exception:
            return Result(renderable=Text(
                "public-apis registry not installed yet (swarm task). "
                "Falls back once finscope.core.providers.public_apis_registry lands.",
                style="grey58"))
        rows = reg.search(query) if query else reg.sample()
        t = Table(title=f"PUBLIC-APIS · {query or 'sample'}", title_style=f"bold {AMBER}",
                  header_style=f"bold {AMBER}", border_style="grey58")
        for col in ("API", "CATEGORY", "AUTH", "HTTPS", "DESCRIPTION"):
            t.add_column(col)
        for r in rows[:25]:
            t.add_row(r.get("name", "-"), r.get("category", "-"), r.get("auth", "-") or "None",
                      "Y" if r.get("https") else "N", (r.get("description", "") or "")[:50])
        return Result(renderable=t, status=f"{len(rows)} apis")

    def cmd_prov(self, args: List[str]) -> Result:
        health = self.hub.health()
        t = Table(title="PROVIDERS", title_style=f"bold {AMBER}",
                  header_style=f"bold {AMBER}", border_style="grey58")
        t.add_column("NAME", style="bold white")
        t.add_column("ASSET CLASSES")
        t.add_column("CAPS")
        t.add_column("HEALTH")
        for name, p in sorted(self.hub.providers.items()):
            caps = "/".join(c for c in ("quote", "top", "history", "search")
                            if getattr(p, f"can_{c}"))
            ok = health.get(name, False)
            t.add_row(name, ",".join(a.value for a in p.asset_classes), caps,
                      Text("● up", style="green") if ok else Text("● down", style="red"))
        return Result(renderable=t, status=f"{len(self.hub.providers)} providers")

    def cmd_wl(self, args: List[str]) -> Result:
        if not args:
            return Result(renderable=Text("Usage: WL <SYMBOL>", style="bold red"))
        sym = args[0].upper()
        series = self.hub.history(sym, days=90)
        if not series or len(series) < 3:
            return Result(renderable=Text(f"No history for {sym}", style="bold red"))
        wl = self.wolfram.wl_annualized_vol(series.returns())
        body = Text(wl, style=AMBER)
        return Result(renderable=Panel(body, title=f"[bold {AMBER}]WL offload · {sym} annualized vol[/]",
                                       border_style=AMBER),
                      status="wolfram source")

    def cmd_quit(self, args: List[str]) -> Result:
        return Result(signal=Signal("quit", "bye"))


def _int_arg(args: List[str], idx: int, default: int) -> int:
    try:
        return int(args[idx])
    except (IndexError, ValueError):
        return default


_COMMANDS = {
    "HELP": CommandRouter.cmd_help, "H": CommandRouter.cmd_help, "?": CommandRouter.cmd_help,
    "CRYPTO": CommandRouter.cmd_crypto, "WEI": CommandRouter.cmd_crypto,
    "TOP": CommandRouter.cmd_top,
    "MOVERS": CommandRouter.cmd_movers,
    "DES": CommandRouter.cmd_des,
    "GP": CommandRouter.cmd_gp,
    "VOL": CommandRouter.cmd_vol,
    "CORR": CommandRouter.cmd_corr,
    "WATCH": CommandRouter.cmd_watch, "WL": CommandRouter.cmd_wl,
    "API": CommandRouter.cmd_api,
    "PROV": CommandRouter.cmd_prov,
    "Q": CommandRouter.cmd_quit, "QUIT": CommandRouter.cmd_quit, "EXIT": CommandRouter.cmd_quit,
}

# ---- optional extension hook -------------------------------------------------
# Swarm agents can add function codes without editing this file: drop a module
# `finscope/ui/functions_ext.py` exposing EXTRA_COMMANDS = {"CODE": handler} and
# EXTRA_HELP = [("CODE", "desc")]. Handlers take (router, args) -> Result.
_ext_modules = []
for _mod_name in ("functions_ext", "functions_ext_market"):
    try:  # pragma: no cover - optional
        import importlib
        _m = importlib.import_module(f"finscope.ui.{_mod_name}")
        _COMMANDS.update(getattr(_m, "EXTRA_COMMANDS", {}))
        _ext_modules.append(_m)
    except Exception:
        pass
_ext = _ext_modules[0] if _ext_modules else None

_HELP_ROWS = [
    ("CRYPTO / WEI", "Crypto market monitor — top coins by market cap"),
    ("TOP <n>", "Top n coins"),
    ("MOVERS", "Biggest 24h gainers & losers"),
    ("DES <SYM>", "Security description"),
    ("GP <SYM> [days]", "Price graph — sparkline + range"),
    ("VOL <SYM> [days]", "Risk/volatility metrics (vol, Sharpe, VaR, drawdown)"),
    ("CORR [SYMS...]", "Correlation matrix (watchlist by default)"),
    ("WATCH <SYM...>", "Show/set the watchlist"),
    ("WL <SYM>", "Show the Wolfram Language for annualized vol (offload demo)"),
    ("API <query>", "Search the public-apis free-API registry"),
    ("PROV", "List data providers + health"),
    ("HELP / H / ?", "This screen"),
    ("Q / QUIT / EXIT", "Leave FinScope"),
]

for _m in _ext_modules:
    _HELP_ROWS.extend(getattr(_m, "EXTRA_HELP", []))
