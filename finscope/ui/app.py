"""
FinScopeApp — the terminal shell.

Modes:
  * REPL (default): Bloomberg-style command line. Type function codes.
  * --live: auto-refreshing crypto market monitor (rich.Live), Ctrl-C to exit.
  * --once "<CMD>": run one command, print, exit (headless / CI / scripting).
  * snapshot(): render a representative frame — used by the smoke test.

The Console is owned here; the CommandRouter stays I/O-free.
"""
from __future__ import annotations

import time
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule

from finscope.core.contracts import AssetClass
from finscope.core.datahub import DataHub
from finscope.analytics.engine import AnalyticsEngine
from finscope.analytics.wolfram_bridge import WolframBridge
from finscope.ui.commands import CommandRouter, Signal
from finscope.ui import panels

AMBER = panels.AMBER


class FinScopeApp:
    def __init__(self, console: Optional[Console] = None,
                 router: Optional[CommandRouter] = None):
        self.console = console or Console()
        self.router = router or CommandRouter(
            hub=DataHub(), engine=AnalyticsEngine(), wolfram=WolframBridge())

    # ---- chrome ----
    def _header(self) -> Panel:
        hub = self.router.hub
        left = Text("FINSCOPE ", style=f"bold {AMBER}")
        left.append("v0.1", style="grey58")
        right = Text(f" {len(hub.providers)} providers · "
                     f"watch: {', '.join(hub.watchlist[:6])}", style="grey58")
        return Panel(Group(left, right),
                     title="[bold]Bloomberg-style terminal · FinceptTerminal data fleet[/]",
                     border_style=AMBER)

    def _status_line(self, status: str = "") -> Text:
        t = Text()
        t.append("FINSCOPE> ", style=f"bold {AMBER}")
        if status:
            t.append(f"[{status}]", style="grey58")
        return t

    # ---- one command ----
    def run_command(self, line: str) -> Optional[Signal]:
        result = self.router.dispatch(line)
        if result.renderable is not None:
            self.console.print(result.renderable)
        if result.signal and result.signal.kind == "quit":
            self.console.print(Text("Goodbye.", style=AMBER))
        return result.signal

    # ---- headless snapshot (used by tests / --demo) ----
    def snapshot(self) -> None:
        self.console.print(self._header())
        self.console.print(Rule(style="grey30"))
        for cmd in ("PROV", "HELP"):
            self.run_command(cmd)
        # try a live monitor; if offline, PROV/HELP already prove the render path
        self.run_command("CRYPTO 10")

    # ---- live auto-refresh monitor ----
    def live(self, interval: float = 10.0, limit: int = 25) -> None:
        from rich.live import Live
        try:
            with Live(console=self.console, screen=False, auto_refresh=False) as live:
                while True:
                    quotes = self.router.hub.top(AssetClass.CRYPTO, limit=limit)
                    frame = Group(
                        self._header(),
                        panels.quotes_table(quotes, f"LIVE CRYPTO MONITOR · {time.strftime('%H:%M:%S')}"),
                        Text("Ctrl-C to exit live mode", style="grey58"),
                    )
                    live.update(frame, refresh=True)
                    time.sleep(interval)
        except KeyboardInterrupt:
            self.console.print(Text("\nExited live monitor.", style=AMBER))

    # ---- interactive REPL ----
    def run(self) -> None:
        self.console.print(self._header())
        self.console.print(Text("Type HELP for functions, Q to quit. Try: CRYPTO · GP BTC · VOL ETH · CORR",
                                style="grey58"))
        while True:
            try:
                line = self.console.input(self._status_line())
            except (EOFError, KeyboardInterrupt):
                self.console.print(Text("\nGoodbye.", style=AMBER))
                return
            sig = self.run_command(line)
            if sig and sig.kind == "quit":
                return
