"""
Rendering helpers — sparklines, quote tables, and the Bloomberg-ish chrome.

Everything returns a `rich` renderable (or a plain str for sparklines) so the
command router stays free of I/O and is unit-testable without a TTY.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.console import Group

from finscope.core.contracts import Quote, Series, RiskMetrics, CorrelationMatrix

# Bloomberg-ish palette: amber on black, green up / red down.
AMBER = "#ffb000"
UP = "bold green"
DOWN = "bold red"
DIM = "grey58"

_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: Sequence[float], width: int = 40) -> str:
    """Unicode block sparkline. Downsamples/pads to `width`."""
    vals = [float(v) for v in values if v == v]  # drop NaN
    if not vals:
        return ""
    if len(vals) > width:
        # simple bucket average downsample
        step = len(vals) / width
        vals = [
            sum(vals[int(i * step):max(int((i + 1) * step), int(i * step) + 1)])
            / max(1, (max(int((i + 1) * step), int(i * step) + 1) - int(i * step)))
            for i in range(width)
        ]
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    return "".join(_BLOCKS[min(len(_BLOCKS) - 1, int((v - lo) / rng * (len(_BLOCKS) - 1)))]
                   for v in vals)


def _fmt_price(x: Optional[float]) -> str:
    if x is None:
        return "-"
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    return f"{x:,.6f}"


def _fmt_big(x: Optional[float]) -> str:
    if x is None:
        return "-"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{x / div:.2f}{unit}"
    return f"{x:.0f}"


def _pct_text(p: Optional[float]) -> Text:
    if p is None:
        return Text("-", style=DIM)
    arrow = "▲" if p >= 0 else "▼"
    return Text(f"{arrow} {p:+.2f}%", style=UP if p >= 0 else DOWN)


def quotes_table(quotes: List[Quote], title: str = "MARKET MONITOR") -> Table:
    t = Table(title=title, title_style=f"bold {AMBER}", expand=True,
              header_style=f"bold {AMBER}", border_style=DIM)
    t.add_column("#", justify="right", style=DIM, no_wrap=True)
    t.add_column("SYM", style="bold white", no_wrap=True)
    t.add_column("NAME", style=DIM, no_wrap=True)
    t.add_column("LAST", justify="right", no_wrap=True)
    t.add_column("24H", justify="right", no_wrap=True)
    t.add_column("VOL 24H", justify="right", style=DIM, no_wrap=True)
    t.add_column("MKT CAP", justify="right", style=DIM, no_wrap=True)
    for q in quotes:
        px_style = UP if q.is_up else DOWN
        t.add_row(
            str(q.rank or ""), q.symbol, (q.name or "")[:18],
            Text(_fmt_price(q.price), style=px_style),
            _pct_text(q.change_24h_pct),
            _fmt_big(q.volume_24h), _fmt_big(q.market_cap),
        )
    return t


def security_desc(q: Quote, series: Optional[Series] = None) -> Panel:
    body = Table.grid(padding=(0, 2))
    body.add_column(style=f"bold {AMBER}", justify="right")
    body.add_column()
    body.add_row("Symbol", f"{q.symbol}  ({q.name})")
    body.add_row("Last", _fmt_price(q.price))
    body.add_row("24h", "")
    body.add_row("", _pct_text(q.change_24h_pct))
    body.add_row("Volume 24h", _fmt_big(q.volume_24h))
    body.add_row("Market Cap", _fmt_big(q.market_cap))
    body.add_row("Rank", str(q.rank or "-"))
    body.add_row("Source", q.source)
    if series and len(series):
        body.add_row("90d spark", sparkline(series.closes(), 48))
    return Panel(body, title=f"[bold {AMBER}]DES · {q.symbol}[/]", border_style=AMBER)


def price_graph(series: Series, q: Optional[Quote] = None) -> Panel:
    closes = series.closes()
    lo, hi = (min(closes), max(closes)) if closes else (0, 0)
    spark = sparkline(closes, 60)
    header = Table.grid(padding=(0, 2))
    header.add_column(style=f"bold {AMBER}", justify="right")
    header.add_column()
    header.add_row("Symbol", series.symbol)
    header.add_row("Points", str(len(series)))
    header.add_row("Range", f"{_fmt_price(lo)}  →  {_fmt_price(hi)}")
    if closes:
        chg = (closes[-1] - closes[0]) / closes[0] * 100 if closes[0] else 0
        header.add_row("Period Δ", "")
        header.add_row("", _pct_text(chg))
    header.add_row("Source", series.source)
    group = Group(header, Text(""), Text(spark, style=AMBER))
    return Panel(group, title=f"[bold {AMBER}]GP · {series.symbol}[/]", border_style=AMBER)


def risk_panel(rm: RiskMetrics) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style=f"bold {AMBER}", justify="right")
    t.add_column(justify="right")
    t.add_row("Observations", str(rm.n))
    t.add_row("Mean return", f"{rm.mean_return * 100:+.3f}%")
    t.add_row("Daily vol", f"{rm.daily_vol * 100:.3f}%")
    t.add_row("Annualized vol", f"{rm.annualized_vol * 100:.2f}%")
    t.add_row("Annualized return", f"{rm.annualized_return * 100:+.2f}%")
    t.add_row("Sharpe", f"{rm.sharpe:.3f}")
    t.add_row("Max drawdown", Text(f"{rm.max_drawdown * 100:.2f}%", style=DOWN))
    t.add_row("VaR 95% (1d)", Text(f"{rm.var_95 * 100:.2f}%", style=DOWN))
    t.add_row("Engine", rm.engine)
    return Panel(t, title=f"[bold {AMBER}]VOL/RISK · {rm.symbol}[/]", border_style=AMBER)


def correlation_table(cm: CorrelationMatrix) -> Table:
    t = Table(title="CORRELATION (returns)", title_style=f"bold {AMBER}",
              header_style=f"bold {AMBER}", border_style=DIM)
    t.add_column("", style=f"bold {AMBER}")
    for s in cm.symbols:
        t.add_column(s, justify="right")
    for i, s in enumerate(cm.symbols):
        cells: List[Text] = []
        for j in range(len(cm.symbols)):
            v = cm.matrix[i][j]
            style = "bold white" if i == j else (UP if v >= 0 else DOWN)
            cells.append(Text(f"{v:+.2f}", style=style))
        t.add_row(s, *cells)
    return t


def banner(width: int = 80) -> Text:
    return Text("FINSCOPE — Bloomberg-style terminal · built on FinceptTerminal data fleet",
                style=f"bold {AMBER}")
