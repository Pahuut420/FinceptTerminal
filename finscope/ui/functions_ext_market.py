"""
Market data extension commands: EQ (equity quote), PM (Polymarket top), BOOK (crypto orderbook).

Handlers added to the command router via the extension hook in commands.py.
"""
from __future__ import annotations

import sys
from typing import List

from rich.text import Text
from rich.table import Table
from rich.panel import Panel

from finscope.core.contracts import AssetClass, ProviderError
from finscope.core.providers.base import get_provider
from finscope.ui import panels
from finscope.ui.commands import Result

AMBER = panels.AMBER
UP = panels.UP
DOWN = panels.DOWN
DIM = panels.DIM


def cmd_eq(router, args: List[str]) -> Result:
    """EQ <SYM> — Equity quote and description."""
    if not args:
        return Result(renderable=Text("Usage: EQ <SYMBOL>", style="bold red"))

    sym = args[0].upper()
    try:
        # Fetch quote
        q = router.hub.quote(sym, asset_class=AssetClass.EQUITY)
        if not q:
            return Result(renderable=Text(f"No quote for {sym}", style="bold red"))

        # Fetch history for graph
        series = router.hub.history(sym, days=90, asset_class=AssetClass.EQUITY)

        # Render security description panel
        return Result(renderable=panels.security_desc(q, series), status=f"EQ {sym}")
    except Exception as e:
        return Result(renderable=Text(f"EQ error: {e}", style="bold red"))


def cmd_pm(router, args: List[str]) -> Result:
    """PM [query] — Polymarket top active prediction markets."""
    query = " ".join(args) if args else None

    try:
        provider = get_provider("polymarket")
    except KeyError:
        return Result(renderable=Text(
            "Polymarket provider not registered (offline?)",
            style="bold red"))

    try:
        if query:
            quotes = provider.search(query, limit=15)
            title = f"POLYMARKET · {query[:30]}"
        else:
            quotes = provider.top(limit=15)
            title = "POLYMARKET · TOP ACTIVE"

        if not quotes:
            return Result(renderable=Text(
                "No prediction markets found",
                style="bold red"))

        # Render as a table: Question, Prob %, Volume
        t = Table(title=title, title_style=f"bold {AMBER}",
                  header_style=f"bold {AMBER}", border_style=DIM, expand=True)
        t.add_column("#", justify="right", style=DIM, no_wrap=True)
        t.add_column("MARKET QUESTION", no_wrap=False)
        t.add_column("YES %", justify="right", no_wrap=True)
        t.add_column("VOL 24H", justify="right", style=DIM, no_wrap=True)

        for i, q in enumerate(quotes, 1):
            # Price is the Yes probability (0-100)
            prob_text = Text(f"{q.price:5.1f}%", style=UP if q.price >= 50 else DOWN)
            vol_text = panels._fmt_big(q.volume_24h)
            t.add_row(str(i), q.name, prob_text, vol_text)

        return Result(renderable=t, status=f"{len(quotes)} markets")

    except ProviderError as e:
        return Result(renderable=Text(f"PM error: {e}", style="bold red"))
    except Exception as e:
        return Result(renderable=Text(f"PM unexpected error: {e}", style="bold red"))


def cmd_book(router, args: List[str]) -> Result:
    """BOOK <SYM> [depth] — Live Crypto.com orderbook."""
    if not args:
        return Result(renderable=Text("Usage: BOOK <SYMBOL> [depth]", style="bold red"))

    sym = args[0].upper()
    depth = 10
    if len(args) > 1:
        try:
            depth = int(args[1])
        except ValueError:
            depth = 10

    try:
        provider = get_provider("cryptocom")
    except KeyError:
        return Result(renderable=Text(
            "Crypto.com provider not registered (offline?)",
            style="bold red"))

    try:
        book = provider.orderbook(sym, depth=depth)
    except ProviderError as e:
        return Result(renderable=Text(f"BOOK error: {e}", style="bold red"))
    except Exception as e:
        return Result(renderable=Text(f"BOOK unexpected error: {e}", style="bold red"))

    # Render orderbook as a two-column table
    bids = book.get("bids", [])
    asks = book.get("asks", [])
    spread = book.get("spread")
    spread_pct = book.get("spread_pct")

    t = Table(title=f"ORDERBOOK · {sym}", title_style=f"bold {AMBER}",
              header_style=f"bold {AMBER}", border_style=DIM)
    t.add_column("BIDS", style=UP, justify="right")
    t.add_column("SIZE", style=DIM, justify="right")
    t.add_column("ASKS", style=DOWN, justify="right")
    t.add_column("SIZE", style=DIM, justify="right")

    # Merge bid/ask rows
    max_rows = max(len(bids), len(asks))
    for i in range(max_rows):
        bid_price = bids[i][0] if i < len(bids) else None
        bid_size = bids[i][1] if i < len(bids) else None
        ask_price = asks[i][0] if i < len(asks) else None
        ask_size = asks[i][1] if i < len(asks) else None

        bid_p_str = f"{bid_price:,.4f}" if bid_price is not None else "-"
        bid_s_str = f"{bid_size:,.2f}" if bid_size is not None else "-"
        ask_p_str = f"{ask_price:,.4f}" if ask_price is not None else "-"
        ask_s_str = f"{ask_size:,.2f}" if ask_size is not None else "-"

        t.add_row(bid_p_str, bid_s_str, ask_p_str, ask_s_str)

    # Add spread info as footer
    spread_info = ""
    if spread is not None and spread_pct is not None:
        spread_info = f" │ Spread: {spread:.4f} ({spread_pct:.3f}%)"

    status = f"{sym} {depth} levels{spread_info}"
    return Result(renderable=t, status=status)


# Extension hook
EXTRA_COMMANDS = {
    "EQ": cmd_eq,
    "PM": cmd_pm,
    "BOOK": cmd_book,
}

EXTRA_HELP = [
    ("EQ <SYM>", "Equity quote and description"),
    ("PM [query]", "Polymarket — top active prediction markets or search"),
    ("BOOK <SYM> [depth]", "Crypto.com live orderbook (bids/asks + spread)"),
]

# Register EXTRA_HELP with the commands module's _HELP_ROWS
# (commands.py only uses EXTRA_HELP from the first extension module, so we patch it here)
# Use sys.modules to access commands if it's in the process of being imported
try:
    if "finscope.ui.commands" in sys.modules:
        _commands_module = sys.modules["finscope.ui.commands"]
        if hasattr(_commands_module, '_HELP_ROWS'):
            # Check if already added (avoid duplicates)
            if not any(code == "EQ <SYM>" for code, _ in _commands_module._HELP_ROWS):
                _commands_module._HELP_ROWS.extend(EXTRA_HELP)
except Exception:
    pass  # optional integration
