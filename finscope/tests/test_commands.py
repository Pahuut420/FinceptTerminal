"""
Tests for finscope.ui.commands.CommandRouter.

CommandRouter.dispatch() is documented to never raise -- any handler exception
is caught in dispatch() itself and turned into an error Result. These tests
exercise the full required command surface (HELP, CRYPTO 5, DES BTC, VOL BTC,
CORR BTC ETH SOL, GP BTC, NONSENSE) plus a couple of malformed-input paths, and
assert each call returns a proper Result without an exception escaping.

Works both under pytest (functions named test_*) and standalone:
    python3 finscope/tests/test_commands.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from finscope.ui.commands import CommandRouter, Result, Signal


REQUIRED_COMMANDS = ["HELP", "CRYPTO 5", "DES BTC", "VOL BTC", "CORR BTC ETH SOL",
                     "GP BTC", "NONSENSE"]


def _renderable_text(res: Result) -> str:
    """Extract plain text from whatever rich renderable dispatch() produced,
    for substring assertions. Text has .plain; other renderables we just str()."""
    r = res.renderable
    if isinstance(r, Text):
        return r.plain
    return str(r)


# ---- required command surface never raises -----------------------------------------

def test_dispatch_never_raises_for_required_commands():
    router = CommandRouter()
    for line in REQUIRED_COMMANDS:
        res = router.dispatch(line)  # must not raise
        assert isinstance(res, Result), f"{line!r} did not return a Result"


def test_help_returns_populated_table():
    router = CommandRouter()
    res = router.dispatch("HELP")
    assert isinstance(res, Result)
    assert isinstance(res.renderable, Table)
    assert res.status == "help"
    # every documented function code should appear as a row in the help table
    assert res.renderable.row_count >= 10


def test_crypto_5_returns_5_quotes():
    router = CommandRouter()
    res = router.dispatch("CRYPTO 5")
    assert isinstance(res.renderable, Table)
    assert res.status == "5 coins"


def test_des_btc_returns_security_description_panel():
    router = CommandRouter()
    res = router.dispatch("DES BTC")
    assert isinstance(res.renderable, Panel)
    assert res.status == "BTC"


def test_vol_btc_returns_risk_panel():
    router = CommandRouter()
    res = router.dispatch("VOL BTC")
    assert isinstance(res.renderable, Panel)
    assert res.status == "BTC risk"


def test_corr_three_symbols_returns_correlation_table():
    router = CommandRouter()
    res = router.dispatch("CORR BTC ETH SOL")
    assert isinstance(res.renderable, Table)
    assert res.status == "corr"


def test_gp_btc_returns_price_graph_panel():
    router = CommandRouter()
    res = router.dispatch("GP BTC")
    assert isinstance(res.renderable, Panel)
    assert res.status == "BTC 90d"


def test_nonsense_command_yields_error_result_not_exception():
    router = CommandRouter()
    res = router.dispatch("NONSENSE")
    assert isinstance(res, Result)
    assert isinstance(res.renderable, Text)
    text = _renderable_text(res)
    assert "unknown function" in text.lower()
    assert "NONSENSE" in text


# ---- malformed-input paths also degrade gracefully (no exception) -------------------

def test_des_without_symbol_yields_usage_error_not_exception():
    router = CommandRouter()
    res = router.dispatch("DES")
    assert isinstance(res, Result)
    assert isinstance(res.renderable, Text)
    assert "usage" in _renderable_text(res).lower()


def test_gp_without_symbol_yields_usage_error_not_exception():
    router = CommandRouter()
    res = router.dispatch("GP")
    assert isinstance(res, Result)
    assert isinstance(res.renderable, Text)
    assert "usage" in _renderable_text(res).lower()


def test_vol_unknown_symbol_yields_error_not_exception():
    router = CommandRouter()
    res = router.dispatch("VOL NOT_A_REAL_SYMBOL_ZZZ")
    assert isinstance(res, Result)
    assert isinstance(res.renderable, Text)
    assert "not enough history" in _renderable_text(res).lower() or \
           "no history" in _renderable_text(res).lower()


def test_corr_single_unknown_symbol_yields_error_not_exception():
    router = CommandRouter()
    res = router.dispatch("CORR NOT_A_REAL_SYMBOL_ZZZ")
    assert isinstance(res, Result)
    assert isinstance(res.renderable, Text)


def test_empty_line_returns_empty_result_not_exception():
    router = CommandRouter()
    res = router.dispatch("")
    assert isinstance(res, Result)
    assert res.status == ""


def test_dispatch_is_case_insensitive():
    router = CommandRouter()
    res = router.dispatch("help")
    assert isinstance(res.renderable, Table)
    assert res.status == "help"


# ---- quit signal --------------------------------------------------------------------

def test_quit_returns_quit_signal():
    router = CommandRouter()
    res = router.dispatch("QUIT")
    assert isinstance(res, Result)
    assert isinstance(res.signal, Signal)
    assert res.signal.kind == "quit"


# ---- test runner (standalone mode) --------------------------------------------------

def _run_standalone() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            failed += 1
        else:
            print(f"PASS {name}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_standalone() else 0)
