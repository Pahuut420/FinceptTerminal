"""
DataHub.top()/quotes() must MERGE across providers, not stop at the first
non-empty-but-incomplete response.

Real-world bug this guards against: BlockchainComProvider is a deliberate
BTC-only cross-check (see core/providers/blockchain_com.py) whose `.top()`
always returns exactly one Quote and whose `.quotes()` only ever answers for
BTC. In the sandbox this session develops in, outbound network is blocked, so
every live provider raises ProviderError and DataHub always falls through to
the bundled `demo` provider (priority 900) -- masking the bug completely.
On GitHub Actions runners (real network), blockchain.com actually answers,
and the old accept-first-non-empty logic silently truncated `top(limit=5)`
to 1 result and `quotes(["BTC","ETH"])` to just `{"BTC"}` -- caught for the
first time by finscope-tests.yml's CI run (PR #12), never before that.

These tests reproduce that exact shape with fake in-process providers (no
network, fully deterministic) so the regression can't come back silently.
"""
from __future__ import annotations

from typing import List

from finscope.core.contracts import AssetClass, Quote
from finscope.core.datahub import DataHub


class _FakeProvider:
    """Minimal stand-in with only the surface DataHub actually calls."""

    def __init__(self, name: str, top_rows: List[Quote] = None, quote_map: dict = None):
        self.name = name
        self._top_rows = top_rows or []
        self._quote_map = quote_map or {}

    def top(self, limit: int = 25, asset_class=AssetClass.CRYPTO) -> List[Quote]:
        return list(self._top_rows[:limit])

    def quotes(self, symbols: List[str]) -> List[Quote]:
        return [self._quote_map[s.upper()] for s in symbols if s.upper() in self._quote_map]


def _q(symbol: str, price: float, source: str) -> Quote:
    return Quote(symbol=symbol, name=symbol, price=price, source=source)


def test_top_merges_when_first_provider_returns_short_list():
    """A blockchain.com-shaped provider (always 1 result) must not truncate top(5)."""
    hub = DataHub()
    btc_only = _FakeProvider("blockchain.com", top_rows=[_q("BTC", 65000.0, "blockchain.com")])
    full = _FakeProvider("coinpaprika", top_rows=[
        _q("BTC", 65001.0, "coinpaprika"), _q("ETH", 3200.0, "coinpaprika"),
        _q("SOL", 150.0, "coinpaprika"), _q("BNB", 550.0, "coinpaprika"),
        _q("XRP", 0.6, "coinpaprika"),
    ])
    hub.providers_for = lambda ac, cap: [btc_only, full]  # priority order: btc_only first

    rows = hub.top(limit=5)
    assert len(rows) == 5
    symbols = [r.symbol for r in rows]
    assert symbols[0] == "BTC" and rows[0].source == "blockchain.com"  # first-provider wins the dupe
    assert set(symbols) == {"BTC", "ETH", "SOL", "BNB", "XRP"}         # no duplicates, all filled


def test_top_stops_once_limit_reached_no_needless_calls():
    hub = DataHub()
    full = _FakeProvider("coinpaprika", top_rows=[_q(s, 1.0, "coinpaprika") for s in ["BTC", "ETH", "SOL"]])
    never_called = _FakeProvider("should-not-be-reached")

    def boom(*a, **k):
        raise AssertionError("provider called after limit already satisfied")
    never_called.top = boom

    hub.providers_for = lambda ac, cap: [full, never_called]
    rows = hub.top(limit=2)
    assert len(rows) == 2 and [r.symbol for r in rows] == ["BTC", "ETH"]


def test_quotes_merges_across_providers_for_missing_symbols():
    """A provider covering only BTC must not stop DataHub from finding ETH elsewhere."""
    hub = DataHub()
    btc_only = _FakeProvider("blockchain.com", quote_map={"BTC": _q("BTC", 65000.0, "blockchain.com")})
    full = _FakeProvider("coinpaprika", quote_map={
        "BTC": _q("BTC", 65001.0, "coinpaprika"), "ETH": _q("ETH", 3200.0, "coinpaprika"),
    })
    hub.providers_for = lambda ac, cap: [btc_only, full]

    rows = hub.quotes(["BTC", "ETH"])
    syms = {r.symbol for r in rows}
    assert syms == {"BTC", "ETH"}
    btc_row = next(r for r in rows if r.symbol == "BTC")
    assert btc_row.source == "blockchain.com"  # first provider's answer preserved, not overwritten


def test_quotes_returns_empty_when_no_provider_has_symbol():
    hub = DataHub()
    hub.providers_for = lambda ac, cap: [_FakeProvider("empty")]
    assert hub.quotes(["ZZZ"]) == []
