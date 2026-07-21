"""
Tests for finscope.core.providers (registry) and finscope.core.datahub.DataHub.

These tests rely ONLY on the bundled DemoProvider for actual data assertions.
The sandbox network is blocked, so any live provider (e.g. CoinPaprika) is
expected to fail fast with a ProviderError that DataHub swallows and routes
past -- exactly the graceful-degradation behavior documented in datahub.py.
Tests do not assume network is unreachable though: they assert the *shape* of
the result (Quote/Series with a source drawn from a registered provider name)
rather than pinning a hardcoded source string.

Covers:
  * DemoProvider is registered in REGISTRY
  * DataHub().top(limit=5) returns 5 Quotes with source in the set of
    registered provider names (e.g. {"demo", "coinpaprika", ...})
  * DataHub().history("BTC") returns a Series with >= 10 candles
  * unknown symbol history returns None gracefully (no exception)

Works both under pytest (functions named test_*) and standalone:
    python3 finscope/tests/test_providers.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from finscope.core.contracts import AssetClass, Quote, Series
from finscope.core.datahub import DataHub
from finscope.core.providers import REGISTRY, providers_for
from finscope.core.providers.demo import DemoProvider


# ---- registry -------------------------------------------------------------------

def test_demo_provider_registered_in_registry():
    assert "demo" in REGISTRY
    assert isinstance(REGISTRY["demo"], DemoProvider)


def test_demo_provider_capability_flags():
    demo = REGISTRY["demo"]
    assert demo.can_quote is True
    assert demo.can_top is True
    assert demo.can_history is True
    assert demo.can_search is True
    assert AssetClass.CRYPTO in demo.asset_classes


def test_demo_provider_is_lowest_priority_fallback():
    # demo.priority == 900 and DataHub tries lower-priority (smaller number)
    # providers first, so demo is only reached once live providers fail.
    demo = REGISTRY["demo"]
    assert demo.priority == 900
    others = [p for p in REGISTRY.values() if p.name != "demo"]
    for p in others:
        assert p.priority < demo.priority


def test_demo_provider_healthy():
    assert REGISTRY["demo"].healthy() is True


def test_registry_has_more_than_one_provider():
    # demo + at least one live provider (e.g. coinpaprika) are both registered,
    # even though the live one may fail at call time in a network-blocked sandbox.
    assert len(REGISTRY) >= 2


def test_providers_for_crypto_top_includes_demo_and_is_priority_sorted():
    ranked = providers_for(AssetClass.CRYPTO, "top")
    names = [p.name for p in ranked]
    assert "demo" in names
    # sorted ascending by priority -> demo (900) must be last
    assert names[-1] == "demo"
    priorities = [p.priority for p in ranked]
    assert priorities == sorted(priorities)


# ---- DataHub.top() ----------------------------------------------------------------

def test_datahub_top_returns_5_quotes_with_known_source():
    hub = DataHub()
    known_sources = set(hub.providers.keys())
    quotes = hub.top(AssetClass.CRYPTO, limit=5)
    assert len(quotes) == 5
    for q in quotes:
        assert isinstance(q, Quote)
        assert q.source in known_sources
        assert q.price >= 0.0
        assert q.symbol  # non-empty


def test_datahub_top_respects_limit():
    hub = DataHub()
    quotes = hub.top(AssetClass.CRYPTO, limit=3)
    assert len(quotes) == 3


# ---- DataHub.history() -------------------------------------------------------------

def test_datahub_history_btc_has_at_least_10_candles():
    hub = DataHub()
    series = hub.history("BTC")
    assert series is not None
    assert isinstance(series, Series)
    assert series.symbol == "BTC"
    assert len(series) >= 10


def test_datahub_history_unknown_symbol_returns_none_gracefully():
    hub = DataHub()
    # Must not raise -- DataHub catches ProviderError internally and degrades
    # to None when no provider (including the demo fallback) has data.
    series = hub.history("NOT_A_REAL_SYMBOL_ZZZ")
    assert series is None


def test_datahub_quote_unknown_symbol_returns_none_gracefully():
    hub = DataHub()
    q = hub.quote("NOT_A_REAL_SYMBOL_ZZZ")
    assert q is None


def test_datahub_quotes_known_symbols_returns_matching_rows():
    hub = DataHub()
    rows = hub.quotes(["BTC", "ETH"])
    syms = {q.symbol for q in rows}
    assert "BTC" in syms
    assert "ETH" in syms


# ---- DataHub.health() ----------------------------------------------------------------

def test_datahub_health_reports_bool_per_provider():
    hub = DataHub()
    health = hub.health()
    assert set(health.keys()) == set(hub.providers.keys())
    for name, ok in health.items():
        assert isinstance(ok, bool)
    # the bundled demo provider must always be healthy (it's a local seed file)
    assert health["demo"] is True


# ---- test runner (standalone mode) --------------------------------------------------------

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
