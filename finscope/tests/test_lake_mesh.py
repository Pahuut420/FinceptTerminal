"""Tests for the data lake + mesh bus. Runnable via pytest or standalone."""
from __future__ import annotations

from finscope.core.contracts import AssetClass, Series
from finscope.mesh.bus import _match, get_bus
from finscope.mesh.schema import Subjects
from finscope.mesh.pipeline import run_demo


def test_nats_matcher():
    assert _match(">", "finscope.quote.BTC")
    assert _match("finscope.signal.>", "finscope.signal.native")
    assert _match("finscope.*.BTC", "finscope.quote.BTC")
    assert not _match("finscope.*.BTC", "finscope.quote.ETH")
    assert not _match("finscope.risk.>", "finscope.signal.native")


def test_bus_pub_sub():
    bus = get_bus()
    seen = []
    bus.subscribe("finscope.signal.>", lambda s, p: seen.append((s, p)))
    bus.publish(Subjects.signal("native"), {"symbol": "BTC", "size": 0.5})
    assert seen and seen[0][1]["symbol"] == "BTC"
    assert len(bus.history(">")) == 1


def test_pipeline_produces_capped_fills():
    out = run_demo(symbols=["BTC", "ETH"], strategies=["momentum", "breakout"])
    assert out["messages"] > 0
    assert out["by_type"].get("signal", 0) > 0
    # every fill must respect the 1% position cap enforced by the RiskGate
    for f in out["fills"]:
        assert abs(f["size"]) <= 0.01 + 1e-9


def test_lake_roundtrip():
    from finscope.lake.store import lake_available
    if not lake_available():
        return  # optional dependency absent — skip cleanly
    from finscope.lake.store import DuckDBLake
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        lake = DuckDBLake(lake_dir=d)
        s = Series.from_closes("TEST", [10, 11, 12, 11, 13], asset_class=AssetClass.CRYPTO)
        n = lake.write_series(s)
        assert n == 5
        back = lake.read_series("TEST", asset_class=AssetClass.CRYPTO)
        assert back is not None and len(back) == 5
        assert abs(back.closes()[-1] - 13) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e}")
    raise SystemExit(1 if failed else 0)
