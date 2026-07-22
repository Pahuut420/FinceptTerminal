"""
DAABridge — bridge to ruvnet/daa (Rust) + QuDAG, paper-first and live-gated.

Detects the DAA/QuDAG toolchain (daa-cli / daa-prime-cli / qudag binaries or a
sibling clone) without importing anything. Order flow:

    submit_order(...) -> paper fill                       (default, always allowed)
    submit_order(..., live=True) -> requires ALL of:
        * FundControls.allow_live() true (env gate + un-halted)
        * a detected daa-cli
        * FundControls.check() approves size
      otherwise it REFUSES and returns why. It never silently goes live.

Quantum-safe comms: when live, orders would be signed/encrypted via QuDAG
(ML-DSA-87 / ML-KEM-1024) — documented as the transport; this bridge shells out
to daa-cli which owns the crypto. In paper mode no keys/network are touched.
"""
from __future__ import annotations

import time
from typing import Optional

from finscope.engines import _detect
from finscope.economy.controls import FundControls


class DAABridge:
    name = "daa"

    def __init__(self, controls: Optional[FundControls] = None):
        self.controls = controls or FundControls()

    # ---- detection ----
    def _cli(self) -> Optional[str]:
        for b in ("daa-cli", "daa", "daa-prime-cli"):
            p = _detect.which(b)
            if p:
                return p
        return None

    def _repo(self) -> Optional[str]:
        return _detect.find_repo("daa")

    def _qudag(self) -> Optional[str]:
        return _detect.which("qudag")

    def available(self) -> bool:
        return bool(self._cli() or self._repo())

    def describe(self) -> dict:
        live = self.controls.allow_live()
        return {
            "name": self.name,
            "daa_cli": self._cli(),
            "daa_repo": self._repo(),
            "qudag_cli": self._qudag(),
            "available": self.available(),
            "live_gate": live,
            "crypto": "QuDAG ML-DSA-87 signatures / ML-KEM-1024 encryption (via daa-cli when live)",
            "enable_live": "build ruvnet/daa (cargo), fund a wallet, set FINSCOPE_ONCHAIN_LIVE=1, "
                           "keep FundControls healthy; then submit_order(live=True).",
        }

    # ---- execution ----
    def submit_order(self, symbol: str, size: float, price: float = 0.0,
                     symbol_exposure: float = 0.0, live: bool = False) -> dict:
        chk = self.controls.check(symbol, size, symbol_exposure=symbol_exposure)
        if not chk["approved"]:
            return {"status": "rejected", "reason": chk["reason"], "symbol": symbol}
        if not live:
            return {"status": "filled", "mode": "paper", "symbol": symbol,
                    "size": round(size, 6), "price": price, "ts": time.time()}
        # live path — refuse unless every gate is satisfied
        gate = self.controls.allow_live()
        if not gate["live_allowed"]:
            return {"status": "refused", "mode": "live", "symbol": symbol,
                    "blockers": gate["blockers"]}
        if not self._cli():
            return {"status": "refused", "mode": "live", "symbol": symbol,
                    "blockers": ["daa-cli not installed — build ruvnet/daa first"]}
        # Real submission would shell out to daa-cli here (QuDAG-signed). Not executed
        # in this environment: no wallet, no built toolchain, no operator confirmation.
        return {"status": "refused", "mode": "live", "symbol": symbol,
                "blockers": ["live submission is intentionally not automated — "
                             "run daa-cli manually with operator confirmation"]}
