"""
Reference mesh pipeline — the multi-engine quant flow, wired over the SignalBus.

Topology (star/hub, per the OMEGA trading discipline):

    providers ──quote──▶  bus
       engines ──signal──▶ bus ──▶ RiskGate ──risk──▶ bus ──▶ Execution ──fill──▶ bus

  * Engines publish target-weight SignalMsgs from their strategies.
  * RiskGate applies fractional-Kelly sizing + a hard per-position cap and emits a
    RiskVerdict (approve/clamp/reject). Never sizes above the cap.
  * Execution turns approved verdicts into paper FillMsgs.

Runs in-proc today (deterministic, $0); set NATS_URL to run the exact same wiring
across processes/hosts on a real NATS mesh. This is the substrate for the
"ultimate quant machine" — zero-latency fan-out with a real risk gate before any
order, and a kill-switch hook (`RiskGate.halted`).
"""
from __future__ import annotations

from typing import List, Optional

from finscope.core.datahub import DataHub
from finscope.core.contracts import AssetClass
from finscope.engines import get_engine
from finscope.engines import strategies as strat_lib
from finscope.mesh.bus import SignalBus, get_bus
from finscope.mesh.schema import Subjects, QuoteMsg, SignalMsg, RiskVerdict, FillMsg


class RiskGate:
    """Fractional-Kelly sizing + hard cap + kill-switch. Deterministic, no LLM."""
    def __init__(self, bus: SignalBus, kelly_fraction: float = 0.25,
                 max_position: float = 0.01):
        self.bus = bus
        self.kelly_fraction = kelly_fraction   # OMEGA: quarter-Kelly
        self.max_position = max_position        # OMEGA: max 1% risk per position
        self.halted = False
        bus.subscribe(f"{Subjects.SIGNAL}.>", self._on_signal)

    def _on_signal(self, subject: str, payload: dict) -> None:
        if self.halted:
            self.bus.publish(Subjects.risk(payload["symbol"]),
                             RiskVerdict(payload["symbol"], False, 0.0, "halted").to_dict())
            return
        raw = float(payload.get("size", 0.0))
        conf = float(payload.get("confidence", 1.0))
        sized = raw * self.kelly_fraction * conf
        clamped = max(-self.max_position, min(self.max_position, sized))
        approved = abs(clamped) > 1e-9
        reason = "ok" if approved else "below-threshold"
        if abs(sized) > self.max_position:
            reason = f"clamped to cap {self.max_position}"
        self.bus.publish(Subjects.risk(payload["symbol"]),
                         RiskVerdict(payload["symbol"], approved, round(clamped, 6),
                                     reason).to_dict())


class Execution:
    """Paper execution — turns approved risk verdicts into fills."""
    def __init__(self, bus: SignalBus, hub: DataHub):
        self.bus = bus
        self.hub = hub
        bus.subscribe(f"{Subjects.RISK}.>", self._on_verdict)

    def _on_verdict(self, subject: str, payload: dict) -> None:
        if not payload.get("approved"):
            return
        sym = payload["symbol"]
        q = self.hub.quote(sym)
        price = q.price if q else 0.0
        action = "long" if payload["size"] > 0 else ("short" if payload["size"] < 0 else "flat")
        self.bus.publish(Subjects.fill("paper"),
                         FillMsg(sym, "paper", action, payload["size"], price, paper=True).to_dict())


def run_demo(symbols: Optional[List[str]] = None,
             strategies: Optional[List[str]] = None,
             bus: Optional[SignalBus] = None) -> dict:
    """Wire the full mesh in-proc and run one pass. Returns the message trace."""
    bus = bus or get_bus()
    hub = DataHub()
    symbols = symbols or ["BTC", "ETH", "SOL"]
    strategies = strategies or ["skew_tilt_tsmom", "momentum", "breakout"]
    gate = RiskGate(bus)
    Execution(bus, hub)
    native = get_engine("native")

    for sym in symbols:
        q = hub.quote(sym)
        if q:
            bus.publish(Subjects.quote(sym), QuoteMsg(sym, q.price, source=q.source).to_dict())
        series = hub.history(sym, days=365)
        if not series or len(series) < 3:
            continue
        for sname in strategies:
            try:
                strat = strat_lib.build(sname) if sname in strat_lib.STRATEGIES else None
                if strat is None:
                    from finscope.research.optimizer import _build_any
                    strat = _build_any(sname, {})
                sig = native.signal(strat, series)
                bus.publish(Subjects.signal("native"),
                            SignalMsg(sym, "native", sname, sig.action, sig.size).to_dict())
            except Exception:
                continue

    trace = bus.history(">")
    counts = {}
    for subj, _ in trace:
        head = subj.split(".")[1] if "." in subj else subj
        counts[head] = counts.get(head, 0) + 1
    return {"backend": bus.backend, "messages": len(trace), "by_type": counts,
            "fills": [p for s, p in trace if s.startswith(Subjects.FILL)],
            "risk_verdicts": [p for s, p in trace if s.startswith(Subjects.RISK)][:10]}
