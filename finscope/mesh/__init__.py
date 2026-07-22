"""
FinScope mesh — the zero-latency signal bus for the multi-engine quant machine.

A `SignalBus` carries typed messages (quotes, signals, fills, risk verdicts)
between data providers, strategy engines, a risk gate, and execution — a star/hub
topology (per the OMEGA trading discipline). Two backends:

  * InProc (default): in-process pub/sub — works everywhere, used for tests and
    single-process runs.
  * NATS (opt-in): NATS core / JetStream when `nats-py` is installed and
    `NATS_URL` is set — the production zero-latency mesh across processes/hosts.

The backend is chosen at runtime; message schemas are backend-agnostic so code
written against the bus runs unchanged whether it's in-proc or on NATS.
"""
from finscope.mesh.schema import QuoteMsg, SignalMsg, FillMsg, RiskVerdict, Subjects
from finscope.mesh.bus import SignalBus, get_bus

__all__ = ["QuoteMsg", "SignalMsg", "FillMsg", "RiskVerdict", "Subjects",
           "SignalBus", "get_bus"]
