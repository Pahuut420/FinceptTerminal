"""
FinScope onchain — the DAA/QuDAG execution bridge + MRAP autonomy loop.

Wraps `ruvnet/daa` (Decentralized Autonomous Agents, Rust) and QuDAG
(quantum-resistant networking) behind a Python bridge that is PAPER-mode by
default and refuses live on-chain orders without explicit multi-gate operator
opt-in. The MRAPLoop (Monitor→Reason→Act→Reflect→Adapt) is the deterministic
autonomy cycle from DAA, driving strategies through the treasury + fund controls.

Nothing here moves real funds unless an operator has built the DAA/QuDAG stack,
funded a wallet, and set the live gate — by design.
"""
from finscope.onchain.daa_bridge import DAABridge
from finscope.onchain.mrap import MRAPLoop

__all__ = ["DAABridge", "MRAPLoop"]
