"""
FinScope economy — deterministic economic controls (the daa-economy pattern in Python).

One pattern, two resources:
  * BudgetGovernor  — meters/gates *spend* (LLM $/tokens, agent calls) against a
    period cap and routes work to the cheapest capable tier. Answers the "use the
    budget wisely" requirement with code, not vibes.
  * TreasuryManager — allocates *trading capital* across strategies/venues with the
    same governor underneath, plus FundControls (hard limits + kill-switch).

Both are deterministic and stdlib-only. On-chain execution goes through DAABridge,
which is PAPER-mode by default and refuses live orders without explicit multi-gate
enablement — the safe analog of daa-economy + QuDAG for the OMEGA live phase.
"""
from finscope.economy.governor import BudgetGovernor, Tier, SpendEvent
from finscope.economy.controls import FundControls
from finscope.economy.treasury import TreasuryManager

__all__ = ["BudgetGovernor", "Tier", "SpendEvent", "FundControls", "TreasuryManager"]
