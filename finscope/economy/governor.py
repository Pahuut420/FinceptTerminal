"""
BudgetGovernor — deterministic spend meter + gate + tier router.

The daa-economy `TokenManager` concept applied to *any* spendable resource
(LLM dollars, tokens, agent-calls, or trading capital). It:
  * tracks spend events against a period cap (rolling window),
  * gates a proposed spend (approve / deny with reason) — never approves over cap,
  * routes a task to the cheapest tier whose budget headroom allows it,

so "use the budget wisely" is enforced by code. Persists to a JSON ledger so the
cap survives across sessions/processes. No LLM, no network.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Tier:
    """A spend tier, cheapest first. cost is per-unit (e.g. $/call or $/1k-tok)."""
    name: str
    cost: float
    note: str = ""


# Default model-routing tiers (mirrors the session discipline: $0 local < haiku < fable).
DEFAULT_TIERS = [
    Tier("local", 0.0, "$0 deterministic code (graphify/backtest/optimizer)"),
    Tier("haiku", 0.02, "cheap breadth work"),
    Tier("sonnet", 0.20, "moderate reasoning"),
    Tier("fable", 0.80, "small hard expertise tasks only"),
]


@dataclass
class SpendEvent:
    amount: float
    tier: str
    label: str
    ts: float = field(default_factory=time.time)
    def to_dict(self) -> dict: return asdict(self)


class BudgetGovernor:
    def __init__(self, cap: float = 100.0, period_seconds: float = 7 * 86400,
                 tiers: Optional[List[Tier]] = None, ledger_path: Optional[str] = None,
                 unit: str = "usd"):
        self.cap = cap
        self.period = period_seconds
        self.unit = unit
        self.tiers = sorted(tiers or DEFAULT_TIERS, key=lambda t: t.cost)
        self.ledger_path = ledger_path or os.path.join(
            os.environ.get("FINSCOPE_ECONOMY_DIR", os.path.expanduser("~/.finscope/economy")),
            "ledger.json")
        self._events: List[SpendEvent] = []
        self._load()

    # ---- persistence ----
    def _load(self) -> None:
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._events = [SpendEvent(**e) for e in data.get("events", [])]
        except (OSError, ValueError, TypeError):
            self._events = []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.ledger_path), exist_ok=True)
        with open(self.ledger_path, "w", encoding="utf-8") as f:
            json.dump({"cap": self.cap, "unit": self.unit,
                       "events": [e.to_dict() for e in self._events]}, f, indent=2)

    # ---- accounting (rolling window) ----
    def _now(self) -> float:
        return time.time()

    def _in_window(self) -> List[SpendEvent]:
        cutoff = self._now() - self.period
        return [e for e in self._events if e.ts >= cutoff]

    def spent(self) -> float:
        return round(sum(e.amount for e in self._in_window()), 6)

    def remaining(self) -> float:
        return round(max(0.0, self.cap - self.spent()), 6)

    def utilization(self) -> float:
        return round(self.spent() / self.cap, 4) if self.cap else 0.0

    # ---- gate + record ----
    def can_afford(self, amount: float) -> bool:
        return amount <= self.remaining() + 1e-9

    def gate(self, estimated: float, label: str = "") -> dict:
        ok = self.can_afford(estimated)
        return {"approved": ok, "estimated": estimated, "remaining": self.remaining(),
                "utilization": self.utilization(),
                "reason": "ok" if ok else f"would exceed cap {self.cap} {self.unit}",
                "label": label}

    def record(self, amount: float, tier: str = "local", label: str = "") -> SpendEvent:
        ev = SpendEvent(amount=amount, tier=tier, label=label, ts=self._now())
        self._events.append(ev)
        self._save()
        return ev

    def spend(self, amount: float, tier: str = "local", label: str = "") -> dict:
        """Gate then record if approved. Returns the gate verdict."""
        v = self.gate(amount, label)
        if v["approved"]:
            self.record(amount, tier, label)
        return v

    # ---- tier routing ----
    def recommend_tier(self, units: float = 1.0, min_tier: str = "local") -> dict:
        """Cheapest tier whose cost*units fits the remaining budget, at or above
        min_tier. Deterministic — this is the 'cheap models first' rule as code."""
        order = {t.name: i for i, t in enumerate(self.tiers)}
        floor = order.get(min_tier, 0)
        rem = self.remaining()
        for t in self.tiers[floor:]:
            if t.cost * units <= rem + 1e-9:
                return {"tier": t.name, "unit_cost": t.cost, "est": round(t.cost * units, 6),
                        "remaining": rem, "note": t.note}
        return {"tier": "local", "unit_cost": 0.0, "est": 0.0, "remaining": rem,
                "note": "budget exhausted — deterministic/$0 path only"}

    def status(self) -> dict:
        return {"cap": self.cap, "unit": self.unit, "spent": self.spent(),
                "remaining": self.remaining(), "utilization": self.utilization(),
                "events_in_window": len(self._in_window()),
                "tiers": [t.name for t in self.tiers]}

    def reset(self) -> None:
        self._events = []
        self._save()
