"""
FundControls — hard risk limits + kill-switch + the live-execution gate.

The fund-control discipline daa-economy implies for an autonomous treasury agent,
made explicit and deterministic. Every proposed order passes `check()`; live
execution additionally requires `allow_live()` — which is false unless an operator
has set FINSCOPE_ONCHAIN_LIVE=1 AND the controls are un-halted AND the order is
within every limit. Paper mode needs none of that.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FundControls:
    max_position: float = 0.01        # max fraction of capital per position (OMEGA: 1%)
    max_daily_loss: float = 0.03      # halt if realised daily loss exceeds this
    max_drawdown: float = 0.20        # halt if equity drawdown exceeds this
    per_symbol_cap: float = 0.05      # max aggregate exposure to one symbol
    whitelist: Optional[List[str]] = None   # None = allow all
    halted: bool = False
    _daily_loss: float = 0.0
    _drawdown: float = 0.0

    def kill(self, reason: str = "manual") -> dict:
        self.halted = True
        return {"halted": True, "reason": reason}

    def resume(self) -> None:
        self.halted = False

    def update_pnl(self, daily_loss: float, drawdown: float) -> None:
        self._daily_loss = daily_loss
        self._drawdown = drawdown
        if daily_loss >= self.max_daily_loss or drawdown >= self.max_drawdown:
            self.kill(f"limit breach daily_loss={daily_loss:.3f} dd={drawdown:.3f}")

    def check(self, symbol: str, size: float, symbol_exposure: float = 0.0) -> dict:
        """Validate a proposed order (size = target weight, fraction of capital)."""
        if self.halted:
            return {"approved": False, "reason": "halted (kill-switch)"}
        if self.whitelist is not None and symbol.upper() not in {s.upper() for s in self.whitelist}:
            return {"approved": False, "reason": f"{symbol} not in whitelist"}
        if abs(size) > self.max_position + 1e-9:
            return {"approved": False, "clamp": self.max_position * (1 if size > 0 else -1),
                    "reason": f"exceeds max_position {self.max_position}"}
        if abs(symbol_exposure + size) > self.per_symbol_cap + 1e-9:
            return {"approved": False, "reason": f"exceeds per_symbol_cap {self.per_symbol_cap}"}
        return {"approved": True, "reason": "ok", "size": size}

    def allow_live(self) -> dict:
        """The live-execution gate. Multiple conditions must ALL hold."""
        env_on = os.environ.get("FINSCOPE_ONCHAIN_LIVE") == "1"
        ok = env_on and not self.halted
        reasons = []
        if not env_on:
            reasons.append("FINSCOPE_ONCHAIN_LIVE != 1")
        if self.halted:
            reasons.append("controls halted")
        return {"live_allowed": ok, "blockers": reasons or None,
                "note": "paper mode is always available; live requires operator opt-in + healthy controls"}
