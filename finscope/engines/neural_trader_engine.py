"""
Neural Trader adapter (ruvnet / ruvector).

Neural Trader lives in the ruvector repo as Rust crates
(`crates/neural-trader-core`, `-strategies`, `-replay`, `-wasm`) and ships an
`npx neural-trader` / MCP surface (see the `neural-trader-nas` skill: Kelly / LSTM
/ DRL / sentiment, the OMEGA dual-engine = Freqtrade + Neural Trader). It is not a
Python lib, so this adapter detects the crate/binary and describes how to run it;
signals are consumed on the user's machine (or via its MCP server).
"""
from __future__ import annotations

import os
from typing import Optional

from finscope.engines.base import Engine, register_engine
from finscope.engines import _detect


@register_engine
class NeuralTraderEngine(Engine):
    name = "neural-trader"
    kind = "agent"
    description = "Neural Trader (ruvnet) — Rust/WASM engine: Kelly/LSTM/DRL/sentiment. Detect + describe."

    def _crate(self) -> Optional[str]:
        repo = _detect.find_repo("ruvector", "RuVector")
        if repo:
            c = os.path.join(repo, "crates", "neural-trader-core")
            if os.path.isdir(c):
                return c
        return None

    def available(self) -> bool:
        return bool(self._crate() or _detect.which("neural-trader"))

    def describe(self) -> dict:
        d = super().describe()
        d.update({
            "crate": self._crate(),
            "cli": _detect.which("neural-trader"),
            "npx": "npx neural-trader",
            "mcp_skill": "neural-trader-nas (persistent MCP server; OMEGA dual-engine)",
            "enable": "run `npx neural-trader` or build the ruvector crates with cargo; "
                      "consume signals via its MCP server or CLI.",
        })
        return d

    def run(self, request: dict) -> dict:
        op = request.get("op", "describe")
        if op == "describe":
            return self.describe()
        return {"error": "neural-trader is driven via its own CLI/MCP; see describe().enable"}
