"""
Mesh message schemas + subject names — the bus contract.

JSON-serialisable dataclasses so messages cross in-proc handlers or NATS subjects
identically. Subjects follow a hierarchical `finscope.<domain>.<detail>` scheme so
NATS wildcard subscriptions (`finscope.signal.>`) work naturally.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field


class Subjects:
    QUOTE = "finscope.quote"          # finscope.quote.<SYMBOL>
    SIGNAL = "finscope.signal"        # finscope.signal.<ENGINE>
    RISK = "finscope.risk"            # finscope.risk.<SYMBOL>
    FILL = "finscope.fill"            # finscope.fill.<ENGINE>

    @staticmethod
    def quote(sym: str) -> str: return f"{Subjects.QUOTE}.{sym.upper()}"
    @staticmethod
    def signal(engine: str) -> str: return f"{Subjects.SIGNAL}.{engine}"
    @staticmethod
    def risk(sym: str) -> str: return f"{Subjects.RISK}.{sym.upper()}"
    @staticmethod
    def fill(engine: str) -> str: return f"{Subjects.FILL}.{engine}"


@dataclass
class QuoteMsg:
    symbol: str
    price: float
    ts: float = field(default_factory=time.time)
    source: str = ""
    def to_dict(self) -> dict: return asdict(self)


@dataclass
class SignalMsg:
    symbol: str
    engine: str
    strategy: str
    action: str        # long | short | flat
    size: float        # target weight [-1, 1]
    ts: float = field(default_factory=time.time)
    confidence: float = 1.0
    def to_dict(self) -> dict: return asdict(self)


@dataclass
class RiskVerdict:
    symbol: str
    approved: bool
    size: float         # size after Kelly/limit clamp
    reason: str = ""
    ts: float = field(default_factory=time.time)
    def to_dict(self) -> dict: return asdict(self)


@dataclass
class FillMsg:
    symbol: str
    engine: str
    action: str
    size: float
    price: float
    ts: float = field(default_factory=time.time)
    paper: bool = True
    def to_dict(self) -> dict: return asdict(self)
