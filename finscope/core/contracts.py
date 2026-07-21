"""
Interface contracts for FinScope.

Every layer agrees on these types. Providers *produce* Quote / OHLCV / Series;
the analytics engine *consumes* Series; the UI *renders* Quote / Series and the
results of analytics. Keeping these frozen lets the data, analytics, and UI
layers be developed independently (and by parallel agents) without integration
drift.

Pure stdlib + dataclasses. No third-party imports here on purpose — this module
must import cleanly in any environment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple


class AssetClass(str, Enum):
    """Coarse asset classification used for provider routing."""
    CRYPTO = "crypto"
    ONCHAIN = "onchain"
    EQUITY = "equity"
    FX = "fx"
    COMMODITY = "commodity"
    MACRO = "macro"
    RATE = "rate"
    INDEX = "index"
    OTHER = "other"


class ProviderError(RuntimeError):
    """Raised by a Provider when a data fetch fails. The DataHub catches these
    and degrades gracefully (cache / empty) rather than crashing the terminal."""


@dataclass(frozen=True)
class Quote:
    """A point-in-time snapshot of an instrument."""
    symbol: str
    name: str
    price: float
    source: str
    asset_class: AssetClass = AssetClass.OTHER
    change_24h_pct: Optional[float] = None
    volume_24h: Optional[float] = None
    market_cap: Optional[float] = None
    rank: Optional[int] = None
    ts: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)

    @property
    def is_up(self) -> bool:
        return (self.change_24h_pct or 0.0) >= 0


@dataclass(frozen=True)
class OHLCV:
    """A single candle. `ts` is a POSIX timestamp (seconds)."""
    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class Series:
    """An ordered time series of candles for one symbol.

    The canonical container the analytics engine consumes. Helpers return plain
    Python lists so numpy is optional at this layer.
    """
    symbol: str
    candles: List[OHLCV] = field(default_factory=list)
    source: str = ""
    asset_class: AssetClass = AssetClass.OTHER

    def __len__(self) -> int:
        return len(self.candles)

    def closes(self) -> List[float]:
        return [c.close for c in self.candles]

    def timestamps(self) -> List[float]:
        return [c.ts for c in self.candles]

    def returns(self) -> List[float]:
        """Simple period-over-period returns of the close series."""
        cl = self.closes()
        out: List[float] = []
        for i in range(1, len(cl)):
            prev = cl[i - 1]
            out.append((cl[i] - prev) / prev if prev else 0.0)
        return out

    def last(self) -> Optional[OHLCV]:
        return self.candles[-1] if self.candles else None

    @classmethod
    def from_closes(
        cls,
        symbol: str,
        closes: Sequence[float],
        start_ts: Optional[float] = None,
        step: float = 86400.0,
        source: str = "",
        asset_class: AssetClass = AssetClass.OTHER,
    ) -> "Series":
        """Build a Series from a bare close vector (o=h=l=c). Handy for tests
        and for providers that only expose closing prices."""
        start = start_ts if start_ts is not None else time.time() - step * len(closes)
        candles = [
            OHLCV(ts=start + i * step, open=c, high=c, low=c, close=c)
            for i, c in enumerate(closes)
        ]
        return cls(symbol=symbol, candles=candles, source=source, asset_class=asset_class)


# ---- analytics result contracts -------------------------------------------------

@dataclass(frozen=True)
class RiskMetrics:
    """Output of the analytics engine for a single series."""
    symbol: str
    n: int
    mean_return: float
    daily_vol: float
    annualized_vol: float
    annualized_return: float
    sharpe: float
    max_drawdown: float
    var_95: float
    engine: str = "numpy"  # "numpy" or "wolfram"


@dataclass(frozen=True)
class CorrelationMatrix:
    """Pairwise correlation across a set of symbols."""
    symbols: Tuple[str, ...]
    matrix: Tuple[Tuple[float, ...], ...]
    engine: str = "numpy"

    def pair(self, a: str, b: str) -> Optional[float]:
        try:
            i, j = self.symbols.index(a), self.symbols.index(b)
        except ValueError:
            return None
        return self.matrix[i][j]
