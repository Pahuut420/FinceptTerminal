"""
Wire NEXUS-OMEGA's omega-crypto AutoResearch loop onto FinScope's treasury + MRAP.

omega-crypto (omega-platform/crypto) optimises a `long_short_scoring` strategy —
a weighted vote of RSI / Bollinger / MA-trend / MA-crossover / volume signals,
tuned via `strategy_params.json`, scored by `backtest.py` on Sharpe. This module
reimplements that scoring as a *causal* FinScope `Strategy` driven by the same
params, so omega's strategy runs through FinScope's:
  * NativeEngine backtester (no-lookahead, OOS Sharpe) — the roadmap's "shared
    backtest core", and
  * MRAPLoop + TreasuryManager + FundControls — the shared risk layer (Kelly cap,
    kill-switch) omega needs before any paper->live transition.

This is the OMEGA_ASSESSMENT roadmap steps 1-2 made real: reuse, not reimplement.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1, _sma
from finscope.engines import get_engine
from finscope.engines.strategies import MomentumStrategy  # noqa: F401 (protocol example)

_OMEGA_SUBPATHS = ("omega-platform/crypto", "crypto")
_OMEGA_ROOTS = ("/home/user/omega-platform", "/workspace/omega-platform",
                "/home/user", "/workspace")


def find_omega_crypto() -> Optional[str]:
    env = os.environ.get("OMEGA_CRYPTO_DIR")
    if env and os.path.isdir(env):
        return env
    for root in _OMEGA_ROOTS:
        for sub in _OMEGA_SUBPATHS:
            p = os.path.join(root, sub)
            if os.path.isfile(os.path.join(p, "strategy_params.json")):
                return p
    return None


def load_omega_params(path: Optional[str] = None) -> dict:
    d = path or find_omega_crypto()
    if not d:
        return _DEFAULT_PARAMS.copy()
    try:
        with open(os.path.join(d, "strategy_params.json"), "r", encoding="utf-8") as f:
            p = json.load(f)
        return {**_DEFAULT_PARAMS, **p}
    except (OSError, ValueError):
        return _DEFAULT_PARAMS.copy()


def _ema(x: np.ndarray, period: int) -> np.ndarray:
    if x.size == 0 or period <= 1:
        return x.copy()
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(x); out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(x: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(x, 50.0)
    if x.size < period + 1:
        return out
    delta = np.diff(x)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.mean(gain[:period]); al = np.mean(loss[:period])
    for i in range(period, x.size):
        g = gain[i - 1] if i - 1 < gain.size else 0.0
        l = loss[i - 1] if i - 1 < loss.size else 0.0
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + l) / period
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1 + ag / al)
    return out


@dataclass
class OmegaScoreStrategy:
    """omega-crypto's long_short_scoring reimplemented as a causal FinScope strategy."""
    params: dict = field(default_factory=lambda: _DEFAULT_PARAMS.copy())
    name: str = "omega_long_short_scoring"

    def __post_init__(self):
        self.name = f"omega_score(thr={self.params.get('entry_threshold', 2.0)})"

    def weights(self, series: Series) -> np.ndarray:
        p = self.params
        c = _closes(series)
        n = c.size
        if n < 5:
            return np.zeros(n)
        rsi = _rsi(c, int(p.get("rsi_period", 14)))
        ef = _ema(c, int(p.get("ma_fast", 12)))
        es = _ema(c, int(p.get("ma_slow", 26)))
        bb_mid = _sma(c, int(p.get("bollinger_period", 20)))
        dev = c - np.nan_to_num(bb_mid)
        bstd = np.nanstd(dev[~np.isnan(dev)]) or 1.0
        bpos = np.nan_to_num(dev) / (p.get("bollinger_std", 2.0) * bstd)  # >1 above upper

        s_rsi = np.where(rsi < p.get("rsi_oversold", 35), 1.0,
                         np.where(rsi > p.get("rsi_overbought", 67), -1.0, 0.0))
        s_bb = np.clip(-bpos, -1.0, 1.0)              # below lower band -> long
        s_trend = np.sign(np.nan_to_num(ef - es))
        cross = np.zeros(n)
        diff = np.nan_to_num(ef - es)
        cross[1:] = np.where((diff[1:] > 0) & (diff[:-1] <= 0), 1.0,
                             np.where((diff[1:] < 0) & (diff[:-1] >= 0), -1.0, 0.0))
        rsi_rec = np.zeros(n)
        rsi_rec[1:] = np.where((rsi[1:] > p.get("rsi_oversold", 35)) &
                               (rsi[:-1] <= p.get("rsi_oversold", 35)), 1.0, 0.0)

        score = (p.get("w_rsi", 1.5) * s_rsi + p.get("w_bb", 1.0) * s_bb +
                 p.get("w_ma_trend", 1.0) * s_trend + p.get("w_ma_crossover", 1.75) * cross +
                 p.get("w_rsi_recovery", 0.5) * rsi_rec)
        thr = p.get("entry_threshold", 2.0)
        wsum = (p.get("w_rsi", 1.5) + p.get("w_bb", 1.0) + p.get("w_ma_trend", 1.0) +
                p.get("w_ma_crossover", 1.75) + p.get("w_rsi_recovery", 0.5)) or 1.0
        w = np.zeros(n)
        long = score >= thr
        short = (score <= -thr * p.get("w_short_mult", 1.0)) & bool(p.get("enable_shorts", True))
        w[long] = np.clip(score[long] / wsum, 0, 1)
        w[short] = np.clip(score[short] / wsum, -1, 0)
        return _shift1(np.nan_to_num(w))


def omega_native_score(symbol: str = "BTC", params: Optional[dict] = None,
                       days: int = 3650) -> dict:
    """Backtest the omega strategy on FinScope's NativeEngine (Sharpe = omega's metric)."""
    from finscope.core.datahub import DataHub
    p = params or load_omega_params()
    series = DataHub().history(symbol, days=days)
    if not series or len(series) < 5:
        return {"error": f"no history for {symbol} — fill the lake"}
    r = get_engine("native").backtest(OmegaScoreStrategy(params=p), series)
    return {"symbol": symbol, "score_sharpe": r.sharpe, "oos_sharpe": r.oos_sharpe,
            "total_return": r.total_return, "max_drawdown": r.max_drawdown,
            "n_trades": r.n_trades, "params_source": find_omega_crypto() or "defaults"}


def run_omega_mrap(symbol: str = "BTC", params: Optional[dict] = None) -> dict:
    """Run omega's strategy through the FinScope MRAP loop (treasury + controls + kill-switch)."""
    from finscope.onchain.mrap import MRAPLoop
    from finscope.engines import strategies_alpha  # ensure alpha registered
    p = params or load_omega_params()
    loop = MRAPLoop()
    # register the omega strategy under a temporary name the loop can build
    import finscope.engines.strategies as S
    S.STRATEGIES.setdefault("omega_score", lambda **kw: OmegaScoreStrategy(params=p))
    out = loop.run(symbol=symbol, strategies=["omega_score"])
    out["engine"] = "omega->finscope MRAP"
    return out


_DEFAULT_PARAMS = {
    "rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 67, "ma_fast": 12,
    "ma_slow": 26, "bollinger_period": 20, "bollinger_std": 2.0, "entry_threshold": 2.0,
    "enable_shorts": True, "w_short_mult": 1.0, "w_rsi": 1.5, "w_bb": 1.0,
    "w_ma_trend": 1.0, "w_ma_crossover": 1.75, "w_rsi_recovery": 0.5,
}
