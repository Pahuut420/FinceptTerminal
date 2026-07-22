"""
Deterministic robustness study — use the $0 research machine many times, honestly.

Re-running the same search on the same inputs converges to the same champion; it
tells you nothing new. The value of free, deterministic re-runs is *robustness*:
score every strategy across many INDEPENDENT synthetic realizations in several
market REGIMES, and rank by consistency — not by a single lucky backtest (which is
the multiple-testing trap that manufactures fake alpha).

Four regimes, each a fat-tailed (Student-t) price process:
  * null      — driftless GBM. Structure = none. A trustworthy strategy scores ~0
                here; a positive score means residual overfitting/lookahead.
  * trend_up  — positive drift. Structure = persistent up-trend.
  * trend_dn  — negative drift. Structure = persistent down-trend (short side).
  * mean_rev  — Ornstein-Uhlenbeck log-price. Structure = reversion to a level.

For each strategy we backtest across K seeds in every regime with the $0 native
engine (no lookahead, tx-cost, out-of-sample Sharpe) and report the per-regime mean
OOS Sharpe + the fraction of seeds positive. Headline `structured_mean` averages the
three structured regimes and subtracts a penalty for any positive null score, so a
robust generalist ranks above a one-regime specialist ranks above a noise-fitter.

Everything here is pure numpy + the native engine: $0, no LLM, fully reproducible
(same seeds → same table). Run it as many times as you like; vary `--base-seed`
for an independent replication.

    python -m finscope.research.robustness                       # 20 seeds, 4 regimes
    python -m finscope.research.robustness --seeds 40 --save     # tighter estimate
    python -m finscope.research.robustness --top 15 --base-seed 5000
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from finscope.core.contracts import Series, OHLCV
from finscope.engines import get_engine
from finscope.engines import strategies as core
from finscope.research.optimizer import DEFAULT_GRIDS, _build_any

_DF = 4          # Student-t degrees of freedom -> fat tails
_SD = 0.03       # per-bar vol anchor (matches the lake's synthetic model)
_START_TS = 1_600_000_000.0
_NULL_PENALTY = 1.0   # weight on max(0, null_mean) subtracted from structured_mean


def _series_from_prices(prices: np.ndarray, symbol: str) -> Series:
    candles = [
        OHLCV(ts=_START_TS + j * 86400,
              open=float(prices[j - 1] if j > 0 else prices[0]),
              high=float(prices[j]), low=float(prices[j]),
              close=float(prices[j]), volume=1.0)
        for j in range(len(prices))
    ]
    return Series(symbol=symbol, candles=candles)


def _gbm(seed: int, bars: int, drift: float) -> np.ndarray:
    """Driftless/trending fat-tailed GBM (drift per bar in log space)."""
    rng = np.random.default_rng(seed)
    innov = rng.standard_t(_DF, size=bars) * _SD / np.sqrt(_DF / (_DF - 2))
    return 100.0 * np.exp(np.cumsum(drift + innov))


def _ou(seed: int, bars: int, kappa: float = 0.06, sigma: float = 0.035) -> np.ndarray:
    """Mean-reverting Ornstein-Uhlenbeck log-price around a constant level."""
    rng = np.random.default_rng(seed)
    x = 0.0
    xs = np.empty(bars)
    for t in range(bars):
        x = x * (1.0 - kappa) + sigma * rng.standard_t(_DF) / np.sqrt(_DF / (_DF - 2))
        xs[t] = x
    return 100.0 * np.exp(xs)


# regime name -> price-path generator(seed, bars)
REGIMES: Dict[str, Callable[[int, int], np.ndarray]] = {
    "null": lambda s, b: _gbm(s, b, 0.0),
    "trend_up": lambda s, b: _gbm(s, b, 0.0006),
    "trend_dn": lambda s, b: _gbm(s, b, -0.0006),
    "mean_rev": lambda s, b: _ou(s, b),
}
STRUCTURED = ("trend_up", "trend_dn", "mean_rev")


@dataclass
class RobustRow:
    strategy: str
    per_regime_mean: Dict[str, float]
    per_regime_pospct: Dict[str, float]
    structured_mean: float
    structured_pospct: float
    worst_structured: float
    n_seeds: int
    n_bars: int

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "structured_mean": round(self.structured_mean, 4),
            "structured_pospct": round(self.structured_pospct, 4),
            "worst_structured": round(self.worst_structured, 4),
            "per_regime_mean": {k: round(v, 4) for k, v in self.per_regime_mean.items()},
            "per_regime_pospct": {k: round(v, 4) for k, v in self.per_regime_pospct.items()},
            "n_seeds": self.n_seeds, "n_bars": self.n_bars,
        }


def _all_strategy_names() -> List[str]:
    """Every searchable strategy: core library + optimiser DEFAULT_GRIDS keys."""
    names = list(dict.fromkeys(list(core.STRATEGIES) + list(DEFAULT_GRIDS)))
    return names


def _build(name: str):
    if name in core.STRATEGIES:
        return core.build(name)
    return _build_any(name, {})


def run_study(seeds: int = 20, bars: int = 730, base_seed: int = 1000,
              names: List[str] | None = None) -> List[RobustRow]:
    eng = get_engine("native")
    names = names or _all_strategy_names()
    # pre-generate every path once (shared across strategies -> fair + fast)
    paths: Dict[str, List[Series]] = {}
    for reg, gen in REGIMES.items():
        paths[reg] = [_series_from_prices(gen(base_seed + i, bars), f"{reg[:3].upper()}{i}")
                      for i in range(seeds)]

    rows: List[RobustRow] = []
    for name in names:
        try:
            strat = _build(name)
        except Exception:
            continue
        reg_mean, reg_pos, struct_vals = {}, {}, []
        ok = True
        for reg in REGIMES:
            vals = []
            for ser in paths[reg]:
                try:
                    vals.append(eng.backtest(_build(name), ser).oos_sharpe)
                except Exception:
                    pass
            if not vals:
                ok = False
                break
            a = np.asarray(vals)
            reg_mean[reg] = float(a.mean())
            reg_pos[reg] = float((a > 0).mean())
            if reg in STRUCTURED:
                struct_vals.append(a)
        if not ok:
            continue
        struct_all = np.concatenate(struct_vals)
        smean = float(np.mean([reg_mean[r] for r in STRUCTURED]))
        null_pen = _NULL_PENALTY * max(0.0, reg_mean["null"])
        rows.append(RobustRow(
            strategy=name, per_regime_mean=reg_mean, per_regime_pospct=reg_pos,
            structured_mean=smean - null_pen,
            structured_pospct=float((struct_all > 0).mean()),
            worst_structured=float(min(reg_mean[r] for r in STRUCTURED)),
            n_seeds=seeds, n_bars=bars))
    rows.sort(key=lambda r: (r.structured_mean, r.structured_pospct), reverse=True)
    return rows


def _fmt_table(rows: List[RobustRow], top: int) -> str:
    head = ("strategy", "struct", "%pos", "null", "up", "dn", "mrev", "worst")
    lines = ["%-30s%8s%7s%8s%8s%8s%8s%8s" % head]
    for r in rows[:top]:
        m = r.per_regime_mean
        lines.append("%-30s%8.3f%7.2f%8.3f%8.3f%8.3f%8.3f%8.3f" % (
            r.strategy[:30], r.structured_mean, r.structured_pospct, m["null"],
            m["trend_up"], m["trend_dn"], m["mean_rev"], r.worst_structured))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic strategy robustness study ($0).")
    ap.add_argument("--seeds", type=int, default=20, help="independent paths per regime")
    ap.add_argument("--bars", type=int, default=730)
    ap.add_argument("--base-seed", type=int, default=1000, help="vary for an independent replication")
    ap.add_argument("--top", type=int, default=20, help="rows to print")
    ap.add_argument("--save", action="store_true", help="persist JSON + TSV under research/results/")
    ap.add_argument("--json", action="store_true", help="print full JSON instead of the table")
    args = ap.parse_args(argv)

    rows = run_study(seeds=args.seeds, bars=args.bars, base_seed=args.base_seed)
    payload = {
        "regimes": list(REGIMES), "structured": list(STRUCTURED),
        "seeds": args.seeds, "bars": args.bars, "base_seed": args.base_seed,
        "n_strategies": len(rows),
        "note": ("structured_mean = mean(trend_up,trend_dn,mean_rev) OOS Sharpe minus "
                 "penalty*max(0,null). Robust structure-detection, not investable alpha: "
                 "single-model synthetic, wide cross-seed spread. Null should sit ~0."),
        "leaderboard": [r.to_dict() for r in rows],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("DETERMINISTIC ROBUSTNESS STUDY — %d strategies x %d seeds x %d regimes (%d bars), $0"
              % (len(rows), args.seeds, len(REGIMES), args.bars))
        print("rank by structured_mean (null must be ~0; a positive null = overfitting flag)\n")
        print(_fmt_table(rows, args.top))
        print("\nnull sanity: mean|null| = %.3f across all strategies (want small)"
              % float(np.mean([abs(r.per_regime_mean["null"]) for r in rows])))

    if args.save:
        outdir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "robustness.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        with open(os.path.join(outdir, "robustness.tsv"), "w", encoding="utf-8") as fh:
            fh.write("strategy\tstructured_mean\tstructured_pospct\tnull\ttrend_up\ttrend_dn\tmean_rev\tworst\n")
            for r in rows:
                m = r.per_regime_mean
                fh.write("%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % (
                    r.strategy, r.structured_mean, r.structured_pospct, m["null"],
                    m["trend_up"], m["trend_dn"], m["mean_rev"], r.worst_structured))
        print("\nsaved: research/results/robustness.{json,tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
