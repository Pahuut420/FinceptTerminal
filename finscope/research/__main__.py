"""
Deterministic strategy research CLI (AutoResearch loop, no LLM).

    python -m finscope.research                      # optimise all strategies, leaderboard
    python -m finscope.research --symbols BTC ETH    # choose symbols
    python -m finscope.research --strategy momentum --genetic
    python -m finscope.research --json               # machine-readable

The search is exhaustive grid (or seeded genetic); every candidate scored by the
$0 native backtester on out-of-sample Sharpe. Reproducible and free.
"""
from __future__ import annotations

import argparse
import json
import sys

from finscope.research import optimize_all, grid_search, genetic_search
from finscope.research.optimizer import StrategyOptimizer, DEFAULT_GRIDS


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="finscope.research")
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--strategy", help="optimise a single strategy")
    ap.add_argument("--genetic", action="store_true", help="seeded genetic search")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--save", action="store_true",
                    help="persist the leaderboard to finscope/research/results/")
    args = ap.parse_args(argv)

    if args.strategy:
        opt = StrategyOptimizer(symbols=args.symbols, days=args.days)
        grid = DEFAULT_GRIDS.get(args.strategy, {})
        cands = (genetic_search if args.genetic else grid_search)(args.strategy, grid, opt=opt) \
            if args.genetic else grid_search(args.strategy, grid, opt=opt)
        rows = [c.to_dict() for c in cands[: args.top]]
        out = {"strategy": args.strategy, "symbols": args.symbols, "candidates": rows}
    else:
        out = optimize_all(symbols=args.symbols, days=args.days, top_k=args.top)

    if args.save:
        import os
        d = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "leaderboard.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved -> {os.path.relpath(os.path.join(d, 'leaderboard.json'))}")

    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print(f"FinScope AutoResearch — symbols={args.symbols} days={args.days}  (deterministic, $0)")
    top = out.get("global_top") or out.get("candidates", [])
    print(f"{'rank':>4}  {'strategy':<24}{'params':<28}{'score':>8}{'oosSharpe':>10}{'ret%':>8}{'maxDD%':>8}")
    for i, c in enumerate(top, 1):
        print(f"{i:>4}  {c['strategy']:<24}{json.dumps(c['params']):<28}"
              f"{c['score']:>8.2f}{c['mean_oos_sharpe']:>10.2f}"
              f"{c['mean_return']*100:>8.2f}{c['mean_max_drawdown']*100:>8.2f}")
    if out.get("note"):
        print(f"\nNOTE: {out['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
