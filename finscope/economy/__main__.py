"""
Budget Governor CLI — deterministic spend meter/gate/router.

    python -m finscope.economy status
    python -m finscope.economy recommend --units 3        # cheapest tier that fits
    python -m finscope.economy gate --amount 2.0          # would this spend fit?
    python -m finscope.economy spend --amount 0.5 --tier haiku --label "breadth agent"
    python -m finscope.economy reset
    (--cap sets/updates the period cap; state persists in ~/.finscope/economy/ledger.json)
"""
from __future__ import annotations

import argparse
import json
import sys

from finscope.economy.governor import BudgetGovernor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="finscope.economy")
    ap.add_argument("cmd", choices=["status", "recommend", "gate", "spend", "reset"])
    ap.add_argument("--cap", type=float, default=100.0)
    ap.add_argument("--amount", type=float, default=0.0)
    ap.add_argument("--units", type=float, default=1.0)
    ap.add_argument("--tier", default="local")
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    g = BudgetGovernor(cap=args.cap)
    if args.cmd == "status":
        out = g.status()
    elif args.cmd == "recommend":
        out = g.recommend_tier(units=args.units)
    elif args.cmd == "gate":
        out = g.gate(args.amount, args.label)
    elif args.cmd == "spend":
        out = g.spend(args.amount, args.tier, args.label)
    elif args.cmd == "reset":
        g.reset(); out = {"reset": True, **g.status()}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
