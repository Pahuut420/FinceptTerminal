"""
Data-lake CLI.

    python -m finscope.lake bootstrap                 # fill from bundled real seed
    python -m finscope.lake ingest --symbols BTC ETH --days 365   # pull via providers
    python -m finscope.lake stats
    python -m finscope.lake list
"""
from __future__ import annotations

import argparse
import json
import sys

from finscope.lake.store import DuckDBLake, lake_available


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="finscope.lake")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("bootstrap")
    ing = sub.add_parser("ingest")
    ing.add_argument("--symbols", nargs="+", required=True)
    ing.add_argument("--days", type=int, default=365)
    sub.add_parser("stats")
    sub.add_parser("list")
    args = ap.parse_args(argv)

    if not lake_available():
        print(json.dumps({"error": "lake unavailable — pip install duckdb pyarrow"}))
        return 1
    lake = DuckDBLake()

    if args.cmd == "bootstrap":
        from finscope.lake.ingest import bootstrap_from_seed
        print(json.dumps(bootstrap_from_seed(lake), indent=2))
    elif args.cmd == "ingest":
        from finscope.lake.ingest import ingest_from_hub
        print(json.dumps(ingest_from_hub(args.symbols, days=args.days, lake=lake), indent=2))
    elif args.cmd == "stats":
        print(json.dumps(lake.stats(), indent=2))
    elif args.cmd == "list":
        print(json.dumps(lake.symbols(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
