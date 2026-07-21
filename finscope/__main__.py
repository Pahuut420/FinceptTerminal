"""
FinScope entrypoint.

    python -m finscope                 # interactive Bloomberg-style REPL
    python -m finscope --live          # auto-refreshing crypto monitor
    python -m finscope --once "CRYPTO 10"   # run one function, print, exit
    python -m finscope --demo          # render a representative snapshot (headless)
"""
from __future__ import annotations

import argparse
import sys

from finscope.ui.app import FinScopeApp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="finscope",
                                 description="Bloomberg-style terminal on the FinceptTerminal data fleet")
    ap.add_argument("--once", metavar="CMD", help="run a single function code and exit")
    ap.add_argument("--demo", action="store_true", help="render a headless snapshot and exit")
    ap.add_argument("--live", action="store_true", help="auto-refreshing crypto market monitor")
    ap.add_argument("--interval", type=float, default=10.0, help="live refresh seconds (default 10)")
    args = ap.parse_args(argv)

    app = FinScopeApp()
    if args.demo:
        app.snapshot()
        return 0
    if args.once:
        app.run_command(args.once)
        return 0
    if args.live:
        app.live(interval=args.interval)
        return 0
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
