"""
Mesh demo CLI — run the full provider→signal→risk→fill flow over the bus.

    python -m finscope.mesh                       # in-proc demo
    NATS_URL=nats://localhost:4222 python -m finscope.mesh   # over real NATS

Prints the message trace summary as JSON.
"""
from __future__ import annotations

import json
import sys

from finscope.mesh.pipeline import run_demo


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    syms = argv or ["BTC", "ETH", "SOL"]
    print(json.dumps(run_demo(symbols=syms), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
