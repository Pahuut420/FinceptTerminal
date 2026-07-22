"""
Omega-crypto ↔ FinScope bridge CLI.

    python -m finscope.integrations omega               # native score + MRAP on BTC
    python -m finscope.integrations omega --symbol SYN_BTC
    python -m finscope.integrations omega --params /path/to/strategy_params.json

Runs NEXUS-OMEGA's long_short_scoring strategy through FinScope's backtester and
MRAP (treasury + fund controls + kill-switch).
"""
from __future__ import annotations

import argparse
import json
import sys

from finscope.integrations.omega_crypto import (
    load_omega_params, omega_native_score, run_omega_mrap, find_omega_crypto)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="finscope.integrations")
    ap.add_argument("target", choices=["omega"])
    ap.add_argument("--symbol", default="BTC")
    ap.add_argument("--params", default=None, help="path to strategy_params.json dir")
    args = ap.parse_args(argv)
    params = load_omega_params(args.params)
    out = {
        "omega_crypto_dir": find_omega_crypto(),
        "native_score": omega_native_score(args.symbol, params),
        "mrap": run_omega_mrap(args.symbol, params),
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
