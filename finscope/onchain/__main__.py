"""
MRAP autonomy-loop demo (paper-mode).

    python -m finscope.onchain                 # MRAP loop on BTC
    python -m finscope.onchain SYN_BTC         # on a synthetic-filled symbol (scale test)
    python -m finscope.onchain --describe      # DAA/QuDAG bridge status + live gate

Runs the Monitor→Reason→Act→Reflect→Adapt walk-forward loop with treasury +
fund controls + kill-switch. Never touches real funds.
"""
from __future__ import annotations

import json
import sys

from finscope.onchain.mrap import MRAPLoop
from finscope.onchain.daa_bridge import DAABridge


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--describe":
        print(json.dumps(DAABridge().describe(), indent=2, default=str))
        return 0
    symbol = argv[0] if argv else "BTC"
    print(json.dumps(MRAPLoop().run(symbol=symbol), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
