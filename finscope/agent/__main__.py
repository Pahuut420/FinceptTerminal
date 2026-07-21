"""
Agent CLI: drive FinScope with a JSON request from the shell or another agent.

    python -m finscope.agent '{"op":"market_snapshot","limit":5}'
    python -m finscope.agent '{"op":"rank_strategies","symbol":"BTC"}'
    python -m finscope.agent --spec        # print the tool schema
    python -m finscope.agent --describe     # capability manifest

Prints JSON to stdout — the natural interface for an LLM agent or a pipe.
"""
from __future__ import annotations

import json
import sys

from finscope.agent.dashboard import DashboardAgent, TOOL_SPEC


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--spec":
        print(json.dumps(TOOL_SPEC, indent=2))
        return 0
    agent = DashboardAgent()
    if argv[0] == "--describe":
        print(json.dumps(agent.describe(), indent=2, default=str))
        return 0
    try:
        request = json.loads(argv[0])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"invalid JSON request: {e}"}))
        return 1
    print(json.dumps(agent.run(request), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
