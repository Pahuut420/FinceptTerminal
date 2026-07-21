"""
RepoGraph — thin accessor over the graphify knowledge graph.

Two layers:
  * the committed, distilled `coordination_map.json` (always available, tiny):
    fleet totals, god nodes, community clusters, and the finscope↔fleet coverage
    map. This is what the swarm coordinated against.
  * the raw graphify graph + CLI (optional, not committed — 49MB): if graphify
    is installed and a graph.json exists, `explain()` / `path()` shell out to it
    for live structural queries.

All methods degrade gracefully when graphify or the raw graph is absent.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import List, Optional

_HERE = os.path.dirname(__file__)
_MAP_PATH = os.path.join(_HERE, "coordination_map.json")
# raw fleet graph (gitignored); present only after `graphify update fincept-qt/scripts`
_FLEET_GRAPH = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)),
    "fincept-qt", "scripts", "graphify-out", "graph.json",
)


def load_coordination_map() -> dict:
    """Return the distilled, committed coordination map (empty dict if missing)."""
    try:
        with open(_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


class RepoGraph:
    def __init__(self, graph_path: Optional[str] = None):
        self.map = load_coordination_map()
        self.graph_path = graph_path or _FLEET_GRAPH

    # ---- distilled map (always available) ----
    def god_nodes(self) -> List[dict]:
        return self.map.get("connector_fleet", {}).get("god_nodes", [])

    def communities(self) -> List[str]:
        return self.map.get("connector_fleet", {}).get("top_communities", [])

    def coverage(self) -> dict:
        return self.map.get("coverage", {})

    def fleet_stats(self) -> dict:
        return self.map.get("connector_fleet", {})

    def summary(self) -> str:
        fleet = self.fleet_stats()
        fs = self.map.get("finscope_module", {})
        return (f"fleet: {fleet.get('nodes')} nodes / {fleet.get('edges')} edges / "
                f"{fleet.get('communities')} communities · "
                f"finscope: {fs.get('nodes')} nodes / {fs.get('edges')} edges")

    # ---- live graphify queries (optional) ----
    def _graphify(self) -> Optional[str]:
        return shutil.which("graphify")

    def available(self) -> bool:
        return bool(self._graphify()) and os.path.exists(self.graph_path)

    def explain(self, node: str) -> str:
        """Plain-language explanation of a node and its neighbours via graphify."""
        if not self.available():
            return "graphify graph not available (run: graphify update fincept-qt/scripts)"
        try:
            r = subprocess.run([self._graphify(), "explain", node, "--graph", self.graph_path],
                               capture_output=True, text=True, timeout=30)
            return (r.stdout or r.stderr).strip()
        except (subprocess.SubprocessError, OSError) as e:
            return f"graphify explain failed: {e}"

    def path(self, a: str, b: str) -> str:
        """Shortest dependency path between two nodes via graphify."""
        if not self.available():
            return "graphify graph not available (run: graphify update fincept-qt/scripts)"
        try:
            r = subprocess.run([self._graphify(), "path", a, b, "--graph", self.graph_path],
                               capture_output=True, text=True, timeout=30)
            return (r.stdout or r.stderr).strip()
        except (subprocess.SubprocessError, OSError) as e:
            return f"graphify path failed: {e}"
