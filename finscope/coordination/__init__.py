"""
FinScope coordination layer.

Wraps the graphify (safishamsi/graphify) knowledge graph that maps the
FinceptTerminal connector fleet (319 scripts, ~31k nodes) and the finscope
module itself. Used as the shared structural map for the build swarm and,
at runtime, to answer "what depends on X" / "shortest path A→B" questions.
"""
from finscope.coordination.repo_graph import RepoGraph, load_coordination_map

__all__ = ["RepoGraph", "load_coordination_map"]
