"""
Ruliology strategies — trading signals derived from Stephen Wolfram's computational
universe (elementary cellular automata, rule classes, computational irreducibility,
the ruliad). Each `rul_*.py` module contributes a `STRATEGIES` dict; they are
auto-discovered and aggregated into `RULIOLOGY_STRATEGIES`.

Thesis: encode the recent return-sign sequence as a symbolic/CA state and read a
signal from its rule-dynamics — markets range from class-1 (trivially trending)
through class-3 (irreducibly random — don't trade) to class-4 (complex,
edge-of-chaos — where structure is exploitable). All strategies are causal.
"""
from __future__ import annotations

import importlib
import pkgutil
import warnings

RULIOLOGY_STRATEGIES = {}


def _discover() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        try:
            m = importlib.import_module(f"{__name__}.{mod.name}")
            RULIOLOGY_STRATEGIES.update(getattr(m, "STRATEGIES", {}))
        except Exception as exc:  # a bad rule-module must not break the rest
            warnings.warn(f"ruliology: module '{mod.name}' failed to load: {exc}")


_discover()

__all__ = ["RULIOLOGY_STRATEGIES"]
