"""
Shared detection helpers for external engine adapters.

External engines (freqtrade, nautilus_trader, neural-trader, TradingAgents) may be
pip-installed, present as a sibling repo clone, or absent. Adapters call these to
decide `available()` without importing heavy deps at module load — so FinScope
always imports cleanly even when nothing is installed.
"""
from __future__ import annotations

import importlib.util
import os
from typing import List, Optional

# common sibling-clone locations
_SEARCH_ROOTS = [
    "/workspace", os.path.expanduser("~"), os.getcwd(),
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
]


def module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def find_repo(*names: str) -> Optional[str]:
    """Return the first existing sibling clone dir among the given repo names."""
    for root in _SEARCH_ROOTS:
        for n in names:
            p = os.path.join(root, n)
            if os.path.isdir(p):
                return p
    return None


def which(binary: str) -> Optional[str]:
    import shutil
    return shutil.which(binary)
