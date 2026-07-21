"""Auto-import engine adapter modules so they self-register via @register_engine."""
from __future__ import annotations

import importlib
import pkgutil
import warnings


def autodiscover() -> None:
    import finscope.engines as pkg
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_") or mod.name == "base":
            continue
        try:
            importlib.import_module(f"finscope.engines.{mod.name}")
        except Exception as exc:  # a bad adapter must not break the whole layer
            warnings.warn(f"finscope: engine '{mod.name}' failed to load: {exc}")
