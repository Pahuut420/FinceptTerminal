"""
Provider package. Importing this package auto-discovers and registers every
sibling provider module, so adding `myprovider.py` here is all it takes to make
it available to the DataHub — no central edit needed.
"""
from __future__ import annotations

import importlib
import pkgutil
import warnings

from finscope.core.providers.base import (  # noqa: F401  (re-exported)
    Provider, register, REGISTRY, all_providers, get_provider, providers_for,
)


def _autodiscover() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if mod.name in ("base",) or mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{mod.name}")
        except Exception as exc:  # pragma: no cover - a bad provider must not kill the app
            warnings.warn(f"finscope: provider '{mod.name}' failed to load: {exc}")


_autodiscover()

__all__ = ["Provider", "register", "REGISTRY", "all_providers", "get_provider", "providers_for"]
