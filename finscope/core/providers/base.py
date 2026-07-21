"""
Provider ABC + a self-registering registry.

A Provider adapts one upstream data source to FinScope's contracts. Providers
declare which AssetClasses and capabilities they support; the DataHub routes
requests accordingly. New providers only need to:

    from finscope.core.providers.base import Provider, register

    @register
    class MyProvider(Provider):
        name = "my-source"
        asset_classes = (AssetClass.CRYPTO,)
        ...

Dropping a new module in this package is enough — `providers/__init__.py`
auto-imports every sibling module, triggering @register at import time. This is
what lets parallel agents each add a provider file without touching shared code.
"""
from __future__ import annotations

from typing import Dict, List, Type

from finscope.core.contracts import AssetClass, Quote, Series, ProviderError

# Public registry: provider-name -> instance (providers are stateless-ish singletons).
REGISTRY: Dict[str, "Provider"] = {}


class Provider:
    """Base provider. Subclasses override the capabilities they support and set
    class attributes `name` and `asset_classes`. Unsupported methods raise
    ProviderError so the DataHub can route past them."""

    name: str = "base"
    asset_classes: tuple = ()
    requires_key: bool = False
    # lower priority is tried first by the DataHub (live=50, demo/fallback=900)
    priority: int = 50
    # capability flags — DataHub reads these to route
    can_quote: bool = False
    can_top: bool = False
    can_history: bool = False
    can_search: bool = False

    # ---- capability methods (override the ones you support) ----
    def quotes(self, symbols: List[str]) -> List[Quote]:
        raise ProviderError(f"{self.name}: quotes() not supported")

    def top(self, limit: int = 25, asset_class: AssetClass = AssetClass.CRYPTO) -> List[Quote]:
        raise ProviderError(f"{self.name}: top() not supported")

    def history(self, symbol: str, days: int = 90) -> Series:
        raise ProviderError(f"{self.name}: history() not supported")

    def search(self, query: str, limit: int = 20) -> List[Quote]:
        raise ProviderError(f"{self.name}: search() not supported")

    def healthy(self) -> bool:
        """Cheap liveness check. Default True; override with a light ping."""
        return True

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        caps = [c for c in ("quote", "top", "history", "search")
                if getattr(self, f"can_{c}")]
        return f"<Provider {self.name} {'/'.join(a.value for a in self.asset_classes)} caps={caps}>"


def register(cls: Type[Provider]) -> Type[Provider]:
    """Class decorator: instantiate and register a provider by its `name`."""
    inst = cls()
    if not inst.name or inst.name == "base":
        raise ValueError(f"Provider {cls!r} must set a unique `name`")
    REGISTRY[inst.name] = inst
    return cls


def all_providers() -> List[Provider]:
    return list(REGISTRY.values())


def get_provider(name: str) -> Provider:
    if name not in REGISTRY:
        raise KeyError(f"unknown provider: {name}")
    return REGISTRY[name]


def providers_for(asset_class: AssetClass, capability: str) -> List[Provider]:
    """All registered providers that cover `asset_class` and support `capability`
    (one of quote/top/history/search)."""
    flag = f"can_{capability}"
    matches = [
        p for p in REGISTRY.values()
        if asset_class in p.asset_classes and getattr(p, flag, False)
    ]
    return sorted(matches, key=lambda p: p.priority)
