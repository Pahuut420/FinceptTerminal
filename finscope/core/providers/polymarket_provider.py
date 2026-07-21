"""
Polymarket provider — Prediction market top listings via Gamma API.

No API key required. Maps prediction market questions to Quote objects where
the price is the Yes probability (0-100) and volume is the 24h market volume.
"""
from __future__ import annotations

from typing import List, Optional

import requests

from finscope.core.contracts import AssetClass, Quote, ProviderError
from finscope.core.providers.base import Provider, register

_BASE = "https://gamma-api.polymarket.com"
_UA = {"User-Agent": "FinScope/0.1 (+finceptterminal)"}


def _get(path: str, params: Optional[dict] = None, timeout: float = 20.0) -> dict:
    try:
        r = requests.get(f"{_BASE}/{path}", params=params, headers=_UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise ProviderError(f"polymarket: {e}") from e


@register
class PolymarketProvider(Provider):
    name = "polymarket"
    asset_classes = (AssetClass.OTHER,)
    requires_key = False
    priority = 50
    can_top = True
    can_search = True

    @staticmethod
    def _to_quote(market: dict) -> Quote:
        """Convert a Polymarket market dict to a Quote.

        Expected structure (from gamma-api):
        {
          "id": "market-id",
          "slug": "market-slug",
          "question": "Will X happen?",
          "outcomes": [{"label": "Yes", "probability": 0.65}, {"label": "No", ...}],
          "volume24h": 150000.0,
          "volume": 1000000.0,
          "liquidity": {...}
        }
        """
        question = market.get("question", "?")
        slug = market.get("slug", "market")
        outcomes = market.get("outcomes", [])

        # Extract Yes probability (first outcome is typically Yes)
        yes_prob = 50.0
        if outcomes and isinstance(outcomes, list) and len(outcomes) > 0:
            yes_outcome = next((o for o in outcomes if o.get("label", "").lower() == "yes"), None)
            if yes_outcome:
                yes_prob = (yes_outcome.get("probability", 0.5) or 0.5) * 100
            elif outcomes[0]:
                yes_prob = (outcomes[0].get("probability", 0.5) or 0.5) * 100

        # Volume: prefer 24h, fall back to total
        volume = market.get("volume24h") or market.get("volume") or 0.0

        # Create a short symbol from the slug (max 10 chars, uppercase)
        symbol = slug[:10].upper() if slug else "POLY"

        # Truncate question to 40 chars for name
        name = question[:40] + "..." if len(question) > 40 else question

        return Quote(
            symbol=symbol,
            name=name,
            price=yes_prob,
            source="polymarket",
            asset_class=AssetClass.OTHER,
            change_24h_pct=None,
            volume_24h=volume,
            extra={"question": question, "slug": slug},
        )

    def top(self, limit: int = 25, asset_class: AssetClass = AssetClass.OTHER) -> List[Quote]:
        """Fetch top active prediction markets by volume."""
        params = {
            "closed": "false",
            "limit": max(limit, 100),  # fetch extra to have room to filter
            "order": "volume",
            "ascending": "false",
        }
        data = _get("markets", params=params)

        if not isinstance(data, list):
            raise ProviderError(f"polymarket: unexpected markets payload: {type(data)}")

        quotes: List[Quote] = []
        for market in data[:limit]:
            try:
                q = self._to_quote(market)
                quotes.append(q)
            except (KeyError, TypeError, ValueError):
                continue

        if not quotes:
            raise ProviderError("polymarket: no markets found")
        return quotes

    def search(self, query: str, limit: int = 20) -> List[Quote]:
        """Search markets by question text."""
        params = {
            "closed": "false",
            "limit": max(limit, 50),
        }
        data = _get("markets", params=params)

        if not isinstance(data, list):
            raise ProviderError(f"polymarket: unexpected markets payload: {type(data)}")

        q_lower = query.lower()
        matches: List[Quote] = []
        for market in data:
            try:
                question = market.get("question", "").lower()
                if q_lower in question or q_lower in market.get("slug", "").lower():
                    matches.append(self._to_quote(market))
                    if len(matches) >= limit:
                        break
            except (KeyError, TypeError, ValueError):
                continue

        if not matches:
            raise ProviderError(f"polymarket: no markets found for {query!r}")
        return matches

    def healthy(self) -> bool:
        try:
            _get("markets", params={"limit": 1}, timeout=8.0)
            return True
        except ProviderError:
            return False
