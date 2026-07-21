"""
Blockchain.com provider — BTC-only quote source (no key).

blockchain.info's public `/ticker` endpoint only reports Bitcoin pricing (no
altcoins, no 24h % change, no volume figures), so this provider is
deliberately small: one symbol, one endpoint. It exists as an independent,
no-key cross-check on BTC/USD pricing rather than a general crypto provider,
which is also why it straddles both CRYPTO and ONCHAIN asset classes.
"""
from __future__ import annotations

from typing import List, Optional

import requests

from finscope.core.contracts import AssetClass, Quote, ProviderError
from finscope.core.providers.base import Provider, register

_BASE = "https://blockchain.info"
_UA = {"User-Agent": "FinScope/0.1 (+finceptterminal)"}


def _get(path: str, params: Optional[dict] = None, timeout: float = 20.0):
    try:
        r = requests.get(f"{_BASE}/{path}", params=params, headers=_UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise ProviderError(f"blockchain.com: {e}") from e
    except ValueError as e:  # bad/non-JSON body
        raise ProviderError(f"blockchain.com: invalid JSON response: {e}") from e


@register
class BlockchainComProvider(Provider):
    name = "blockchain.com"
    asset_classes = (AssetClass.CRYPTO, AssetClass.ONCHAIN)
    requires_key = False
    priority = 50
    can_quote = True
    can_top = True

    def _btc_quote(self) -> Quote:
        data = _get("ticker")
        usd = data.get("USD") if isinstance(data, dict) else None
        if not usd:
            raise ProviderError("blockchain.com: no USD ticker in response")
        try:
            price = float(usd.get("last") or 0.0)
        except (TypeError, ValueError):
            raise ProviderError("blockchain.com: malformed USD ticker payload")
        return Quote(
            symbol="BTC",
            name="Bitcoin",
            price=price,
            source="blockchain.com",
            asset_class=AssetClass.CRYPTO,
            extra={"buy": usd.get("buy"), "sell": usd.get("sell"), "15m": usd.get("15m")},
        )

    def quotes(self, symbols: List[str]) -> List[Quote]:
        want = {s.upper() for s in symbols}
        if "BTC" not in want:
            return []
        return [self._btc_quote()]

    def top(self, limit: int = 25, asset_class: AssetClass = AssetClass.CRYPTO) -> List[Quote]:
        return [self._btc_quote()]

    def healthy(self) -> bool:
        try:
            _get("ticker", timeout=8.0)
            return True
        except ProviderError:
            return False
