"""
Token optimisation — deterministic-first techniques to cut LLM spend.

The cheapest token is the one you never send. In priority order (most → least
impactful), all deterministic:

  1. Deterministic tool results (FACT pattern) — FinScope's DashboardAgent returns
     JSON with NO model call. Prefer a tool over asking an LLM. (already the design)
  2. Response cache — identical (model, system, messages) => return the cached
     answer for $0. `PromptCache` below (sha256 keyed, on disk).
  3. KV / prompt caching — mark the stable prefix (system prompt + tool schema)
     with Anthropic `cache_control`; repeated calls reuse the cached prefix at
     ~10% input cost. `add_cache_control()` shapes the request.
  4. Compression — `compact()` strips filler/whitespace (caveman-style) before
     sending; `estimate_tokens()` sizes payloads.
  5. Tier routing — BudgetGovernor.recommend_tier(): cheapest capable model.

Nothing here calls an LLM. `TokenOptimizer` ties them together and records the
estimated savings against a BudgetGovernor.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import List, Optional

from finscope.economy.governor import BudgetGovernor

_FILLER = re.compile(r"\b(please|kindly|just|really|very|actually|basically|"
                     r"in order to|as you can see|it should be noted that)\b", re.I)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Deterministic, no tokenizer dep."""
    return max(1, len(text) // 4)


def compact(text: str, drop_articles: bool = False) -> str:
    """Caveman-style deterministic compression: collapse whitespace, drop filler."""
    t = _FILLER.sub("", text)
    t = re.sub(r"\s*,(\s*,)+", ",", t)   # collapse commas orphaned by filler removal
    t = re.sub(r"\s+([,.;:])", r"\1", t)  # no space before punctuation
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    if drop_articles:
        t = re.sub(r"\b(the|a|an)\s+", "", t, flags=re.I)
    return t.strip()


def compaction_ratio(before: str, after: str) -> float:
    b = estimate_tokens(before)
    return round(1.0 - estimate_tokens(after) / b, 4) if b else 0.0


def add_cache_control(system: str, tools: Optional[list] = None) -> dict:
    """Shape an Anthropic request so the stable prefix (system + tools) is KV-cached.

    Returns the pieces to pass to the Messages API. The `cache_control` breakpoint
    marks everything up to it as reusable across calls — the biggest single win for
    an agent that reissues the same system prompt + tool schema every turn.
    """
    out = {"system": [{"type": "text", "text": system,
                       "cache_control": {"type": "ephemeral"}}]}
    if tools:
        tools = [dict(t) for t in tools]
        tools[-1]["cache_control"] = {"type": "ephemeral"}  # cache the whole tool block
        out["tools"] = tools
    return out


class PromptCache:
    """Deterministic on-disk response cache. Identical request => $0 cached reply."""
    def __init__(self, cache_dir: Optional[str] = None):
        self.dir = cache_dir or os.path.join(
            os.environ.get("FINSCOPE_ECONOMY_DIR", os.path.expanduser("~/.finscope/economy")),
            "prompt_cache")
        os.makedirs(self.dir, exist_ok=True)

    @staticmethod
    def _key(model: str, system: str, messages) -> str:
        blob = json.dumps({"m": model, "s": system, "x": messages}, sort_keys=True,
                          default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, model: str, system: str, messages) -> Optional[str]:
        p = os.path.join(self.dir, self._key(model, system, messages) + ".json")
        if os.path.exists(p):
            try:
                return json.load(open(p, encoding="utf-8"))["response"]
            except (OSError, ValueError, KeyError):
                return None
        return None

    def put(self, model: str, system: str, messages, response: str) -> None:
        p = os.path.join(self.dir, self._key(model, system, messages) + ".json")
        json.dump({"response": response}, open(p, "w", encoding="utf-8"))


class TokenOptimizer:
    def __init__(self, governor: Optional[BudgetGovernor] = None,
                 cache: Optional[PromptCache] = None):
        self.governor = governor or BudgetGovernor()
        self.cache = cache or PromptCache()

    def plan(self, system: str, messages: List[dict], model_units: float = 1.0) -> dict:
        """Return a deterministic optimisation plan for a would-be LLM call."""
        raw = system + json.dumps(messages, default=str)
        comp = compact(raw)
        cached = self.cache.get("*", system, messages) is not None
        tier = self.governor.recommend_tier(units=model_units)
        return {
            "cached_hit": cached,                          # if true -> $0, skip the call
            "estimated_tokens_raw": estimate_tokens(raw),
            "estimated_tokens_compacted": estimate_tokens(comp),
            "compaction_ratio": compaction_ratio(raw, comp),
            "use_prompt_cache": True,                       # KV-cache the system+tools prefix
            "recommended_tier": tier["tier"],
            "budget_remaining": self.governor.remaining(),
            "advice": ("serve from cache ($0)" if cached else
                       f"compact -> KV-cache prefix -> {tier['tier']} tier"),
        }
