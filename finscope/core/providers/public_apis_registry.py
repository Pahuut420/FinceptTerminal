"""
public-apis registry — search surface over the community `public-apis` catalog.

Backs the `API <query>` terminal function (see `finscope.ui.commands.cmd_api`).
Loads a pre-built, distilled JSON bundle (`finscope/data/public_apis.json`,
generated once offline from https://github.com/public-apis/public-apis's
README.md) into a module-level cache and exposes a few read-only lookups.

Design notes:
    * stdlib only (json, pathlib, re) — no network calls, no third-party deps.
    * Never raises on import or on a bad/missing data file: every public
      function degrades to an empty list rather than propagating an
      exception, so a corrupt or absent bundle can never crash the terminal.
    * Every row returned to callers carries at least the contract keys the
      command handler expects: name, category, auth, https, description.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["search", "sample", "all_apis", "categories"]

# finscope/core/providers/public_apis_registry.py -> finscope/data/public_apis.json
_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "public_apis.json"

# Categories from the source catalog surfaced ahead of everything else — this
# is a finance terminal, so finance/crypto APIs are the ones users actually want.
_PRIORITY_CATEGORIES = ("Finance", "Cryptocurrency")

_cache: Optional[List[Dict[str, Any]]] = None


def _load() -> List[Dict[str, Any]]:
    """Load and cache the distilled bundle. Never raises."""
    global _cache
    if _cache is not None:
        return _cache

    rows: List[Dict[str, Any]] = []
    try:
        with _DATA_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, list):
            rows = [r for r in raw if isinstance(r, dict)]
    except Exception:
        rows = []

    _cache = rows
    return _cache


def _is_free(row: Dict[str, Any]) -> bool:
    auth = (row.get("auth") or "").strip()
    return not auth or auth.lower() == "none"


def _is_priority_category(row: Dict[str, Any]) -> bool:
    return (row.get("category") or "").strip().lower() in {c.lower() for c in _PRIORITY_CATEGORIES}


def _project(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a raw bundle row into the shape cmd_api expects (plus a
    couple of extras that don't hurt)."""
    return {
        "name": row.get("name") or "-",
        "category": row.get("category") or "-",
        "auth": row.get("auth") or "None",
        "https": bool(row.get("https", False)),
        "description": row.get("description") or "",
        "cors": row.get("cors") or "Unknown",
        "link": row.get("link") or "",
    }


def _matches(row: Dict[str, Any], needle: str) -> bool:
    haystack = " ".join((
        row.get("name") or "",
        row.get("description") or "",
        row.get("category") or "",
    )).lower()
    return needle in haystack


def _rank_key(row: Dict[str, Any]):
    # Finance/Cryptocurrency first, then free (no-auth) APIs first, then name.
    return (
        0 if _is_priority_category(row) else 1,
        0 if _is_free(row) else 1,
        (row.get("name") or "").lower(),
    )


def search(query: str, limit: int = 25) -> List[Dict[str, Any]]:
    """Case-insensitive search over name/description/category.

    Ranks Finance/Cryptocurrency APIs first, then no-auth ("None") APIs
    first, within the set of matches. Returns [] if the bundle is missing
    or the query matches nothing — never raises.
    """
    data = _load()
    if not data:
        return []

    needle = (query or "").strip().lower()
    if not needle:
        return sample(limit)

    matched = [r for r in data if _matches(r, needle)]
    matched.sort(key=_rank_key)
    return [_project(r) for r in matched[:max(limit, 0)]]


def sample(limit: int = 25) -> List[Dict[str, Any]]:
    """A spread across categories (round-robin), Finance/Cryptocurrency
    categories surfaced first. Returns [] if the bundle is missing.
    """
    data = _load()
    if not data:
        return []

    by_category: Dict[str, List[Dict[str, Any]]] = {}
    for row in data:
        cat = row.get("category") or "Other"
        by_category.setdefault(cat, []).append(row)

    for rows in by_category.values():
        rows.sort(key=lambda r: (0 if _is_free(r) else 1, (r.get("name") or "").lower()))

    priority = [c for c in _PRIORITY_CATEGORIES if c in by_category]
    rest = sorted(c for c in by_category if c not in priority)
    ordered_categories = priority + rest

    result: List[Dict[str, Any]] = []
    cursors = {c: 0 for c in ordered_categories}
    limit = max(limit, 0)
    while len(result) < limit:
        progressed = False
        for cat in ordered_categories:
            if len(result) >= limit:
                break
            rows = by_category[cat]
            pos = cursors[cat]
            if pos < len(rows):
                result.append(rows[pos])
                cursors[cat] = pos + 1
                progressed = True
        if not progressed:
            break

    return [_project(r) for r in result[:limit]]


def all_apis() -> List[Dict[str, Any]]:
    """Every parsed API row, normalized to the standard shape."""
    return [_project(r) for r in _load()]


def categories() -> List[str]:
    """Sorted list of distinct category names present in the bundle."""
    return sorted({r.get("category") for r in _load() if r.get("category")})
