"""
Autonomous AutoResearch loop — deterministic core, budget-gated creativity.

Each iteration (all $0, no LLM):
  1. Re-run the deterministic optimizer over the CURRENT lake data + strategy set.
  2. Compare the best candidate to the reigning champion (`results/champion.json`).
  3. Keep-if-better (Karpathy AutoResearch); append to `results/loop_history.tsv`.
  4. Track a plateau counter (consecutive iterations with no improvement).

The loop *improves* whenever its inputs change — new lake history (the user runs
`finscope.lake ingest`), a new strategy (Fable), or a widened search space. When it
plateaus for `plateau_patience` iterations, it flags an **escalation**: author a new
strategy with Fable — but only if the BudgetGovernor has headroom. So the loop runs
free forever and spends only when genuinely stuck. Idempotent + reproducible.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, List, Optional

from finscope.research.optimizer import optimize_all
from finscope.economy.governor import BudgetGovernor

_RESULTS = os.path.join(os.path.dirname(__file__), "results")
_CHAMPION = os.path.join(_RESULTS, "champion.json")
_HISTORY = os.path.join(_RESULTS, "loop_history.tsv")


def _load_champion() -> Optional[dict]:
    try:
        return json.load(open(_CHAMPION, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_champion(c: dict) -> None:
    os.makedirs(_RESULTS, exist_ok=True)
    json.dump(c, open(_CHAMPION, "w", encoding="utf-8"), indent=2)


def _append_history(row: dict) -> None:
    os.makedirs(_RESULTS, exist_ok=True)
    new = not os.path.exists(_HISTORY)
    with open(_HISTORY, "a", encoding="utf-8") as f:
        if new:
            f.write("ts\titeration\tbest_strategy\tbest_score\timproved\tplateau\tnote\n")
        f.write(f"{row['ts']}\t{row['iteration']}\t{row['best_strategy']}\t"
                f"{row['best_score']}\t{row['improved']}\t{row['plateau']}\t{row['note']}\n")


def run_iteration(symbols: Optional[List[str]] = None, days: int = 365,
                  iteration: int = 0, now: Optional[float] = None) -> dict:
    now = now if now is not None else time.time()
    res = optimize_all(symbols=symbols, days=days, top_k=3)
    top = (res.get("global_top") or [None])[0]
    champ = _load_champion()
    plateau = int(champ.get("plateau", 0)) if champ else 0
    improved = bool(top) and (champ is None or top["score"] > champ.get("best_score", -1e9))
    if improved:
        _save_champion({"best_strategy": top["strategy"], "best_params": top["params"],
                        "best_score": top["score"], "symbols": res.get("symbols"),
                        "days": days, "ts": now, "plateau": 0})
        plateau = 0
        note = f"new champion: {top['strategy']} score={top['score']}"
    else:
        plateau += 1
        if champ:
            champ["plateau"] = plateau
            _save_champion(champ)
        note = f"no improvement (plateau {plateau})"
    row = {"ts": int(now), "iteration": iteration,
           "best_strategy": top["strategy"] if top else "-",
           "best_score": top["score"] if top else 0.0,
           "improved": improved, "plateau": plateau, "note": note}
    _append_history(row)
    return {**row, "evaluated": res.get("evaluated"), "data_note": res.get("note", "")}


def run_loop(iterations: int = 1, symbols: Optional[List[str]] = None, days: int = 365,
             plateau_patience: int = 3, governor: Optional[BudgetGovernor] = None,
             escalate: Optional[Callable[[dict], dict]] = None,
             escalation_cost: float = 0.8) -> dict:
    """Run N iterations. On a plateau, call `escalate` (author a new strategy) only
    if the BudgetGovernor can afford it — otherwise flag it and keep running free."""
    gov = governor or BudgetGovernor()
    history, escalations = [], []
    for i in range(iterations):
        r = run_iteration(symbols=symbols, days=days, iteration=i)
        history.append(r)
        if r["plateau"] >= plateau_patience:
            gate = gov.gate(escalation_cost, "autoresearch-escalation")
            if gate["approved"] and escalate is not None:
                gov.record(escalation_cost, "fable", "author new strategy")
                escalations.append(escalate(r))
            else:
                escalations.append({"skipped": True, "reason":
                                    "escalation flagged but " +
                                    ("no escalate() wired" if escalate is None
                                     else f"budget: {gate['reason']}"),
                                    "iteration": i})
    champ = _load_champion()
    return {"iterations": iterations, "champion": champ,
            "escalations": escalations, "history_tail": history[-5:],
            "budget": gov.status(),
            "note": "deterministic $0 core; escalation is budget-gated Fable authoring"}
