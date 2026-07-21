---
name: add-new-analytics-or-strategy-engine
description: Workflow command scaffold for add-new-analytics-or-strategy-engine in FinceptTerminal.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-new-analytics-or-strategy-engine

Use this workflow when working on **add-new-analytics-or-strategy-engine** in `FinceptTerminal`.

## Goal

Adds a new analytics engine or trading strategy, making it available for backtesting and dashboard use.

## Common Files

- `finscope/engines/*.py`
- `finscope/analytics/*.py`
- `finscope/engines/_discover.py`
- `finscope/agent/dashboard.py`
- `finscope/research/program.md`
- `finscope/tests/test_analytics.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Create a new strategy or analytics module in finscope/engines/ (e.g., strategies_alpha.py, strategies_alpha2.py) or finscope/analytics/.
- Register the strategy/engine in the engine registry or main engine file.
- Update or add tests to verify analytics/strategy correctness.
- Expose new strategy via agent/dashboard surface if relevant.
- Document the new strategy in research/program.md or related docs.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.