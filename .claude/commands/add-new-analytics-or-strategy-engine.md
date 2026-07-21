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

- `finscope/analytics/*.py`
- `finscope/engines/*.py`
- `finscope/engines/strategies*.py`
- `finscope/engines/__init__.py`
- `finscope/engines/_discover.py`
- `finscope/tests/test_analytics.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Create or update files in finscope/analytics/ (for analytics) or finscope/engines/ (for strategies/engines).
- Register the new engine or strategy in the appropriate registry or __init__.py.
- Update or create tests to verify the new analytics or strategy logic.
- If relevant, update UI commands or agent interfaces to expose the new functionality.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.