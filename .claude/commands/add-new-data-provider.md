---
name: add-new-data-provider
description: Workflow command scaffold for add-new-data-provider in FinceptTerminal.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-new-data-provider

Use this workflow when working on **add-new-data-provider** in `FinceptTerminal`.

## Goal

Adds a new data provider (API or data source) to the system, making it available for analytics and UI.

## Common Files

- `finscope/core/providers/*.py`
- `finscope/core/providers/__init__.py`
- `finscope/core/providers/public_apis_registry.py`
- `finscope/data/demo_seed.json`
- `finscope/tests/test_providers.py`
- `finscope/ui/functions_ext.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Create a new provider implementation in finscope/core/providers/ (e.g., yfinance_provider.py, stooq_provider.py, cryptocom_provider.py, polymarket_provider.py).
- Register the provider so it is discoverable by the DataHub or provider registry.
- Optionally add demo data or update provider registry files if needed.
- Update or create tests in finscope/tests/ if relevant.
- Update UI commands or function extensions if the provider exposes new commands.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.