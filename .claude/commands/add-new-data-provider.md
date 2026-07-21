---
name: add-new-data-provider
description: Workflow command scaffold for add-new-data-provider in FinceptTerminal.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-new-data-provider

Use this workflow when working on **add-new-data-provider** in `FinceptTerminal`.

## Goal

Adds a new data provider to the FinceptTerminal platform, enabling access to new data sources (e.g., yfinance, stooq, polymarket, cryptocom).

## Common Files

- `finscope/core/providers/*.py`
- `finscope/core/datahub.py`
- `finscope/core/providers/public_apis_registry.py`
- `finscope/data/public_apis.json`
- `finscope/tests/test_providers.py`
- `finscope/ui/commands.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Create a new provider implementation in finscope/core/providers/ (e.g., yfinance_provider.py).
- Register the provider with the DataHub or provider registry.
- Optionally, add demo data or update public_apis.json if relevant.
- Add or update tests in finscope/tests/test_providers.py.
- Update UI command handlers if new commands are exposed.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.