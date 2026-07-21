---
name: add-or-update-documentation-ledger
description: Workflow command scaffold for add-or-update-documentation-ledger in FinceptTerminal.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-or-update-documentation-ledger

Use this workflow when working on **add-or-update-documentation-ledger** in `FinceptTerminal`.

## Goal

Adds or updates documentation files, especially program specs or request ledgers, to track features and research goals.

## Common Files

- `finscope/docs/*.md`
- `finscope/research/*.md`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Create or update markdown documentation in finscope/docs/ or finscope/research/ (e.g., REQUEST_LEDGER.md, program.md).
- Force-add *.md files if ignored by .gitignore.
- Reference new docs in commit messages or PRs.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.