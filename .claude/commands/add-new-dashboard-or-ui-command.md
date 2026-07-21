---
name: add-new-dashboard-or-ui-command
description: Workflow command scaffold for add-new-dashboard-or-ui-command in FinceptTerminal.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /add-new-dashboard-or-ui-command

Use this workflow when working on **add-new-dashboard-or-ui-command** in `FinceptTerminal`.

## Goal

Adds a new dashboard panel, UI command, or function extension to the terminal interface.

## Common Files

- `finscope/ui/commands.py`
- `finscope/ui/panels.py`
- `finscope/ui/functions_ext.py`
- `finscope/ui/functions_ext_market.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Implement new command or panel logic in finscope/ui/commands.py or finscope/ui/panels.py.
- If needed, add or update function extensions in finscope/ui/functions_ext.py or related files.
- Wire the new command/panel into the main UI or command registry.
- Optionally update documentation or help mnemonics.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.