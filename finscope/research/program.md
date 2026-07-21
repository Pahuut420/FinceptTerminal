# AutoResearch program — FinScope strategy search

Karpathy-style autonomous optimisation loop, **deterministic mutator (no LLM)**.

## Goal
Find the highest risk-adjusted-return trading strategies on the available price
history, without overfitting.

## Metric (one number)
`score = mean_oos_sharpe - turnover_penalty * mean_turnover`, aggregated across
symbols. Out-of-sample Sharpe (Sharpe on the held-out tail) is the primary term
so the search cannot win by fitting in-sample noise. Tie-break: lower mean
max-drawdown.

## Editable surface
- `finscope/engines/strategies.py` — core strategy weight functions
- `finscope/engines/strategies_alpha.py` — advanced/alpha strategies (Fable-authored)
- `finscope/research/optimizer.py::DEFAULT_GRIDS` — the parameter search space

## Loop (deterministic, $0)
1. Mutate: enumerate the parameter grid (or seeded genetic search) — pure code.
2. Evaluate: `NativeEngine.backtest()` on each candidate across all symbols — numpy, no LLM.
3. Select: keep candidates ranked by `score`; the leaderboard is the result.
4. Repeat over the whole strategy library.

Run it:
```
python -m finscope.research                    # full leaderboard
python -m finscope.research --strategy kama_trend --genetic
python -m finscope.research --json             # machine-readable for an agent
```

## Rules
- No lookahead: every strategy shifts its signal forward one bar.
- Costs: turnover charged at `cost_bps` per bar.
- Honesty: small samples are flagged, never hidden. Scores become meaningful
  once the DuckDB lake (`finscope/lake/`) is filled with real history.
- LLM use: only for *authoring* new strategy ideas (Fable, occasional, small).
  The *search* for the best is always deterministic code.

## Next
- Fill the data lake so OOS windows are statistically meaningful.
- Add OMEGA-style fitness gates (IC/ICIR, Kelly ¼ sizing, kill-switch) before any
  paper/live promotion — see `omega-trading-master`.
