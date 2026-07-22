# FinScope notebooks

Interactive tours of the `finscope/` deterministic quant-research engine.

## `finscope_research.ipynb`

A runnable, end-to-end walk through the machine — every cell is pure numpy/DuckDB
($0, no LLM calls, reproducible):

1. **Data** — providers → `DataHub` (offline via the bundled real Crypto.com snapshot)
2. **Strategies** — the causal `weights(series) -> [-1,1]` protocol, `_shift1`
   no-lookahead proof (core + Fable alpha + 40 Ruliology cellular-automaton strategies)
3. **Backtest** — the $0 native engine (no lookahead, tx-cost, out-of-sample Sharpe) + equity curve
4. **The null test** — driftless noise ⇒ ≈0 Sharpe is what makes results trustworthy
5. **The robustness study** — 54 strategies × 20 seeds × 4 regimes = 4,320 free
   backtests; per-regime profile + the robust-edge-vs-null honesty scatter
6. **Run it yourself** — reproducible, independent replications

The committed copy already has outputs + charts embedded, so it reads without running.

### Run it fresh

```bash
pip install jupyter matplotlib pandas numpy duckdb
jupyter notebook finscope/notebooks/finscope_research.ipynb
# or headless, re-executing every cell:
jupyter nbconvert --to notebook --execute --inplace finscope/notebooks/finscope_research.ipynb
```

Runs from the repo root; the setup cell adds the package to `sys.path` automatically.
