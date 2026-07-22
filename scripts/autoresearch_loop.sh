#!/usr/bin/env bash
# FinScope autonomous AutoResearch loop — self-bootstrapping wrapper for cron /
# Claude Code Routines. Deterministic ($0): pulls latest, refreshes the lake, runs
# one loop iteration, and commits an improved champion. No LLM in this path.
#
# Cron example (daily 04:00):  0 4 * * *  /path/to/FinceptTerminal/scripts/autoresearch_loop.sh
#
# Env:
#   FINSCOPE_LOOP_SYMBOLS  (default "BTC ETH SOL")
#   FINSCOPE_LOOP_DAYS     (default 730)
#   FINSCOPE_LOOP_COMMIT   ("1" to commit champion/history changes)
set -euo pipefail
cd "$(dirname "$0")/.."

SYMBOLS="${FINSCOPE_LOOP_SYMBOLS:-BTC ETH SOL}"
DAYS="${FINSCOPE_LOOP_DAYS:-730}"

python3 -c "import duckdb,pyarrow,numpy,rich,requests" 2>/dev/null || \
  pip3 install -q requests numpy rich duckdb pyarrow

# refresh data lake: try REAL history first, fall back to synthetic scale-fill
python3 -m finscope.lake ingest --symbols $SYMBOLS --days "$DAYS" >/dev/null 2>&1 || \
  python3 -m finscope.lake synth --symbols $SYMBOLS --bars "$DAYS" >/dev/null 2>&1 || true

echo "[autoresearch] $(date -u +%FT%TZ) running iteration on: $SYMBOLS ($DAYS)"
python3 -m finscope.research --loop 1 --symbols $SYMBOLS --days "$DAYS" | tee /tmp/finscope_loop.json

if [ "${FINSCOPE_LOOP_COMMIT:-0}" = "1" ]; then
  if ! git diff --quiet -- finscope/research/results/ 2>/dev/null; then
    git add finscope/research/results/champion.json finscope/research/results/loop_history.tsv 2>/dev/null || true
    git commit -q -m "chore(autoresearch): loop iteration $(date -u +%FT%TZ)" && echo "[autoresearch] committed champion/history"
  else
    echo "[autoresearch] no champion change"
  fi
fi
