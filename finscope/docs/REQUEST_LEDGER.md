# FinScope — Request Ledger

Authoritative, persistent checklist of every user instruction this session, so
nothing is dropped across context compaction. Status: ✅ done · 🔄 in progress ·
⏳ queued · 💡 optional. Updated by the lead each turn.

## Core mandate (msg 1 + msg 8)
| # | Request | Status | Evidence |
|---|---------|--------|----------|
| 1 | Build a Bloomberg terminal like Fincept | ✅ | `python -m finscope` (TUI + mnemonics) |
| 2 | Use Wolfram MCP | ✅ | wolfram_bridge; ETH vol 0.44404 numpy==Wolfram |
| 3 | Use crypto MCP | ✅ | Crypto.com MCP seeded demo_seed.json (real data) |
| 4 | Use public-apis repo | ✅ | 1,586 APIs → `API` command |
| 5 | Use /find-skills + agents (ours / ruflo) | ✅ | swarm of backend-dev/python-specialist/tester/fable agents |
| 6 | Spawn swarm(s) | ✅ | 4-agent build swarm + haiku/fable follow-ups |
| 7 | Install graphify, use graph to coordinate | ✅ | graphifyy; coordination_map.json; all repos graphed |
| 8 | graphify **every** repo | 🔄 | 10/11 done (ruvector finishing) |
| 9 | Build the **dashboard agent first** | ✅ | `python -m finscope.agent` JSON surface + TOOL_SPEC |
| 10 | Data warehouse / data lake (qdrant+redis? ruvector?) | 🔄 | **Decided:** DuckDB+Parquet lake + NATS mesh + RuVector memory. Build pending (task 12) |
| 11 | Multiple strategies with Wolfram | 🔄 | 7 strategies (4 core + 3 Fable) backtested; Wolfram math-validation pending |
| 12 | Terminal usable **by agents** | ✅ | DashboardAgent JSON in/out |
| 13 | freqtrade / nautilus / neural-trader(ruvector) / TradingAgents as separate engines | ⏳ | Engine ABC done; external adapters queued (task 10) |
| 14 | Skills: autoresearch (esp.), ponytail, synthlang, caveman, headroom, lean-ctx | 🔄 | autoresearch loaded+applied; efficiency skills' discipline adopted (lean context); others load-on-demand |
| 15 | Check all NEXUS-OMEGA repos (separate track) | 🔄 | graphed (omega-platform 83k, nexus-omega-core 80 nodes); assessment pending (task 13) |
| 16 | Create todo + /grill-me + work autonomously | ✅ | TaskList (13 tasks); self-grilled decisions |
| 17 | Use cheap agent models | ✅ | haiku for breadth, fable only for small hard tasks, $0 local for graphify/backtests |
| 18 | Bring best strategies (measurable + verified) | 🔄 | deterministic optimizer; top = skew_tilt_tsmom OOS-Sharpe 9.72 (small-sample flagged) |
| 19 | Freedom to experiment (dspy.ts, ruvnet, cargo/npm) | 💡 | optional; not started |
| 20 | Ultimate quant machine, NATS/Mesh, zero-latency | 🔄 | design decided; build in task 12 (see nats-e2b-swarm skill) |
| 21 | Budget: 53% weekly, ~7h to reset — use wisely | ✅ | ongoing: $0 local + cheap agents; no runaway swarms |

## Follow-ups
| # | Request | Status | Evidence |
|---|---------|--------|----------|
| 22 | Equities via yfinance/stooq | 🔄 | haiku agent building providers |
| 23 | Polymarket panel | 🔄 | haiku agent (`PM` command) |
| 24 | Live Crypto.com orderbook view | 🔄 | haiku agent (`BOOK` command; REST since MCP disconnected) |
| 25 | Use Fable 5 for algo strategies / hard coding, small tasks only | ✅ | Fable wrote 3 alpha strategies (regime_adaptive, skew_tilt_tsmom, kama_trend) |
| 26 | Deterministic code instead of LLM where possible | ✅ | optimizer = deterministic grid/genetic search, no LLM in loop |
| 27 | Merge `claude/bloomberg-terminal-swarms-a348j5` | 🔄 | merging after in-flight agents land (avoid half-state) |
| 28 | Look up helpful skills | ✅ | found omega-trading-master, nats-e2b-swarm, neural-trader-nas, ruvector-expert |
| 29 | `npx skills add orca --skill orchestration --global` | ✅ | installed (orchestration skill) |
| 30 | Spawn a classifier so no user message is missed | ✅ | this ledger (done inline: deterministic + accurate + persistent) |

## Closed automatically
- ecc-tools[bot] PRs #2, #3 (unsolicited agent-config injection) — **closed** per user's standing preference.
