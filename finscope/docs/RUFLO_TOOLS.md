# ruflo MCP tools — classification + adoption suggestions (for FinScope/OMEGA)

The `ruflo` MCP server exposes ~327 tools. Below they're grouped by function with a
concrete **verdict** for the FinScope quant machine: **ADOPT** (wire in now),
**USE** (call opportunistically), **LATER** (real fit for a future phase), **SKIP**
(orthogonal). Deterministic-first: prefer $0 local code; use these where they add
persistence, audit, or coordination we don't already have.

| Category | Representative tools | Verdict | Why / how for FinScope |
|---|---|---|---|
| **Memory / AgentDB** | `memory_store` `memory_search` `agentdb_pattern-store` `agentdb_pattern-search` `agentdb_semantic-route` `agentdb_graph-query` `memory_import_claude` | **ADOPT** (done) | Cross-session persistence of strategies, decisions, leaderboards. Already storing FinScope architecture + winning pattern. Add: semantic-route research queries; causal graph of strategy relationships. |
| **AgenticOW (copy-on-write)** | `agenticow_branch` `agenticow_checkpoint` `agenticow_speculate` `agenticow_promote` `agenticow_rollback` `agenticow_diff` | **ADOPT** | Perfect for AutoResearch: branch a strategy variant, speculate/test, promote if better / rollback if worse — the Karpathy keep-if-better loop with real state isolation. |
| **MetaHarness (audit)** | `metaharness_score` `metaharness_mcp_scan` `metaharness_threat_model` `metaharness_redblue` `metaharness_security_bench` `metaharness_genome` | **USE** | Audit FinScope's agent surface + the DAA bridge for reliability/security before any live phase. `mcp_scan --fail-on high` as a CI-style gate. |
| **Hooks / codemod** | `hooks_codemod` `hooks_pre-task` `hooks_post-edit` `hooks_route` `hooks_intelligence_*` | **USE** | `hooks_codemod` = deterministic $0 code transforms. `hooks_route` for model-tier routing (complements the Budget Governor). |
| **Neural / SONA / ruvLLM** | `neural_train` `neural_predict` `ruvllm_sona_*` `ruvllm_microlora_*` `ruvllm_hnsw_*` `embeddings_*` | **LATER** | SONA self-learning for the strategy-selection router (adaptive ensemble weighting). Meaningful once the lake has real history + a reward stream. |
| **Swarm / coordination / hive-mind** | `swarm_init` `agent_spawn` `coordination_orchestrate` `hive-mind_consensus` `daa_agent_create` `consensus` | **USE (budget-gated)** | Orchestrate research swarms — but each agent costs; gate behind the Budget Governor. Use Claude Code's own Agent tool for the actual work per ruflo's own guidance. |
| **AI Defence** | `aidefence_scan` `aidefence_analyze` `aidefence_has_pii` `aidefence_is_safe` | **USE** | Scan external data (news, PR comments, market feeds) that FinScope agents ingest for prompt-injection / PII before acting. Directly relevant to the untrusted-data handling. |
| **Performance** | `performance_benchmark` `performance_profile` `performance_bottleneck` | **USE** | Profile the backtester / optimizer at lake scale. |
| **Task / workflow / progress** | `task_*` `workflow_*` `progress_*` | **USE** | Durable task/workflow state for long autoresearch runs (complements Claude Code TaskCreate). |
| **GitHub** | `github_pr_manage` `github_repo_analyze` `github_metrics` `github_workflow` | **SKIP** | The github MCP + this session already cover PR/repo ops. |
| **Guidance / autopilot / claims / federation / business / transfer / browser / terminal / config / session / system** | `guidance_recommend` `autopilot_*` `claims_*` `federation_bbs_*` `business_pod_*` `transfer_plugin-search` `browser_*` `terminal_*` `config_*` `session_*` `system_health` | **SKIP / situational** | Orthogonal to quant. `system_health` useful for a ruflo-daemon check; `transfer_plugin-search` to discover ruflo plugins; the rest not on the quant path. |

## Top 5 to wire next (highest value × reliability)
1. **`agenticow_*`** → speculative strategy-variant testing with promote/rollback (AutoResearch state isolation).
2. **`memory_search` / `agentdb_semantic-route`** → recall past strategy decisions by meaning across sessions.
3. **`metaharness_mcp_scan` + `threat_model`** → security/reliability gate on the agent + DAA surfaces before live.
4. **`aidefence_scan`** → sanitise external market/news/PR data FinScope agents consume.
5. **`hooks_codemod`** → deterministic $0 refactors across the growing finscope/ package.

## Caveats
- ruflo MCP may be absent in headless/cron runs — every integration must degrade gracefully (detect-and-degrade, like the engine adapters).
- Spawning ruflo swarm agents costs tokens: gate all agent/swarm calls behind the Budget Governor; prefer deterministic $0 tools.
