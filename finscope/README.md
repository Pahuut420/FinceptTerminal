# FinScope

**A Bloomberg-style terminal built on FinceptTerminal's data fleet.**

FinScope is a runnable, dependency-light Python TUI that turns FinceptTerminal's
300+ market data connectors into a single command-driven terminal: live crypto &
on-chain quotes, a numpy + Wolfram analytics engine, and a `rich` dashboard driven
by Bloomberg-style mnemonic function codes.

```
python -m finscope                 # interactive Bloomberg-style REPL
python -m finscope --live          # auto-refreshing crypto market monitor
python -m finscope --once "VOL BTC" # run one function and exit
python -m finscope --demo          # headless snapshot (works offline)
```

## Why

FinceptTerminal ships a **C++/Qt desktop app** plus a fleet of **319 Python data
connectors** (crypto, equities, macro, commodities, on-chain, ESG). FinScope is a
thin, portable terminal that unifies those data sources behind one interface and
layers quant analytics + a keyboard-driven UI on top — a Bloomberg homage you can
run in any shell.

## Function codes

Type `HELP` in the app. Core functions:

| Code | Description |
|------|-------------|
| `CRYPTO` / `WEI` | Crypto market monitor — top coins by market cap |
| `TOP <n>` | Top n coins |
| `MOVERS` | Biggest 24h gainers & losers |
| `DES <SYM>` | Security description |
| `GP <SYM> [days]` | Price graph — sparkline + range |
| `VOL <SYM> [days]` | Risk metrics — vol, Sharpe, VaR, max drawdown |
| `CORR [SYMS...]` | Correlation matrix (watchlist by default) |
| `WATCH <SYM...>` | Show/set the watchlist |
| `WL <SYM>` | Show the Wolfram Language for annualized vol (offload demo) |
| `API <query>` | Search the public-apis free-API registry (1,586 APIs) |
| `PROV` | Data providers + health |
| `BETA/SORTINO/ES/MC <SYM>` | Advanced risk (beta vs BTC, Sortino, expected shortfall, Monte-Carlo VaR) |

## Architecture

```
finscope/
├── core/
│   ├── contracts.py        # Quote / OHLCV / Series / RiskMetrics — the shared interface
│   ├── datahub.py          # DataHub: routing, TTL cache, graceful degradation
│   └── providers/          # self-registering data adapters (auto-discovered)
│       ├── base.py         #   Provider ABC + @register + priority routing
│       ├── coinpaprika.py  #   live crypto (no key)
│       ├── coincap.py / cryptocompare.py / blockchain_com.py / defillama.py
│       ├── public_apis_registry.py   # 1,586 free APIs mined from the public-apis repo
│       └── demo.py         #   bundled REAL snapshot (Crypto.com) — offline/CI fallback
├── analytics/
│   ├── engine.py           # numpy: returns, vol, Sharpe, drawdown, VaR, SMA/EMA/RSI, corr, MC-VaR
│   ├── risk.py             # advanced: beta, Sortino, Calmar, ES, Hurst, drawdown series
│   └── wolfram_bridge.py   # optional Wolfram MCP offload (query-gen + numpy fallback)
├── ui/
│   ├── app.py              # REPL / --live / --once / snapshot
│   ├── commands.py         # mnemonic router (+ functions_ext.py plugin hook)
│   └── panels.py           # rich tables, sparklines, Bloomberg-amber chrome
├── coordination/
│   ├── repo_graph.py       # graphify knowledge-graph accessor
│   └── coordination_map.json  # distilled fleet↔finscope structural map
├── data/
│   ├── demo_seed.json      # real Crypto.com snapshot (top-30 + BTC/ETH/SOL candles)
│   └── public_apis.json    # distilled public-apis catalog
└── tests/                  # contracts, analytics, providers, commands
```

### Data layer

Providers are self-registering: drop a `Provider` subclass in `core/providers/`
and it's auto-discovered and routed by asset class + capability + priority. Live
providers (priority 50) are tried first; the bundled `demo` provider (priority
900) answers only when the network is unreachable — so the terminal always
renders. The `DataHub` catches every `ProviderError` and falls back to cache →
next provider → empty, never crashing the terminal.

### Analytics + Wolfram

`AnalyticsEngine` is pure numpy and always available. `WolframBridge` can offload
selected computations to a Wolfram Language evaluator (via the Wolfram MCP) and
transparently falls back to numpy when none is wired. Cross-checked live: ETH
annualized vol = `0.44404` from both numpy and the Wolfram MCP.

### Coordination graph (graphify)

The build was coordinated with a [graphify](https://github.com/safishamsi/graphify)
knowledge graph over the connector fleet (**31,235 nodes / 83,513 edges / 1,679
communities**, built locally with tree-sitter, $0, no LLM). The distilled map in
`coordination/coordination_map.json` records the fleet's god nodes, community
clusters, and how FinScope's providers map onto the fleet scripts.

```python
from finscope.coordination import RepoGraph
g = RepoGraph()
print(g.summary())          # fleet + finscope graph stats
print(g.god_nodes()[:5])    # most-connected core abstractions
print(g.explain("Series"))  # live graphify query (if graph.json present)
```

## Data sources

- **Crypto (no key):** CoinPaprika, CoinCap, CryptoCompare, Blockchain.com, DefiLlama
- **Free-API registry:** 1,586 APIs across 51 categories (from the `public-apis` repo)
- **Bundled demo data:** real Crypto.com snapshot (sourced via the Crypto.com MCP)
- Extensible to the full FinceptTerminal fleet of 319 connectors

## Requirements

Python 3.9+, `requests`, `numpy`, `rich`. No API keys required for the default
crypto providers.

## Notes

- Live data needs outbound HTTPS; in restricted sandboxes the bundled demo data
  keeps every function working.
- This is a portable companion to the FinceptTerminal Qt app, reusing its data
  connector philosophy in a lightweight, scriptable form.
