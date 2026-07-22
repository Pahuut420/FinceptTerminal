# Ruliology Strategy Batch — Results & Verification

40 causal trading strategies derived from Stephen Wolfram's computational universe
(*A New Kind of Science* / Ruliology), authored by 20 Fable-5 agents (one isolated
`rul_*.py` module each, 2 strategies per module) and integrated deterministically.
Every strategy conforms to the FinScope `Strategy` protocol: `weights(series) ->
np.ndarray` in [-1, 1], causal via `_shift1` (the weight held over bar *t* uses
information only through *t-1*).

## The 20 modules → 40 strategies

| Module | Ruliology object | Strategies |
|---|---|---|
| rul_01 rule30 | Rule 30 (class-3 chaos, entropy source) | `rule30_entropy_gate`, `rule30_entropy_margin` |
| rul_02 rule110 | Rule 110 (class-4, universal, gliders) | `rule110_glider_momentum`, `rule110_structure_polarity` |
| rul_03 rule90 | Rule 90 (additive XOR, Sierpiński/nesting) | `rule90_fractal_trend`, `rule90_xor_echo` |
| rul_04 wolfram_class | 4 behavior classes (fixed/periodic/chaotic/complex) | `wolfram_class_regime`, `wolfram_class_ensemble` |
| rul_05 rule184 | Rule 184 (traffic / particle transport) | `rule184_flow`, `rule184_jam` |
| rul_06 majority | Rule 232 (majority / consensus) | `rul232_consensus`, `rul232_coarsen` |
| rul_07 reversible | Reversible (2nd-order) CA | `rul07_reversible_echo_fade`, `rul07_reversible_parity_echo` |
| rul_08 totalistic | 3-color totalistic CA | `ca3_totalistic_center`, `ca3_totalistic_drift` |
| rul_09 mobile | Mobile automata (moving active cell) | `mobile_drift`, `mobile_escape` |
| rul_10 turing | Turing machines (tape + head) | `turing_drift`, `turing_regime` |
| rul_11 tag | Tag systems (halting dynamics) | `tag2_halt_trend`, `tag2_transient_fade` |
| rul_12 substitution | Substitution systems (Thue-Morse, Fibonacci word) | `thue_morse_phase`, `fibonacci_word_flow` |
| rul_13 multiway | Multiway systems (branchial space) | `multiway_branch`, `multiway_branchial` |
| rul_14 rulefit | Nearest-ECA-rule program induction (256 rules) | `eca_rulefit`, `eca_rulefit_ensemble` |
| rul_15 irreducibility | Computational irreducibility (LZ76 complexity) | `lz76_irreducibility_gate`, `lz76_reducibility_scale` |
| rul_16 continuous | Continuous CA (Lenia) | `lenia_growth`, `lenia_field` |
| rul_17 cyclic | Cyclic CA (Griffeath) | `cca_phase`, `cca_wavefront` |
| rul_18 network | Network / Wolfram-model automata | `net_connectivity`, `net_clustering` |
| rul_19 transition | Class-transition (order↔chaos edge) | `class_transition`, `class_transition_impulse` |
| rul_20 ruliad | Ruliad ensemble (skill-weighted rule panel) | `ruliad_vote`, `ruliad_softmax` |

Auto-discovered by `ruliology/__init__.py` into `RULIOLOGY_STRATEGIES`; auto-wired
into `research/optimizer.py::DEFAULT_GRIDS` (default params) so the deterministic
grid/genetic search covers them with no further edits.

## Verification

**1. Causality (deterministic, 3 independent checks — all 40 pass):**
- Each authoring agent verified truncation invariance + last-close perturbation.
- Re-verified independently here: bounds/finiteness, last-close perturbation, and
  truncation invariance `weights(px[:k]) == weights(px)[:k]`.
- Locked in as a regression guard: `finscope/tests/test_ruliology.py` (5 tests).

**2. Null test (statistical — the trustworthiness check):** 16 independent
driftless fat-tailed GBM paths (730 bars). A correct no-lookahead strategy must
score ≈0 OOS Sharpe on structureless noise; a strong *positive* Sharpe would mean
a lookahead/overfitting bug.
- **Zero strategies flagged.** Cross-seed mean OOS Sharpe per strategy ranged
  [-0.93, +0.66]; batch mean **-0.34** (turnover cost dragging a zero-edge signal
  slightly negative — the expected null signature). `multiway_branch` = exactly
  0.000 on every seed (ideal abstention — flat unless its pattern appears).
- A single-seed run had thrown `cca_phase` at +1.89; across 16 seeds it averages
  +0.05 — confirming that apparent outlier was sampling variance, not lookahead,
  which is exactly why the multi-seed null matters when screening 40 candidates.

**3. Structure response (are they live, or degenerate-flat?):** 16 up-drift paths.
The strategies respond to real structure rather than sitting flat. Standout:
`rule110_structure_polarity` +0.745 mean OOS Sharpe (positive on 69% of seeds),
beating baseline momentum (-0.08) on the same fat-tailed trending series — a
genuine candidate for real-data validation.

## Honesty

Passing the null test means **no lookahead**, not **profitable**. These magnitudes
are on a single synthetic model with wide cross-seed spread (std ≈ 1.2); they are
structure-response diagnostics, not investable alpha. Real-data ranking awaits a
filled DuckDB lake with enough history. Before any paper→live move, the mesh/onchain
risk layer (fitness gates, fractional-Kelly, hard cap, kill-switch) still applies.
