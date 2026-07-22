"""
rul_10 — Small-Turing-machine state signal (NKS ch. 3: Turing machines).

In NKS a Turing machine is a head with a tiny internal state that reads/writes a
tape and moves left/right; even 2-state, 2-color machines reach class-4
complexity (Wolfram's 2,3 machine is universal). Here the tape is the recent
return-sign sequence (oldest -> newest = left -> right) and the head is driven
across it by a fixed transition table:

  * the head's NET DISPLACEMENT encodes direction — rightward drift means the
    machine keeps confirming toward the newest (up) bars, leftward the reverse;
  * the head's INTERNAL STATE (bull/bear) plus how often it flips encodes
    regime — a "committed" state with persistent drift is a trend, rapid state
    cycling is indecision and the signal is attenuated to flat.

Both machines genuinely WRITE: a lone counter-move gets overwritten (forgiven)
on first contact, so revisits interact with a modified tape — hysteresis and
feedback, not just a passive walk over the symbols.

Causality: for bar t the tape uses returns up to and including bar t only; the
resulting signal is then shifted one bar forward, so weights()[t] is decided at
the close of t-1. No lookahead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from finscope.core.contracts import Series

# transition tables: (state, symbol) -> (write, move, next_state)
# states: 0 = bull-leaning "A", 1 = bear-leaning "B"
_Rule = Dict[Tuple[int, int], Tuple[int, int, int]]

# --- 2-state, 2-color machine (symbols: 1 = up bar, 0 = down bar) -----------
# Mirror-symmetric under (A<->B, 1<->0, R<->L): no long/short bias by design.
_RULE_DRIFT: _Rule = {
    (0, 1): (1, +1, 0),  # bull reads up:    confirm, march toward newest
    (0, 0): (1, +1, 1),  # bull reads down:  forgive (overwrite to up), arm bear
    (1, 0): (0, -1, 1),  # bear reads down:  confirm, march toward oldest
    (1, 1): (0, -1, 0),  # bear reads up:    forgive (overwrite to down), arm bull
}

# --- 2-state, 3-color machine (symbols: 0 = down, 1 = flat, 2 = up) ---------
# Also mirror-symmetric (A<->B, 0<->2). A committed state absorbs flat bars in
# its own direction; a counter-bar is softened to flat and reverses the head.
_RULE_REGIME: _Rule = {
    (0, 2): (2, +1, 0),  # bull reads up:    confirm
    (0, 1): (2, +1, 0),  # bull reads flat:  absorb quiet bar into the trend
    (0, 0): (1, -1, 1),  # bull reads down:  soften to flat, reverse, flip bear
    (1, 0): (0, -1, 1),  # bear reads down:  confirm
    (1, 1): (0, -1, 1),  # bear reads flat:  absorb quiet bar into the trend
    (1, 2): (1, +1, 0),  # bear reads up:    soften to flat, reverse, flip bull
}


def _closes(series: Series) -> np.ndarray:
    return np.asarray(series.closes(), dtype=float)


def _shift1(sig: np.ndarray) -> np.ndarray:
    """Shift a signal forward one bar (decision at t-1 applies to bar t)."""
    out = np.zeros_like(sig)
    if sig.size > 1:
        out[1:] = sig[:-1]
    return np.nan_to_num(out)


def _returns(c: np.ndarray) -> np.ndarray:
    """Per-bar simple returns aligned to bars (r[0] = 0), div-by-zero safe."""
    r = np.zeros_like(c)
    if c.size > 1:
        prev = c[:-1]
        r[1:] = np.where(np.abs(prev) > 1e-12, (c[1:] - prev) / prev, 0.0)
    return np.nan_to_num(r)


def _run_tm(tape: np.ndarray, rule: _Rule, pos: int, state: int,
            steps: int) -> Tuple[float, float, int]:
    """Drive the head `steps` steps over a private copy of `tape`.

    Returns (drift, cycle, terminal_state) where drift = mean attempted signed
    move in [-1, 1] (wall clamps do not cap it) and cycle = state flips per
    step in [0, 1].
    """
    tape = tape.copy()
    w = tape.size
    disp = 0
    flips = 0
    for _ in range(steps):
        write, move, nstate = rule[(state, int(tape[pos]))]
        tape[pos] = write
        disp += move
        if nstate != state:
            flips += 1
        state = nstate
        pos = min(w - 1, max(0, pos + move))
    return disp / steps, flips / steps, state


@dataclass
class TuringDriftStrategy:
    """2-state 2-color TM on the return-sign tape; trade the head's net drift.

    The head starts mid-tape in the majority state (ties broken by the newest
    symbol) and marches with agreeing bars, forgiving a single counter-bar by
    overwriting it. Signal = drift gated by state commitment: persistent drift
    with few state flips is a trend; rapid cycling collapses the weight to 0.
    """
    window: int = 8
    steps_mult: int = 2
    commit_floor: float = 0.25
    name: str = "turing_drift"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"turing_drift({self.window})"
        self.params = {"window": self.window, "steps_mult": self.steps_mult,
                       "commit_floor": self.commit_floor}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        sig = np.zeros(n)
        w = self.window
        if n <= w:
            return sig
        symbols = (_returns(c) > 0).astype(np.int64)  # 1 = up, 0 = down
        steps = max(1, self.steps_mult * w)
        for t in range(w, n):
            tape = symbols[t - w + 1:t + 1]  # oldest -> newest, ends at bar t
            ups = int(tape.sum())
            if 2 * ups > w:
                state = 0
            elif 2 * ups < w:
                state = 1
            else:  # tie: newest bar decides
                state = 0 if tape[-1] == 1 else 1
            drift, cycle, _ = _run_tm(tape, _RULE_DRIFT, w // 2, state, steps)
            commitment = 1.0 - cycle
            gate = (commitment - self.commit_floor) / max(1e-9, 1.0 - self.commit_floor)
            sig[t] = drift * min(1.0, max(0.0, gate))
        return np.clip(np.nan_to_num(_shift1(sig)), -1.0, 1.0)


@dataclass
class TuringRegimeStrategy:
    """2-state 3-color TM on a deadbanded return tape; trade the terminal state.

    Returns quieter than `deadband` x the window's mean |return| are a third
    'flat' color the committed state absorbs. The machine is run twice, once
    from each internal state (initialization-free): if both runs agree the
    regime is committed and the signals add; if they disagree they cancel to
    flat. Direction = terminal state, size = |drift|, gated by state cycling.
    """
    window: int = 10
    deadband: float = 0.35
    steps_mult: int = 2
    cycle_cap: float = 0.5
    name: str = "turing_regime"
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = f"turing_regime({self.window})"
        self.params = {"window": self.window, "deadband": self.deadband,
                       "steps_mult": self.steps_mult, "cycle_cap": self.cycle_cap}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        n = c.size
        sig = np.zeros(n)
        w = self.window
        if n <= w:
            return sig
        r = _returns(c)
        steps = max(1, self.steps_mult * w)
        for t in range(w, n):
            rw = r[t - w + 1:t + 1]  # window of returns ending at bar t
            scale = float(np.mean(np.abs(rw)))
            thr = self.deadband * scale
            tape = np.ones(w, dtype=np.int64)          # 1 = flat
            tape[rw > thr] = 2                          # 2 = up
            tape[rw < -thr] = 0                         # 0 = down
            s = 0.0
            for init_state in (0, 1):
                drift, cycle, term = _run_tm(tape, _RULE_REGIME, w // 2,
                                             init_state, steps)
                direction = 1.0 if term == 0 else -1.0
                gate = min(1.0, max(0.0, 1.0 - cycle / max(1e-9, self.cycle_cap)))
                s += direction * min(1.0, abs(drift)) * gate
            sig[t] = s / 2.0
        return np.clip(np.nan_to_num(_shift1(sig)), -1.0, 1.0)


STRATEGIES = {
    "turing_drift": TuringDriftStrategy,
    "turing_regime": TuringRegimeStrategy,
}
