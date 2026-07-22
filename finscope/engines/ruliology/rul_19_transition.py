"""
rul_19 — Wolfram class-transition detector (ruliology strategy family).

Thesis
------
Wolfram's four behavior classes (1 fixed, 2 periodic, 3 chaotic, 4 complex)
describe a system's dynamics; real market regimes drift *between* classes.
This module tracks a continuous, rolling class estimate of the return-sign
tape and trades the *derivative* of that estimate — never its level:

  * a drop in the class estimate (3 -> 2/1: chaos condensing into order) is
    the moment a regime becomes computationally reducible — an emerging trend.
    ENTER in the direction of the prevailing move at the instant order appears.
  * a rise in the class estimate (1/2 -> 3: periodic order breaking into
    chaos) is a phase change that invalidates whatever structure was being
    ridden. EXIT to flat — there is nothing left to compress.
  * no significant transition: hold (or decay) the standing position; the
    level of the class estimate alone never opens or sizes a trade.

Rolling class proxy (per bar, causal)
-------------------------------------
Over the trailing `window` return signs (up-tick = 1, else 0):

  H  = normalized Shannon entropy of overlapping k-blocks of the tape, in
       [0, 1]  (0 = frozen tape, 1 = i.i.d.-uniform blocks),
  P  = periodicity score: best self-match fraction of the tape over lags
       1..window//2, rescaled so chance-level (0.5) maps to 0, in [0, 1]
       (lag 1 catches class-1 frozen tapes, lag >= 2 catches class-2 cycles),
  X  = H * (1 - P)   -- chaos index in [0, 1]: high entropy that is NOT
       explained by a short cycle is genuine class-3 turbulence,
  C  = 1 + 2 * X     -- continuous class estimate in [1, 3], EMA-smoothed.

(A scalar cannot represent class 4 as a separate point — the *edge* of chaos
lives in the mid-range of C, and transitions across it are exactly what the
derivative picks up.)

Causality: the class estimate at bar t uses returns realized up to and
including the close of bar t only; the finished signal passes through
`_shift1`, so weights[t] depends on closes through bar t-1 exclusively.
Perturbing the final close changes no emitted weight.

numpy + finscope only; safe on series of any length (>= 0 bars).

Registry: `STRATEGIES` (name -> class).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from finscope.core.contracts import Series
from finscope.engines.strategies import _closes, _shift1


# ---- sign-tape statistics ----------------------------------------------------

def _returns_and_bits(closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple returns and their sign tape (up-tick = 1, else 0)."""
    rets = np.diff(closes) / np.where(closes[:-1] != 0.0, closes[:-1], 1.0)
    rets = np.nan_to_num(rets)
    bits = (rets > 0.0).astype(np.int64)
    return rets, bits


def _tape_entropy(tape: np.ndarray, k: int) -> float:
    """Normalized Shannon entropy of overlapping k-blocks of a 0/1 tape.

    Returns a value in [0, 1]: 0 = perfectly ordered, 1 = maximal block
    entropy (all 2**k blocks equally likely).
    """
    m = tape.size - k + 1
    if m <= 0:
        return 0.0
    codes = np.zeros(m, dtype=np.int64)
    for j in range(k):
        codes = (codes << 1) | tape[j:j + m]
    counts = np.bincount(codes, minlength=2 ** k).astype(float)
    p = counts[counts > 0.0] / float(m)
    return float(-(p * np.log2(p)).sum() / k)


def _tape_periodicity(tape: np.ndarray) -> float:
    """Best periodic self-match of the tape over lags 1..len//2, in [0, 1].

    match(p) = fraction of positions where tape[i] == tape[i-p]. A random
    tape matches ~0.5 at any lag, so the score rescales [0.5, 1] -> [0, 1].
    Lag 1 fires on frozen (class-1) tapes; lags >= 2 fire on class-2 cycles.
    """
    w = tape.size
    best = 0.0
    for p in range(1, w // 2 + 1):
        match = float(np.mean(tape[p:] == tape[:-p]))
        if match > best:
            best = match
    return float(np.clip((best - 0.5) * 2.0, 0.0, 1.0))


def _class_estimate(closes: np.ndarray, window: int, block_k: int,
                    ema_alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bar (rets, smoothed class estimate C in [1,3], chaos index X in [0,1]).

    C[t] and X[t] are NaN until `window` returns are available. C[t] uses
    information through the close of bar t only.
    """
    n = closes.size
    cls = np.full(n, np.nan)
    chaos = np.full(n, np.nan)
    if n < window + 1:
        return np.zeros(max(n - 1, 0)), cls, chaos
    rets, bits = _returns_and_bits(closes)
    ema = np.nan
    for t in range(window, n):
        tape = bits[t - window:t]              # signs of returns through bar t
        h = _tape_entropy(tape, block_k)
        p_score = _tape_periodicity(tape)
        x = h * (1.0 - p_score)
        raw = 1.0 + 2.0 * x
        ema = raw if not np.isfinite(ema) else ema_alpha * raw + (1.0 - ema_alpha) * ema
        cls[t] = ema
        chaos[t] = x
    return rets, cls, chaos


# ---- shared transition signal core -------------------------------------------

def _transition_signal(closes: np.ndarray, window: int, block_k: int,
                       ema_alpha: float, lag: int, enter_drop: float,
                       exit_rise: float, mode: str, decay: float) -> np.ndarray:
    """Raw (unshifted) signal: position[t] from class-estimate *changes* only.

    mode 'hold'    — an ordering transition sets the position outright; it is
                     held unchanged until a chaos transition flattens it.
    mode 'impulse' — each ordering transition injects an impulse that decays
                     every quiet bar; a chaos transition clears everything.
    """
    n = closes.size
    sig = np.zeros(n, dtype=float)
    if n < window + lag + 1:
        return sig
    rets, cls, chaos = _class_estimate(closes, window, block_k, ema_alpha)
    pos = 0.0
    for t in range(window + lag, n):
        c_now, c_then = cls[t], cls[t - lag]
        if not (np.isfinite(c_now) and np.isfinite(c_then)):
            sig[t] = pos
            continue
        d = c_now - c_then                      # the derivative — the signal driver
        if d <= -enter_drop:
            # order emerging (toward class 2/1): join the prevailing move
            drift = float(np.sum(rets[t - window:t]))
            direction = float(np.sign(drift))
            if direction != 0.0:
                strength = float(np.clip(abs(d) / 0.8, 0.0, 1.0))
                size = (0.35 + 0.65 * strength) * (1.0 - 0.5 * float(chaos[t]))
                if mode == "hold":
                    pos = float(np.clip(direction * size, -1.0, 1.0))
                else:  # impulse: accumulate evidence of ordering
                    pos = float(np.clip(pos + direction * size, -1.0, 1.0))
        elif d >= exit_rise:
            # order breaking into chaos (toward class 3): regime invalidated
            pos = 0.0
        elif mode == "impulse":
            pos *= decay                        # quiet bar: evidence fades
        sig[t] = pos
    return sig


def _finalize(sig: np.ndarray) -> np.ndarray:
    """Sanitize, clip and shift a raw signal into a causal weight vector."""
    return np.clip(np.nan_to_num(_shift1(np.nan_to_num(sig))), -1.0, 1.0)


# ---- strategies --------------------------------------------------------------

@dataclass
class ClassTransitionStrategy:
    """Trade Wolfram class-estimate *transitions*, hold between them.

    A significant drop of the smoothed class estimate over `lag` bars
    (chaos -> order) opens a position with the prevailing drift, sized by the
    transition magnitude and how ordered the tape now is; a significant rise
    (order -> chaos) exits to flat. Between transitions the position is held
    untouched — the class level itself never trades.
    """
    window: int = 10
    lag: int = 3
    enter_drop: float = 0.22
    exit_rise: float = 0.22
    block_k: int = 2
    ema_alpha: float = 0.55
    name: str = "class_transition"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"class_transition(w={self.window},lag={self.lag})"
        self.params = {"window": self.window, "lag": self.lag,
                       "enter_drop": self.enter_drop, "exit_rise": self.exit_rise,
                       "block_k": self.block_k, "ema_alpha": self.ema_alpha}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        sig = _transition_signal(c, self.window, self.block_k, self.ema_alpha,
                                 self.lag, self.enter_drop, self.exit_rise,
                                 mode="hold", decay=1.0)
        return _finalize(sig)


@dataclass
class ClassTransitionImpulseStrategy:
    """Impulse-decay variant: exposure = accumulated ordering transitions.

    Every chaos->order transition injects a signed impulse (direction from the
    prevailing drift, magnitude from the transition size); the accumulator
    decays each quiet bar, so exposure persists only while ordering evidence
    keeps arriving. Any order->chaos transition clears the book instantly.
    A pure derivative integrator — with no transitions the position bleeds
    to flat on its own.
    """
    window: int = 10
    lag: int = 2
    enter_drop: float = 0.18
    exit_rise: float = 0.26
    decay: float = 0.88
    block_k: int = 2
    ema_alpha: float = 0.55
    name: str = "class_transition_impulse"
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = f"class_transition_impulse(w={self.window},lag={self.lag},d={self.decay})"
        self.params = {"window": self.window, "lag": self.lag,
                       "enter_drop": self.enter_drop, "exit_rise": self.exit_rise,
                       "decay": self.decay, "block_k": self.block_k,
                       "ema_alpha": self.ema_alpha}

    def weights(self, series: Series) -> np.ndarray:
        c = _closes(series)
        sig = _transition_signal(c, self.window, self.block_k, self.ema_alpha,
                                 self.lag, self.enter_drop, self.exit_rise,
                                 mode="impulse", decay=self.decay)
        return _finalize(sig)


STRATEGIES = {
    "class_transition": ClassTransitionStrategy,
    "class_transition_impulse": ClassTransitionImpulseStrategy,
}
