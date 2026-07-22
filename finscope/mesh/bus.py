"""
SignalBus — publish/subscribe over an in-proc backend (default) or NATS (opt-in).

    bus = get_bus()                      # InProc unless NATS_URL + nats-py present
    bus.subscribe("finscope.signal.>", handler)
    bus.publish(Subjects.signal("native"), msg.to_dict())

The in-proc backend supports NATS-style trailing-wildcard subjects (`a.b.>` and
`a.*.c`) so subscription code is identical across backends. The NATS backend is
lazily imported; if `nats-py` isn't installed or no server is reachable, `get_bus`
transparently returns the in-proc bus.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List

Handler = Callable[[str, dict], None]


def _match(pattern: str, subject: str) -> bool:
    """NATS subject matching: '*' matches exactly one token, '>' matches one or
    more trailing tokens (rest of the subject)."""
    pt = pattern.split(".")
    st = subject.split(".")
    for i, p in enumerate(pt):
        if p == ">":
            return len(st) > i          # '>' needs at least one remaining token
        if i >= len(st) or (p != "*" and p != st[i]):
            return False
    return len(st) == len(pt)


class InProcBus:
    backend = "inproc"

    def __init__(self):
        self._subs: List[tuple] = []   # (pattern, handler)
        self._log: List[tuple] = []    # (subject, payload) for replay/inspection

    def publish(self, subject: str, payload: dict) -> None:
        self._log.append((subject, payload))
        for pattern, handler in list(self._subs):
            if _match(pattern, subject):
                try:
                    handler(subject, payload)
                except Exception:
                    pass  # a bad subscriber must not break the bus

    def subscribe(self, pattern: str, handler: Handler) -> None:
        self._subs.append((pattern, handler))

    def history(self, pattern: str = ">") -> List[tuple]:
        return [(s, p) for s, p in self._log if _match(pattern, s)]

    def close(self) -> None:
        self._subs.clear()


class NatsBus:  # pragma: no cover - requires a running NATS server
    """Thin NATS wrapper. Activated by get_bus() only when nats-py + NATS_URL are
    present. Uses core NATS for the hot path; JetStream can be layered for
    durability. Kept minimal on purpose."""
    backend = "nats"

    def __init__(self, url: str):
        import asyncio
        import nats  # type: ignore
        self._nats = nats
        self._asyncio = asyncio
        self._url = url
        self._nc = None
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._connect())

    async def _connect(self):
        self._nc = await self._nats.connect(self._url)

    def publish(self, subject: str, payload: dict) -> None:
        self._loop.run_until_complete(
            self._nc.publish(subject, json.dumps(payload).encode()))

    def subscribe(self, pattern: str, handler: Handler) -> None:
        async def _cb(msg):
            handler(msg.subject, json.loads(msg.data.decode()))
        self._loop.run_until_complete(self._nc.subscribe(pattern, cb=_cb))

    def close(self) -> None:
        if self._nc:
            self._loop.run_until_complete(self._nc.close())


class SignalBus:
    """Facade selecting a backend."""
    def __init__(self, backend=None):
        self._b = backend or InProcBus()

    @property
    def backend(self) -> str:
        return self._b.backend

    def publish(self, subject: str, payload: dict) -> None:
        self._b.publish(subject, payload)

    def subscribe(self, pattern: str, handler: Handler) -> None:
        self._b.subscribe(pattern, handler)

    def history(self, pattern: str = ">"):
        return getattr(self._b, "history", lambda p=">": [])(pattern)

    def close(self) -> None:
        self._b.close()


def get_bus(prefer_nats: bool = True) -> SignalBus:
    url = os.environ.get("NATS_URL")
    if prefer_nats and url:
        try:
            return SignalBus(NatsBus(url))
        except Exception:
            pass  # fall back to in-proc
    return SignalBus(InProcBus())
