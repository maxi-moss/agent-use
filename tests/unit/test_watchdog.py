"""Watchdog: fires only when stale AND Herdr says idle/blocked; reset defers."""

import asyncio

from broker.herdr.schemas import AgentStatus
from broker.session.watchdog import Watchdog


class Reconciler:
    def __init__(self) -> None:
        self.count = 0

    async def __call__(self) -> None:
        self.count += 1


async def test_fires_only_when_idle_and_stale() -> None:
    reconcile = Reconciler()
    state: AgentStatus = "working"
    dog = Watchdog(0.05, lambda: state, reconcile)
    dog.start()
    try:
        await asyncio.sleep(0.15)
        assert reconcile.count == 0  # working state gates the read out
        state = "idle"
        await asyncio.sleep(0.15)
        assert reconcile.count >= 1  # idle + stale -> reconcile
    finally:
        await dog.stop()


async def test_blocked_state_also_fires() -> None:
    reconcile = Reconciler()
    dog = Watchdog(0.05, lambda: "blocked", reconcile)
    dog.start()
    try:
        await asyncio.sleep(0.15)
        assert reconcile.count >= 1
    finally:
        await dog.stop()


async def test_reset_defers_firing() -> None:
    reconcile = Reconciler()
    dog = Watchdog(0.1, lambda: "idle", reconcile)
    dog.start()
    try:
        for _ in range(5):
            await asyncio.sleep(0.03)
            dog.reset()  # hook events keep arriving -> never fires
        assert reconcile.count == 0
    finally:
        await dog.stop()


async def test_stop_cancels_cleanly() -> None:
    dog = Watchdog(10.0, lambda: "idle", Reconciler())
    dog.start()
    await dog.stop()
    await dog.stop()  # idempotent
