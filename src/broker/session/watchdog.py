"""Watchdog: reconciliation backstop for lost hook events.

Hook events are the trigger; the watchdog only converts a lost event from a
permanent hang into a bounded delay. Herdr state GATES the reconciliation read
(idle/blocked only) — it never classifies.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class Watchdog:
    def __init__(
        self,
        deadline_s: float,
        herdr_state: Callable[[], str],
        reconcile: Callable[[], Awaitable[None]],
    ) -> None:
        """Arm the watchdog against ``deadline_s`` of hook-event silence.

        Args:
            deadline_s: Seconds without a hook event before the deadline
                expires.
            herdr_state: Returns the session's current Herdr state.
            reconcile: Runs the sanctioned pure-transcript reconciliation read.
        """
        self._deadline_s = deadline_s
        self._herdr_state = herdr_state
        self._reconcile = reconcile
        self._last_reset = time.monotonic()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the background deadline task.

        Raises:
            RuntimeError: The watchdog has already been started.
        """
        if self._task is not None:
            raise RuntimeError("watchdog already started")
        self._task = asyncio.create_task(self._run(), name="watchdog")

    def reset(self) -> None:
        """Slide the deadline forward; called on every hook event."""
        self._last_reset = time.monotonic()

    async def stop(self) -> None:
        """Cancel the deadline task and await it; a no-op if not started."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        """Sleep to the deadline forever, reconciling on each silent expiry.

        Only ``idle``/``blocked`` Herdr state triggers a reconciliation read;
        the deadline re-arms either way, turning a lost hook event into a
        bounded delay rather than a permanent hang.
        """
        while True:
            remaining = self._last_reset + self._deadline_s - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(remaining)
                continue
            # Deadline expired with no hook event. Gate on Herdr state:
            # only idle/blocked warrant the sanctioned pure-transcript read.
            state = await asyncio.to_thread(self._herdr_state)
            if state in {"idle", "blocked"}:
                await self._reconcile()
            else:
                logger.debug("watchdog expiry gated out (state=%s)", state)
            self.reset()  # re-arm either way
