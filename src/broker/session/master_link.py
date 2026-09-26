"""The session broker's upstream channel to the master.

`send_to_master` is the one place a master refusal becomes fatal: it raises
`MasterRefusedError` unless the caller tolerates the refusal's code.
`LiveStatusPusher` is the other half of the channel, coalescing live-status
changes into full-snapshot pushes.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path
from typing import Protocol

from broker.protocol import client
from broker.protocol.constants import NackCode
from broker.protocol.schemas import (
    LiveStatusPayload,
    NackPayload,
    WireMessage,
    parse_nack,
)

logger = logging.getLogger(__name__)

MASTER_TIMEOUT_S = 10.0
STATUS_RETRY_S = 1.0


def describe_refusal(msg_type: str, nack: NackPayload) -> str:
    """Render one master refusal for the developer."""
    return f"master refused {msg_type}: {nack.error or '(no reason given)'}" + (
        f" [{nack.reason_code}]" if nack.reason_code else ""
    )


class MasterRefusedError(Exception):
    def __init__(self, msg_type: str, nack: NackPayload) -> None:
        """Record the refused message type and the master's refusal."""
        super().__init__(describe_refusal(msg_type, nack))
        self.msg_type = msg_type
        self.nack = nack


class MasterSender(Protocol):
    """The session's bound sender to the master, shaped like `send_to_master`."""

    async def __call__(
        self,
        payload: WireMessage,
        *,
        tolerated: frozenset[NackCode] = ...,
    ) -> NackPayload | None:
        """Send one message and return the tolerated refusal, if any."""
        ...


async def send_to_master(
    master_socket: Path,
    session_id: str,
    payload: WireMessage,
    *,
    tolerated: frozenset[NackCode] = frozenset(),
) -> NackPayload | None:
    """Send one message to the master and wait for its reply.

    Args:
        master_socket: The master's unix socket.
        session_id: This session's name, carried on the envelope.
        payload: The message; its ``MESSAGE_TYPE`` is the envelope type.
        tolerated: Refusal codes the caller handles itself.

    Returns:
        ``None`` once the master accepted the message, otherwise the
        refusal, whose code is in ``tolerated``.

    Raises:
        MasterRefusedError: The master refused the message with a code
            outside ``tolerated``, or with no code.
    """
    resp = await client.send(
        master_socket, payload, session_id=session_id, timeout_s=MASTER_TIMEOUT_S
    )
    if resp.ok:
        return None
    nack = parse_nack(resp)
    if nack.reason_code is not None and nack.reason_code in tolerated:
        return nack
    raise MasterRefusedError(payload.MESSAGE_TYPE, nack)


class LiveStatusPusher:
    def __init__(
        self,
        *,
        snapshot: Callable[[], LiveStatusPayload],
        send: Callable[[LiveStatusPayload], Awaitable[object]],
    ) -> None:
        """Wire the pusher to the session's status and the master.

        Args:
            snapshot: Builds the session's current live status.
            send: Delivers one push to the master; raising means it did not
                arrive.
        """
        self._snapshot = snapshot
        self._send = send
        self._dirty = asyncio.Event()
        self._phrases: set[str] = set()
        self._task: asyncio.Task[None] | None = None

    @property
    def current_activity(self) -> str:
        """The active dashboard phrases joined, ``""`` when idle."""
        return " · ".join(sorted(self._phrases))

    def start(self) -> None:
        """Start the background push task.

        Raises:
            RuntimeError: The pusher has already been started.
        """
        if self._task is not None:
            raise RuntimeError("live-status pusher already started")
        self._task = asyncio.create_task(self._run(), name="live-status")

    def mark_dirty(self) -> None:
        """Schedule a push of the current snapshot."""
        self._dirty.set()

    @contextlib.contextmanager
    def activity(self, phrase: str) -> Generator[None]:
        """Show ``phrase`` on the dashboard for the duration of the block."""
        # A set, not a string: a permission decision on a socket-handler task
        # and a triage on the event loop run concurrently, so each must show
        # and clear exactly its own phrase.
        self._phrases.add(phrase)
        self._dirty.set()
        try:
            yield
        finally:
            self._phrases.discard(phrase)
            self._dirty.set()

    async def aclose(self) -> None:
        """Stop pushing; a change not yet pushed is dropped."""
        task = self._task
        if task is None:
            return
        # Cancellation, not a shutdown flag: a task parked in the dirty wait
        # would never observe one.
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        """Coalesce changes into full-snapshot pushes until cancelled."""
        while True:
            await self._dirty.wait()
            self._dirty.clear()
            try:
                await self._send(self._snapshot())
            except Exception as exc:
                # Master unreachable: re-send the current snapshot next loop;
                # the backoff keeps a dead master from spinning the broker hot.
                logger.warning("live-status push failed, retrying: %r", exc)
                self._dirty.set()
                await asyncio.sleep(STATUS_RETRY_S)
