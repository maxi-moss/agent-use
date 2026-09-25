"""Synthetic peers that speak the real NDJSON protocol.

``FakeBrokerClient`` raises and retracts decision and pane escalations over
the master socket through ``protocol.client.send``.
``FakeSessionSocket`` binds a session socket that records every envelope it
receives and ACKs it, standing in for a broker's listening end.
"""

import asyncio
import contextlib
from pathlib import Path

from broker.protocol import client
from broker.protocol.constants import SessionState, T_STATUS
from broker.protocol.schemas import (
    DecisionDeliveredPayload,
    Envelope,
    EscalationPayload,
    EscalationRetractPayload,
    PaneEscalationPayload,
    PaneRetractPayload,
    Response,
    StatusPayload,
    WireMessage,
)
from broker.protocol.server import serve_unix


class FakeBrokerClient:
    """A session broker's upward end: real envelopes over the master socket."""

    def __init__(
        self, master_socket: Path, session: str, *, timeout_s: float = 5.0
    ) -> None:
        """Bind the client to one master socket and one session identity."""
        self._master_socket = master_socket
        self._session = session
        self._timeout_s = timeout_s

    async def escalate(self, payload: EscalationPayload) -> Response:
        """Raise a broker escalation and return the master's reply."""
        return await self._send(payload)

    async def pane_escalate(self, payload: PaneEscalationPayload) -> Response:
        """Raise a pane escalation and return the master's reply."""
        return await self._send(payload)

    async def escalation_retract(self, escalation_id: str, reason: str) -> Response:
        """Withdraw a broker escalation and return the master's reply."""
        return await self._send(
            EscalationRetractPayload(escalation_id=escalation_id, reason=reason)
        )

    async def pane_retract(self, escalation_id: str, reason: str) -> Response:
        """Withdraw a pane escalation and return the master's reply."""
        return await self._send(
            PaneRetractPayload(escalation_id=escalation_id, reason=reason)
        )

    async def deliver(self, escalation_id: str) -> Response:
        """Confirm a dispatched decision reached the pane."""
        return await self._send(DecisionDeliveredPayload(escalation_id=escalation_id))

    async def _send(self, payload: WireMessage) -> Response:
        """Send one message to the master as this session."""
        return await client.send(
            self._master_socket,
            payload,
            session_id=self._session,
            timeout_s=self._timeout_s,
        )


class FakeSessionSocket:
    """A recording session-socket stub behind a real ``serve_unix``.

    It ACKs every envelope and answers a status probe with a driving session,
    so a listening instance reads as alive; closing it frees the socket.
    """

    def __init__(self) -> None:
        """Start with no bound server and an empty receive log."""
        self.received: list[Envelope] = []
        self._server: asyncio.Server | None = None

    async def start(self, path: Path) -> None:
        """Bind and start serving on ``path``."""
        self._server = await serve_unix(path, self._handle)

    async def _handle(self, env: Envelope) -> Response:
        """Record the envelope and ACK it, answering a status probe."""
        self.received.append(env)
        if env.type == T_STATUS:
            return Response(
                id=env.id,
                ok=True,
                payload=StatusPayload(state=SessionState.DRIVING).model_dump(),
            )
        return Response(id=env.id, ok=True)

    async def stop(self) -> None:
        """Close the server and release the socket path."""
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(OSError, ConnectionError):
                await self._server.wait_closed()
            self._server = None
