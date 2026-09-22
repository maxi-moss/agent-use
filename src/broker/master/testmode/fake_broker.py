"""Synthetic peers that speak the real NDJSON protocol.

``FakeBrokerClient`` raises and retracts escalations and permission
escalations over the master socket through the same
``protocol.client.request`` a session broker and its permission module use.
``FakeSessionSocket`` binds a session socket that records every envelope it
receives and ACKs it, standing in for a broker's listening end.
"""

import asyncio
import contextlib
import uuid
from pathlib import Path

from broker.protocol import client
from broker.protocol.constants import (
    SessionState,
    T_DECISION_DELIVERED,
    T_ESCALATION,
    T_ESCALATION_RETRACT,
    T_PERMISSION_ESCALATION,
    T_PERMISSION_RETRACT,
    T_STATUS,
)
from broker.protocol.schemas import (
    DecisionDeliveredPayload,
    Envelope,
    EscalationPayload,
    EscalationRetractPayload,
    PermissionEscalationPayload,
    PermissionRetractPayload,
    Response,
    StatusPayload,
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
        return await self._send(T_ESCALATION, payload)

    async def permission_escalate(
        self, payload: PermissionEscalationPayload
    ) -> Response:
        """Raise a permission escalation and return the master's reply."""
        return await self._send(T_PERMISSION_ESCALATION, payload)

    async def escalation_retract(self, escalation_id: str, reason: str) -> Response:
        """Withdraw a broker escalation and return the master's reply."""
        return await self._send(
            T_ESCALATION_RETRACT,
            EscalationRetractPayload(escalation_id=escalation_id, reason=reason),
        )

    async def permission_retract(self, escalation_id: str, reason: str) -> Response:
        """Withdraw a permission escalation and return the master's reply."""
        return await self._send(
            T_PERMISSION_RETRACT,
            PermissionRetractPayload(escalation_id=escalation_id, reason=reason),
        )

    async def deliver(self, escalation_id: str) -> Response:
        """Confirm a dispatched decision reached the pane."""
        return await self._send(
            T_DECISION_DELIVERED,
            DecisionDeliveredPayload(escalation_id=escalation_id),
        )

    async def _send(
        self,
        msg_type: str,
        payload: (
            EscalationPayload
            | PermissionEscalationPayload
            | EscalationRetractPayload
            | PermissionRetractPayload
            | DecisionDeliveredPayload
        ),
    ) -> Response:
        """Wrap a payload in a fresh-id envelope and send it to the master."""
        env = Envelope(
            id=uuid.uuid4().hex,
            type=msg_type,
            session_id=self._session,
            payload=payload.model_dump(),
        )
        return await client.request(
            self._master_socket, env, timeout_s=self._timeout_s
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
