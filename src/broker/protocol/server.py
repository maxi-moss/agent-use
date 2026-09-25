"""NDJSON unix-domain-socket server. Brokers only.

Rules encoded here:
- Stale socket files are unlinked before bind (covers SIGKILL leftovers).
  Python 3.14 silently rebinds over a LIVE listener too — this module binds
  each path exactly once per process, so that hazard cannot arise here.
- An oversized line closes THAT CONNECTION, never the server.
- Socket handlers do no slow work: validate, dispatch, reply. Anything slow is
  the handler's job to enqueue.
"""

import asyncio
import contextlib
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import ValidationError

from broker.protocol.constants import MAX_LINE_BYTES
from broker.protocol.schemas import Envelope, Response

logger = logging.getLogger(__name__)

Handler = Callable[[Envelope], Awaitable[Response | None]]

# Every sender writes immediately after connecting, so this must exceed only
# that write latency, not a full request/reply round trip.
READ_TIMEOUT_S = 5.0


async def serve_unix(
    path: Path,
    handler: Handler,
    *,
    limit: int = MAX_LINE_BYTES,
    read_timeout_s: float = READ_TIMEOUT_S,
) -> asyncio.Server:
    """Bind an NDJSON line server on ``path`` and start accepting.

    Args:
        path: Unix socket path to bind. Parent directories are created as
            needed.
        handler: Called with each validated envelope; a returned response is
            written back as one line, ``None`` sends nothing.
        limit: Maximum bytes the stream reader buffers for a single line.
        read_timeout_s: Deadline for each line read; a peer that misses it is
            closed.

    Returns:
        The listening server.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        path.unlink()  # stale file from an unclean exit; free and safe
    server = await asyncio.start_unix_server(
        lambda reader, writer: _serve_connection(
            handler, reader, writer, read_timeout_s
        ),
        path=str(path),
        limit=limit,
    )
    os.chmod(path, 0o600)
    return server


async def _serve_connection(
    handler: Handler,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    read_timeout_s: float,
) -> None:
    """Read and dispatch envelopes from one connection until the peer closes.

    Args:
        handler: Called with each validated envelope; a returned response is
            written back as one line, ``None`` sends nothing.
        reader: Read stream for the accepted connection.
        writer: Write stream for the accepted connection.
        read_timeout_s: Deadline for each line read; a peer that misses it is
            closed.
    """
    try:
        while True:
            try:
                async with asyncio.timeout(read_timeout_s):
                    line = await reader.readline()
            except TimeoutError:
                logger.warning(
                    "no line within %.1fs; closing connection", read_timeout_s
                )
                return
            except (asyncio.LimitOverrunError, ValueError):
                # Oversized line: close this connection, keep serving.
                logger.warning("oversized line; closing connection")
                return
            if not line:
                return  # peer closed
            if not line.strip():
                continue
            try:
                envelope = Envelope.model_validate_json(line)
            except ValidationError:
                logger.warning("invalid envelope; closing connection")
                return
            response = await handler(envelope)
            if response is not None:
                writer.write(response.model_dump_json().encode() + b"\n")
                await writer.drain()
    finally:
        writer.close()
        with contextlib.suppress(OSError, ConnectionError):
            await writer.wait_closed()
