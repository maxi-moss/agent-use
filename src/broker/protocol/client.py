"""NDJSON unix-domain-socket client helpers. Brokers only.

Every request carries an explicit timeout; timeouts and connection failures
propagate — the caller decides how to fail loud.
"""

import asyncio
import contextlib
from pathlib import Path

from broker.protocol.constants import MAX_LINE_BYTES
from broker.protocol.schemas import Envelope, Response


async def request(path: Path, env: Envelope, *, timeout_s: float) -> Response:
    """Send one envelope and wait for the one reply line.

    Args:
        path: Unix socket to connect to.
        env: Envelope written as one NDJSON line.
        timeout_s: Hard deadline for the whole exchange.

    Returns:
        The peer's decoded reply.

    Raises:
        ConnectionError: The peer closed without sending a reply line.
    """
    async with asyncio.timeout(timeout_s):
        reader, writer = await asyncio.open_unix_connection(
            str(path), limit=MAX_LINE_BYTES
        )
        try:
            writer.write(env.model_dump_json().encode() + b"\n")
            await writer.drain()
            line = await reader.readline()
        finally:
            writer.close()
            with contextlib.suppress(OSError, ConnectionError):
                await writer.wait_closed()
    if not line:
        raise ConnectionError(f"no reply from {path}")
    return Response.model_validate_json(line)


async def notify(path: Path, env: Envelope, *, timeout_s: float = 5.0) -> None:
    """Send one envelope without reading a reply.

    Args:
        path: Unix socket to connect to.
        env: Envelope written as one NDJSON line.
        timeout_s: Hard deadline covering connect and write together.
    """
    async with asyncio.timeout(timeout_s):
        _, writer = await asyncio.open_unix_connection(
            str(path), limit=MAX_LINE_BYTES
        )
        try:
            writer.write(env.model_dump_json().encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            with contextlib.suppress(OSError, ConnectionError):
                await writer.wait_closed()
