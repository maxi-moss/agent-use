"""serve_unix + client against real sockets under /private/tmp."""

import asyncio
import tempfile
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from broker.protocol import client
from broker.protocol.constants import T_STATUS
from broker.protocol.schemas import Envelope, Response
from broker.protocol.server import serve_unix


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
        yield Path(d)


def make_envelope(payload_text: str = "hi") -> Envelope:
    return Envelope(
        id=uuid.uuid4().hex,
        type=T_STATUS,
        session_id="s1",
        payload={"text": payload_text},
    )


class RecordingHandler:
    def __init__(self) -> None:
        self.received: list[Envelope] = []

    async def __call__(self, env: Envelope) -> Response | None:
        self.received.append(env)
        return Response(id=env.id, ok=True, payload={"echo": env.payload})


@pytest.fixture
async def echo_server(sock_dir: Path) -> AsyncIterator[tuple[Path, RecordingHandler]]:
    handler = RecordingHandler()
    sock_path = sock_dir / "s.sock"
    server = await serve_unix(sock_path, handler, limit=4096)
    try:
        yield sock_path, handler
    finally:
        server.close()
        await server.wait_closed()


async def test_envelope_round_trip(
    echo_server: tuple[Path, RecordingHandler],
) -> None:
    sock_path, handler = echo_server
    env = make_envelope()
    resp = await client.request(sock_path, env, timeout_s=5.0)
    assert resp.ok is True
    assert resp.id == env.id
    assert resp.payload == {"echo": {"text": "hi"}}
    assert handler.received == [env]


async def test_oversized_line_closes_connection_not_server(
    echo_server: tuple[Path, RecordingHandler],
) -> None:
    sock_path, handler = echo_server
    reader, writer = await asyncio.open_unix_connection(str(sock_path))
    writer.write(b"x" * 8192 + b"\n")  # over the 4096 limit
    await writer.drain()
    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
    assert line == b""  # connection closed on us, no reply
    writer.close()
    # The SERVER is still alive: a normal request succeeds afterwards.
    resp = await client.request(sock_path, make_envelope("after"), timeout_s=5.0)
    assert resp.ok is True
    assert len(handler.received) == 1


async def test_rebind_unlinks_stale_socket(sock_dir: Path) -> None:
    sock_path = sock_dir / "s.sock"
    sock_path.touch()  # stale leftover from an unclean exit
    handler = RecordingHandler()
    server = await serve_unix(sock_path, handler)
    try:
        resp = await client.request(sock_path, make_envelope(), timeout_s=5.0)
        assert resp.ok is True
    finally:
        server.close()
        await server.wait_closed()


async def test_request_timeout_raises(sock_dir: Path) -> None:
    class NeverReplies:
        async def __call__(self, env: Envelope) -> Response | None:
            # Long enough to outlive the client timeout, short enough that
            # Server.wait_closed() (which waits for handlers) stays fast.
            await asyncio.sleep(1.0)
            return None

    sock_path = sock_dir / "s.sock"
    server = await serve_unix(sock_path, NeverReplies())
    try:
        with pytest.raises(TimeoutError):
            await client.request(sock_path, make_envelope(), timeout_s=0.2)
    finally:
        server.close()
        await server.wait_closed()
