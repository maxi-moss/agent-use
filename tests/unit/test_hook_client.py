"""Hook client proven against a stub broker socket.

Real subprocess (`sys.executable -m broker.hook`) against an in-test
socketserver stub. Sockets live under /private/tmp — never /tmp.
"""

import contextlib
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

EXPECTED_ALLOW = {
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "permissionDecisionReason": "broker approved",
    }
}

PRE_TOOL_USE_PAYLOAD = {
    "session_id": "sess-1",
    "transcript_path": "/private/tmp/x/t.jsonl",
    "cwd": "/private/tmp/x",
    "hook_event_name": "PreToolUse",
    "tool_name": "Bash",
    "tool_input": {"command": "ls", "description": "list"},
    "tool_use_id": "toolu_test_1",
    "permission_mode": "default",
}

STOP_PAYLOAD = {
    "session_id": "sess-1",
    "transcript_path": "/private/tmp/x/t.jsonl",
    "cwd": "/private/tmp/x",
    "hook_event_name": "Stop",
    "last_assistant_message": "done",
}


class StubBroker(socketserver.ThreadingUnixStreamServer):
    """Records every envelope; replies according to `mode`."""

    daemon_threads = True

    def __init__(self, sock_path: str, mode: str) -> None:
        self.mode = mode
        self.received: list[dict[str, Any]] = []
        super().__init__(sock_path, _StubHandler)


class _StubHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = cast(StubBroker, self.server)
        line = self.rfile.readline()
        if not line:
            return
        envelope = cast(dict[str, Any], json.loads(line))
        server.received.append(envelope)
        if server.mode == "allow":
            reply = {
                "v": 1,
                "id": envelope["id"],
                "type": "response",
                "ok": True,
                "payload": {"decision": "allow"},
            }
        elif server.mode == "escalated":
            reply = {
                "v": 1,
                "id": envelope["id"],
                "type": "response",
                "ok": True,
                "payload": {"decision": "escalated"},
            }
        else:  # "mute": hold the connection open, never reply
            time.sleep(1.0)
            return
        self.wfile.write(json.dumps(reply).encode() + b"\n")


@pytest.fixture
def sock_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
        yield Path(d)


@contextlib.contextmanager
def start_stub(sock_path: Path, mode: str) -> Generator[StubBroker]:
    server = StubBroker(str(sock_path), mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def run_hook(
    payload: dict[str, Any],
    sock_path: Path | None,
    timeout_override: str | None = "5",
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("BROKER_SOCKET", None)
    if sock_path is not None:
        env["BROKER_SOCKET"] = str(sock_path)
    if timeout_override is not None:
        env["BROKER_HOOK_TIMEOUT"] = timeout_override
    return subprocess.run(
        [sys.executable, "-m", "broker.hook"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_allow_prints_exact_decision_json(sock_dir: Path) -> None:
    sock_path = sock_dir / "broker.sock"
    with start_stub(sock_path, "allow") as stub:
        proc = run_hook(PRE_TOOL_USE_PAYLOAD, sock_path)
        assert proc.returncode == 0
        lines = proc.stdout.splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == EXPECTED_ALLOW
        # envelope that reached the broker
        assert len(stub.received) == 1
        env = stub.received[0]
        assert env["v"] == 1
        assert env["type"] == "permission_request"
        assert env["session_id"] == "sess-1"
        assert env["payload"]["tool_name"] == "Bash"
        assert env["payload"]["tool_use_id"] == "toolu_test_1"
        assert env["payload"]["cwd"] == "/private/tmp/x"
        assert env["payload"]["transcript_path"] == "/private/tmp/x/t.jsonl"
        assert env["payload"]["permission_mode"] == "default"


def test_escalated_prints_nothing(sock_dir: Path) -> None:
    sock_path = sock_dir / "broker.sock"
    with start_stub(sock_path, "escalated"):
        proc = run_hook(PRE_TOOL_USE_PAYLOAD, sock_path)
        assert proc.returncode == 0
        assert proc.stdout == ""


def test_timeout_prints_nothing_exits_zero(sock_dir: Path) -> None:
    sock_path = sock_dir / "broker.sock"
    with start_stub(sock_path, "mute"):
        start = time.monotonic()
        proc = run_hook(PRE_TOOL_USE_PAYLOAD, sock_path, timeout_override="0.2")
        elapsed = time.monotonic() - start
        assert proc.returncode == 0
        assert proc.stdout == ""
        assert elapsed < 2.0


def test_unreachable_socket_exits_zero(sock_dir: Path) -> None:
    proc = run_hook(PRE_TOOL_USE_PAYLOAD, sock_dir / "nonexistent.sock")
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_unset_socket_noop(sock_dir: Path) -> None:
    """BROKER_SOCKET unset -> exit 0, nothing written to a canary socket."""
    canary_path = sock_dir / "canary.sock"
    with start_stub(canary_path, "allow") as canary:
        proc = run_hook(PRE_TOOL_USE_PAYLOAD, sock_path=None)
        assert proc.returncode == 0
        assert proc.stdout == ""
        assert canary.received == []


def test_non_pretooluse_event_is_fire_and_forget(sock_dir: Path) -> None:
    sock_path = sock_dir / "broker.sock"
    # "mute" would block a reply-waiting client; a fire-and-forget hook returns
    # immediately, proving no reply is expected for non-PreToolUse events.
    with start_stub(sock_path, "mute") as stub:
        start = time.monotonic()
        proc = run_hook(STOP_PAYLOAD, sock_path)
        elapsed = time.monotonic() - start
        assert proc.returncode == 0
        assert proc.stdout == ""
        assert elapsed < 1.0  # did not wait for the stub's 1s hold
        deadline = time.monotonic() + 2.0
        while not stub.received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(stub.received) == 1
        env = stub.received[0]
        assert env["type"] == "hook_event"
        assert env["payload"]["hook_event_name"] == "Stop"
        assert env["payload"]["raw"] == STOP_PAYLOAD
