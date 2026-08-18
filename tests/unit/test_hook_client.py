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

from broker.protocol.constants import HOOK_SETTINGS_TIMEOUT, HOOK_WAIT_SECONDS

EXPECTED_ALLOW = {
    "hookSpecificOutput": {
        "hookEventName": "PermissionRequest",
        "decision": {"behavior": "allow", "message": "broker approved"},
    }
}

PERMISSION_REQUEST_PAYLOAD = {
    "session_id": "sess-1",
    "transcript_path": "/private/tmp/x/t.jsonl",
    "cwd": "/private/tmp/x",
    "hook_event_name": "PermissionRequest",
    "tool_name": "Bash",
    "tool_input": {"command": "rm -rf build", "description": "clean"},
    "permission_mode": "default",
    "permission_suggestions": [
        {"type": "addDirectories", "directories": ["/private/tmp/x"]}
    ],
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

ASK_USER_QUESTION_PAYLOAD = {
    "session_id": "sess-1",
    "transcript_path": "/private/tmp/x/t.jsonl",
    "cwd": "/private/tmp/x",
    "hook_event_name": "PreToolUse",
    "tool_name": "AskUserQuestion",
    "tool_input": {"questions": [{"question": "which one?"}]},
    "tool_use_id": "toolu_ask_1",
    "permission_mode": "default",
}

ASK_UPDATED_INPUT = {
    "questions": [{"question": "which one?"}],
    "answers": {"which one?": "option A"},
}

EXPECTED_ASK_ANSWER = {
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "permissionDecisionReason": "broker answered",
        "updatedInput": ASK_UPDATED_INPUT,
    }
}

STOP_PAYLOAD = {
    "session_id": "sess-1",
    "transcript_path": "/private/tmp/x/t.jsonl",
    "cwd": "/private/tmp/x",
    "hook_event_name": "Stop",
    "last_assistant_message": "done",
}


class StubBroker(socketserver.ThreadingUnixStreamServer):
    """Counts accepted connections, records every envelope, replies per `mode`."""

    daemon_threads = True

    def __init__(self, sock_path: str, mode: str) -> None:
        self.mode = mode
        self.received: list[dict[str, Any]] = []
        self.connections = 0
        self._lock = threading.Lock()
        super().__init__(sock_path, _StubHandler)

    def note_connection(self) -> None:
        """Count one accepted connection, whether or not anything is sent on it."""
        with self._lock:
            self.connections += 1


class _StubHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server = cast(StubBroker, self.server)
        server.note_connection()
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
        elif server.mode == "ask_answer":
            reply = {
                "v": 1,
                "id": envelope["id"],
                "type": "response",
                "ok": True,
                "payload": {
                    "decision": "answer",
                    "updated_input": ASK_UPDATED_INPUT,
                },
            }
        elif server.mode == "ask_escalated":
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


def wait_for_connections(stub: StubBroker, count: int) -> None:
    deadline = time.monotonic() + 2.0
    while stub.connections < count and time.monotonic() < deadline:
        time.sleep(0.01)


def test_pretooluse_other_tool_never_connects(sock_dir: Path) -> None:
    """An ordinary tool's PreToolUse never reaches the socket.

    A zero-connection assertion is only worth as much as the counter behind
    it, so the same stub instance is then handed a payload that must connect:
    the counter is shown able to move before its zero is believed.
    """
    sock_path = sock_dir / "broker.sock"
    with start_stub(sock_path, "allow") as stub:
        start = time.monotonic()
        proc = run_hook(PRE_TOOL_USE_PAYLOAD, sock_path)
        elapsed = time.monotonic() - start
        assert proc.returncode == 0
        assert proc.stdout == ""
        assert elapsed < 1.0
        # A connect racing the hook's exit would still land on the listener,
        # so give the handler thread time to record one before claiming none.
        time.sleep(0.3)
        assert stub.connections == 0
        assert stub.received == []

        armed = run_hook(ASK_USER_QUESTION_PAYLOAD, sock_path)
        assert armed.returncode == 0
        wait_for_connections(stub, 1)
        assert stub.connections == 1


def test_askuserquestion_answer_prints_flat_shape(sock_dir: Path) -> None:
    sock_path = sock_dir / "broker.sock"
    with start_stub(sock_path, "ask_answer") as stub:
        proc = run_hook(ASK_USER_QUESTION_PAYLOAD, sock_path)
        assert proc.returncode == 0
        lines = proc.stdout.splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == EXPECTED_ASK_ANSWER
        assert len(stub.received) == 1
        env = stub.received[0]
        assert env["v"] == 1
        assert env["type"] == "ask_question"
        assert env["session_id"] == "sess-1"
        assert env["payload"] == {
            "tool_input": {"questions": [{"question": "which one?"}]},
            "tool_use_id": "toolu_ask_1",
        }


def test_askuserquestion_escalated_prints_nothing(sock_dir: Path) -> None:
    sock_path = sock_dir / "broker.sock"
    with start_stub(sock_path, "ask_escalated"):
        proc = run_hook(ASK_USER_QUESTION_PAYLOAD, sock_path)
        assert proc.returncode == 0
        assert proc.stdout == ""


def test_askuserquestion_mute_prints_nothing_within_budget(
    sock_dir: Path,
) -> None:
    sock_path = sock_dir / "broker.sock"
    with start_stub(sock_path, "mute"):
        start = time.monotonic()
        proc = run_hook(
            ASK_USER_QUESTION_PAYLOAD, sock_path, timeout_override="0.2"
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0
        assert proc.stdout == ""
        assert elapsed < 2.0


def test_permission_request_allow_prints_nested_shape(sock_dir: Path) -> None:
    sock_path = sock_dir / "broker.sock"
    with start_stub(sock_path, "allow") as stub:
        proc = run_hook(PERMISSION_REQUEST_PAYLOAD, sock_path)
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
        assert env["payload"] == {
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf build", "description": "clean"},
            "cwd": "/private/tmp/x",
            "transcript_path": "/private/tmp/x/t.jsonl",
            "permission_mode": "default",
            "permission_suggestions": [
                {"type": "addDirectories", "directories": ["/private/tmp/x"]}
            ],
        }
        # The event carries no call identity; the hook must not invent one.
        assert "tool_use_id" not in env["payload"]


def test_permission_request_escalated_prints_nothing(sock_dir: Path) -> None:
    sock_path = sock_dir / "broker.sock"
    with start_stub(sock_path, "escalated"):
        proc = run_hook(PERMISSION_REQUEST_PAYLOAD, sock_path)
        assert proc.returncode == 0
        assert proc.stdout == ""


def test_timeout_prints_nothing_exits_zero(sock_dir: Path) -> None:
    sock_path = sock_dir / "broker.sock"
    with start_stub(sock_path, "mute"):
        start = time.monotonic()
        proc = run_hook(
            PERMISSION_REQUEST_PAYLOAD, sock_path, timeout_override="0.2"
        )
        elapsed = time.monotonic() - start
        assert proc.returncode == 0
        assert proc.stdout == ""
        assert elapsed < 2.0


def test_unreachable_socket_exits_zero(sock_dir: Path) -> None:
    proc = run_hook(PERMISSION_REQUEST_PAYLOAD, sock_dir / "nonexistent.sock")
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_unset_socket_noop(sock_dir: Path) -> None:
    """BROKER_SOCKET unset -> exit 0, nothing written to a canary socket."""
    canary_path = sock_dir / "canary.sock"
    with start_stub(canary_path, "allow") as canary:
        proc = run_hook(PERMISSION_REQUEST_PAYLOAD, sock_path=None)
        assert proc.returncode == 0
        assert proc.stdout == ""
        assert canary.received == []


def test_non_pretooluse_event_is_fire_and_forget(sock_dir: Path) -> None:
    sock_path = sock_dir / "broker.sock"
    # "mute" would block a reply-waiting client; a fire-and-forget hook returns
    # immediately, proving no reply is expected for non-decision events.
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


def test_hook_deadline_fires_before_claude_codes() -> None:
    # The hook must hit its own deadline and exit 0 on its own terms. If Claude
    # Code's settings.json timeout fired first it would kill the process
    # mid-wait and the degraded outcome would stop being predictable.
    assert HOOK_WAIT_SECONDS < HOOK_SETTINGS_TIMEOUT
