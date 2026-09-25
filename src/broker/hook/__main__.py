"""Claude Code hook client. `python -m broker.hook`.

Import closure: stdlib + broker.protocol.constants ONLY. This process starts
on the synchronous permission path of every tool call in every supervised
session — pydantic here would tax every single tool call.

Invariants (binding):
- stdout carries the PermissionRequest allow-decision JSON, the PreToolUse
  AskUserQuestion answer JSON, or NOTHING. Never "deny", never
  allow-by-default, never an answer the broker did not supply.
- exit 0 on every path, including every exception. A dead broker degrades the
  session to stock Claude Code; it never breaks one.
- BROKER_SOCKET unset -> immediate silent no-op (the isolation gate).
"""

import json
import os
import socket
import sys
import traceback
import uuid
from typing import Any, cast


def _record_failure(text: str) -> None:
    """Append a hook failure to the log file named by ``BROKER_HOOK_LOG``.

    Args:
        text: The failure text to append.
    """
    # Spelled as a literal, not the constant: this must still work when
    # broker.protocol.constants itself is what failed to import.
    log_path = os.environ.get("BROKER_HOOK_LOG")
    if not log_path:
        return
    try:
        with open(
            log_path, "a", encoding="utf-8", errors="backslashreplace"
        ) as log_file:
            log_file.write(text)
    except OSError:
        pass


try:
    from broker.protocol.constants import (
        ASK_DECISION_ANSWER,
        ASK_USER_QUESTION,
        DECISION_ALLOW,
        ENV_BROKER_SOCKET,
        HOOK_WAIT_SECONDS,
        HookEventName,
        MAX_LINE_BYTES,
        T_ASK_QUESTION,
        T_HOOK_EVENT,
        T_PERMISSION_REQUEST,
    )
except Exception:
    # Exit-0 degradation is for the hook entrypoint; an importer (the closure
    # tests) must still see the failure.
    if __name__ != "__main__":
        raise
    _record_failure(traceback.format_exc())
    sys.exit(0)

# PermissionRequest's own nested shape. Claude Code validates it and treats
# PreToolUse's flat permissionDecision shape here as if nothing was printed.
_ALLOW_OUTPUT = {
    "hookSpecificOutput": {
        "hookEventName": HookEventName.PERMISSION_REQUEST,
        "decision": {"behavior": "allow", "message": "broker approved"},
    }
}


def _timeout_seconds() -> float:
    """Return the wait in seconds, overridable via ``BROKER_HOOK_TIMEOUT``."""
    try:
        return float(os.environ["BROKER_HOOK_TIMEOUT"])
    except (KeyError, ValueError):
        return float(HOOK_WAIT_SECONDS)


def _read_line(sock: socket.socket, timeout: float) -> bytes | None:
    """Read one \\n-terminated reply line, capped at MAX_LINE_BYTES.

    Args:
        sock: Connected socket to read from.
        timeout: Per-recv socket timeout in seconds.

    Returns:
        The line without its terminator, or ``None`` if closed early or capped.
    """
    sock.settimeout(timeout)
    buf = b""
    while b"\n" not in buf:
        if len(buf) > MAX_LINE_BYTES:
            return None
        chunk = sock.recv(65536)
        if not chunk:
            return None
        buf += chunk
    return buf.split(b"\n", 1)[0]


def main() -> None:
    """Relay one hook event to the broker and print its allow-decision, if any."""
    raw_payload: Any = json.load(sys.stdin)
    if not isinstance(raw_payload, dict):
        return
    payload = cast(dict[str, Any], raw_payload)
    sock_path = os.environ.get(ENV_BROKER_SOCKET)
    if not sock_path:
        return  # isolation gate

    event = payload.get("hook_event_name")
    if (
        event == HookEventName.PRE_TOOL_USE
        and payload.get("tool_name") != ASK_USER_QUESTION
    ):
        # PreToolUse fires on every tool call. Returning here — above the
        # socket — is what keeps ordinary tool calls free of any broker cost:
        # no connection, no wait. The decision path is PermissionRequest.
        return

    timeout = _timeout_seconds()
    blocking = event in (HookEventName.PERMISSION_REQUEST, HookEventName.PRE_TOOL_USE)

    if event == HookEventName.PERMISSION_REQUEST:
        envelope = {
            "id": uuid.uuid4().hex,
            "type": T_PERMISSION_REQUEST,
            "payload": {
                "tool_name": payload.get("tool_name", ""),
                "tool_input": payload.get("tool_input", {}),
                # Forwarded verbatim: this process is stdlib-only, so the
                # broker owns validating the arms.
                "permission_suggestions": payload.get(
                    "permission_suggestions", []
                ),
            },
        }
    elif event == HookEventName.PRE_TOOL_USE:
        envelope = {
            "id": uuid.uuid4().hex,
            "type": T_ASK_QUESTION,
            "payload": {
                "tool_input": payload.get("tool_input", {}),
                "tool_use_id": payload.get("tool_use_id", ""),
            },
        }
    else:
        envelope = {
            "id": uuid.uuid4().hex,
            "type": T_HOOK_EVENT,
            "payload": {"hook_event_name": event, "raw": payload},
        }

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(sock_path)
        sock.sendall(json.dumps(envelope).encode() + b"\n")
        if not blocking:
            return  # fire-and-forget

        line = _read_line(sock, timeout)
        if line is None:
            _record_failure(f"{event}: reply closed early or over the line cap\n")
            return
        raw_reply: Any = json.loads(line)
        if not isinstance(raw_reply, dict):
            _record_failure(f"{event}: reply is not a JSON object\n")
            return
        raw_decision: Any = cast(dict[str, Any], raw_reply).get("payload")
        if not isinstance(raw_decision, dict):
            _record_failure(f"{event}: reply payload is not a JSON object\n")
            return
        decision_payload = cast(dict[str, Any], raw_decision)

        if event == HookEventName.PERMISSION_REQUEST:
            if decision_payload.get("decision") == DECISION_ALLOW:
                # The sanctioned stdout write for permissions. Anything but an
                # explicit allow (escalated / malformed / timeout) prints
                # nothing -> native flow.
                print(json.dumps(_ALLOW_OUTPUT))
            return

        # PreToolUse / AskUserQuestion: print the broker's answers, or nothing.
        if decision_payload.get("decision") != ASK_DECISION_ANSWER:
            return
        updated: Any = decision_payload.get("updated_input")
        if not isinstance(updated, dict):
            _record_failure(f"{event}: answer carries no updated_input object\n")
            return
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": HookEventName.PRE_TOOL_USE,
                        "permissionDecision": "allow",
                        "permissionDecisionReason": "broker answered",
                        "updatedInput": updated,
                    }
                }
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # ALWAYS — degradation, never breakage.
        _record_failure(traceback.format_exc())
    sys.exit(0)
