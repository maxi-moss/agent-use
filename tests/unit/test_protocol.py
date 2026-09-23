"""Protocol package tests: stdlib-only constants, envelope round-trip, Literal sync."""

import subprocess
import sys
import typing
from typing import Any

import pytest
from pydantic import ValidationError

from broker.protocol import constants
from broker.protocol.constants import SessionState
from broker.protocol.schemas import (
    AddDirectoriesSuggestion,
    PermissionDecisionPayload,
    PermissionEscalationPayload,
    PermissionRequestPayload,
    StatusPayload,
)

COMPLETE_PERMISSION_ESCALATION: dict[str, Any] = {
    "escalation_id": "esc-1",
    "session_id": "s1",
    "tool_name": "Bash",
    "tool_input": {"command": "git push"},
    "task_intent": "ship the parser fix",
    "reason": "publishes work outside the working tree",
    "raised_at": "2026-07-29T12:00:00Z",
}


def test_constants_import_without_pydantic() -> None:
    """constants.py must be importable with zero third-party imports."""
    code = (
        "import broker.protocol.constants, sys; "
        "assert 'pydantic' not in sys.modules, 'pydantic leaked into constants'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_decision_literals_match_constants() -> None:
    """schemas.py Literal values and constants.py strings must not drift."""
    field = PermissionDecisionPayload.model_fields["decision"]
    literal_values = set(typing.get_args(field.annotation))
    assert literal_values == {
        constants.DECISION_ALLOW,
        constants.DECISION_ESCALATED,
    }


def test_status_payload_extension_is_additive() -> None:
    """Old-style state-only payloads must still validate."""
    old = StatusPayload.model_validate({"state": "driving"})
    assert old.pane_id is None
    assert old.claude_session_id is None
    assert old.transcript_path is None
    new = StatusPayload(
        state=SessionState.DRIVING,
        pane_id="w3:p2",
        claude_session_id="sess-1",
        transcript_path="/private/tmp/t.jsonl",
    )
    assert StatusPayload.model_validate_json(new.model_dump_json()) == new


def test_permission_pane_rejects_every_missing_field() -> None:
    """Dropping any field must fail: a partial one cannot be acted on."""
    assert PermissionEscalationPayload.model_validate(
        COMPLETE_PERMISSION_ESCALATION
    ).permission_suggestions == []
    for omitted in COMPLETE_PERMISSION_ESCALATION:
        partial = {
            k: v
            for k, v in COMPLETE_PERMISSION_ESCALATION.items()
            if k != omitted
        }
        with pytest.raises(ValidationError):
            PermissionEscalationPayload.model_validate(partial)


def test_unknown_suggestion_arm_survives_as_a_dict() -> None:
    """An arm the binary grew must not fail the request that carries it."""
    payload = PermissionEscalationPayload.model_validate(
        {
            **COMPLETE_PERMISSION_ESCALATION,
            "permission_suggestions": [
                {"type": "addDirectories", "directories": ["/repo"]},
                {"type": "somethingNewer", "detail": 7},
            ],
        }
    )
    known, unknown = payload.permission_suggestions
    assert isinstance(known, AddDirectoriesSuggestion)
    assert known.directories == ["/repo"]
    assert unknown == {"type": "somethingNewer", "detail": 7}


def test_permission_request_needs_no_tool_use_id() -> None:
    """The live PermissionRequest payload carries no tool_use_id."""
    payload = PermissionRequestPayload.model_validate(
        {
            "tool_name": "Read",
            "tool_input": {"file_path": "/repo/x.py"},
            "cwd": "/repo",
            "transcript_path": "/private/tmp/t.jsonl",
        }
    )
    assert payload.permission_mode is None
    assert payload.permission_suggestions == []
    assert "tool_use_id" not in PermissionRequestPayload.model_fields
