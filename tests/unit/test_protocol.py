"""Protocol package tests: stdlib-only constants, socket-map coverage, Literal sync, NACKs."""

import subprocess
import sys
import typing
from typing import Any

import pytest
from pydantic import ValidationError

from broker.protocol import constants
from broker.protocol.constants import NackCode
from broker.protocol.schemas import (
    MASTER_SOCKET_PAYLOADS,
    SESSION_SOCKET_PAYLOADS,
    AddDirectoriesSuggestion,
    Envelope,
    PermissionDecisionPayload,
    PermissionEscalationPayload,
    PermissionRequestPayload,
    Response,
    nack_response,
    parse_nack,
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


def test_every_message_type_in_exactly_one_socket_map() -> None:
    """A type missing from both maps has no validator; one in both is ambiguous."""
    message_types = {
        value for name, value in vars(constants).items() if name.startswith("T_")
    }
    assert SESSION_SOCKET_PAYLOADS.keys().isdisjoint(MASTER_SOCKET_PAYLOADS)
    assert SESSION_SOCKET_PAYLOADS.keys() | MASTER_SOCKET_PAYLOADS.keys() == (
        message_types
    )


def test_decision_literals_match_constants() -> None:
    """schemas.py Literal values and constants.py strings must not drift."""
    field = PermissionDecisionPayload.model_fields["decision"]
    literal_values = set(typing.get_args(field.annotation))
    assert literal_values == {
        constants.DECISION_ALLOW,
        constants.DECISION_ESCALATED,
    }


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
        }
    )
    assert payload.permission_suggestions == []
    assert "tool_use_id" not in PermissionRequestPayload.model_fields


def test_nack_code_survives_the_wire() -> None:
    """A refusal's code must read back as the same member after serialization."""
    env = Envelope(id="r1", type=constants.T_SEND_PROMPT)
    sent = nack_response(env, "session is 'error'", NackCode.WRONG_STATE)
    received = Response.model_validate_json(sent.model_dump_json())
    nack = parse_nack(received)
    assert nack.reason_code is NackCode.WRONG_STATE
    assert nack.error == "session is 'error'"


def test_refusal_without_error_fails_to_parse() -> None:
    """A NACK that does not say why must fail loud, never read as a refusal."""
    with pytest.raises(ValidationError):
        parse_nack(Response(id="r1", ok=False))
