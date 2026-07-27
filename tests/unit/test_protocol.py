"""Protocol package tests: stdlib-only constants, envelope round-trip, Literal sync."""

import subprocess
import sys
import typing

from broker.protocol import constants
from broker.protocol.schemas import (
    Envelope,
    PermissionDecisionPayload,
    Response,
)


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


def test_envelope_ndjson_round_trip() -> None:
    env = Envelope(
        id="abc123",
        type=constants.T_PERMISSION_REQUEST,
        session_id="sess-1",
        payload={"tool_name": "Bash"},
    )
    line = env.model_dump_json() + "\n"
    assert line.endswith("\n")
    back = Envelope.model_validate_json(line)
    assert back == env
    assert back.v == constants.PROTOCOL_VERSION


def test_response_ndjson_round_trip() -> None:
    resp = Response(id="abc123", ok=True, payload={"decision": "allow"})
    back = Response.model_validate_json(resp.model_dump_json() + "\n")
    assert back == resp
    assert back.type == constants.T_RESPONSE


def test_decision_literals_match_constants() -> None:
    """schemas.py Literal values and constants.py strings must not drift."""
    field = PermissionDecisionPayload.model_fields["decision"]
    literal_values = set(typing.get_args(field.annotation))
    assert literal_values == {
        constants.DECISION_ALLOW,
        constants.DECISION_ESCALATED,
    }


def test_oversized_line_guard_value() -> None:
    assert constants.MAX_LINE_BYTES == 1_048_576
    line = "x" * (constants.MAX_LINE_BYTES + 1)
    assert len(line.encode()) > constants.MAX_LINE_BYTES
