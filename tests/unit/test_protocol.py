"""Protocol package tests: stdlib-only constants, envelope round-trip, Literal sync."""

import subprocess
import sys
import typing

from broker.protocol import constants
from broker.protocol.schemas import (
    PermissionDecisionPayload,
    StatusPayload,
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
        state="driving",
        pane_id="w3:p2",
        claude_session_id="sess-1",
        transcript_path="/private/tmp/t.jsonl",
    )
    assert StatusPayload.model_validate_json(new.model_dump_json()) == new
