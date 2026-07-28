"""Protocol package tests: stdlib-only constants, envelope round-trip, Literal sync."""

import subprocess
import sys
import typing

from broker.protocol import constants
from broker.protocol.schemas import (
    ApprovePromptPayload,
    BudgetUpdatePayload,
    Envelope,
    PermissionDecisionPayload,
    PromptProposalPayload,
    Response,
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


def test_phase1_payloads_round_trip() -> None:
    proposal = PromptProposalPayload(
        proposal_id="p1",
        proposed_prompt="Do the thing.",
        grounding_summary="from CLAUDE.md",
    )
    assert (
        PromptProposalPayload.model_validate_json(proposal.model_dump_json())
        == proposal
    )
    approve = ApprovePromptPayload(proposal_id="p1", prompt="Do the thing, revised.")
    assert (
        ApprovePromptPayload.model_validate_json(approve.model_dump_json())
        == approve
    )
    budget = BudgetUpdatePayload(count=3)
    assert (
        BudgetUpdatePayload.model_validate_json(budget.model_dump_json()) == budget
    )


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


def test_phase1_type_constants() -> None:
    assert constants.T_PROMPT_PROPOSAL == "prompt_proposal"
    assert constants.T_APPROVE_PROMPT == "approve_prompt"
    assert constants.T_BUDGET_UPDATE == "budget_update"


def test_oversized_line_guard_value() -> None:
    assert constants.MAX_LINE_BYTES == 1_048_576
    line = "x" * (constants.MAX_LINE_BYTES + 1)
    assert len(line.encode()) > constants.MAX_LINE_BYTES
