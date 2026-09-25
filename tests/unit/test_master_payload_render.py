"""Pure renderer output: byte-for-byte values, no timestamp leakage,
deterministic formatting. No MasterRuntime involved."""

from collections.abc import Iterator
from typing import Any, cast

from broker.master.payload_render import (
    render_escalation,
    render_permission_escalation,
    render_proposal,
)
from broker.protocol.schemas import (
    EscalationPayload,
    PermissionEscalationPayload,
    PromptProposalPayload,
)


def _leaf_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in cast(list[Any], value):
            yield from _leaf_values(item)
    elif isinstance(value, dict):
        for item in cast(dict[str, Any], value).values():
            yield from _leaf_values(item)


def test_escalation_rendered_verbatim() -> None:
    payload = {
        "escalation_id": "e1",
        "session_id": "s1",
        "task_context": "ctx-task-value",
        "disclosure": {
            "escalation_title": "title-value",
            "situation": "situation-value",
            "what_was_asked": "asked-value",
            "what_is_at_stake": "stake-value",
            "alternatives": [
                {"option": "opt-a", "pros": "pros-a", "cons": "cons-a"},
                {"option": "opt-b", "pros": "pros-b", "cons": "cons-b"},
            ],
            "recommendation": "recommendation-value",
            "uncertainty": "uncertainty-value",
            "what_would_change_my_mind": "change-mind-value",
        },
    }
    rendered = render_escalation(EscalationPayload.model_validate(payload))
    # Every value the developer decides on appears byte-for-byte — no
    # paraphrase.
    for value in _leaf_values(payload):
        assert value in rendered


def test_permission_pane_rendered_names_pane_and_offers_no_dispatch() -> None:
    payload: dict[str, Any] = {
        "kind": "permission",
        "escalation_id": "p1",
        "session_id": "s1",
        "tool_name": "tool-name-value",
        "tool_input": {"command": "command-value"},
        "task_intent": "task-intent-value",
        "reason": "reason-value",
        "raised_at": "2026-07-29T12:00:00+00:00",
        "permission_suggestions": [{"type": "setMode", "mode": "mode-value"}],
    }
    rendered = render_permission_escalation(
        PermissionEscalationPayload.model_validate(payload), "w3:p2"
    )
    # Every field the developer judges the prompt on appears byte-for-byte.
    # raised_at is the resolution baseline, not something they read.
    judged = {k: v for k, v in payload.items() if k != "raised_at"}
    for value in _leaf_values(judged):
        assert value in rendered
    # No timestamp reaches the block: it is carried into the master's LLM
    # context, where a clock reading is only ever something to reason from.
    assert payload["raised_at"] not in rendered
    assert "w3:p2" in rendered  # the pane the native prompt is waiting in
    assert "cannot be answered here" in rendered
    assert "dispatch" not in rendered.lower()  # no affordance to answer it here


def test_proposal_rendered_verbatim() -> None:
    payload = PromptProposalPayload.model_validate(
        {
            "proposal_id": "p1",
            "proposed_prompt": "the exact proposed prompt",
            "grounding_summary": "the exact grounding summary",
            "retrieved": [
                {"name": "a.py::f", "score": 0.81},
                {"name": "a.py::g", "score": None},
            ],
        }
    )
    rendered = render_proposal(payload)
    assert "the exact proposed prompt" in rendered
    assert "the exact grounding summary" in rendered
    assert (
        "## Retrieved code\n- a.py::f (seed 0.81)\n- a.py::g" in rendered
    )
