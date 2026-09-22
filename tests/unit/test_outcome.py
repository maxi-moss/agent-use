"""build_outcome: decision-log rows -> SessionOutcome, pure and renderer-neutral."""

from broker.decision_log import DecisionKind, DecisionRow
from broker.master.outcome import OutcomeEvent, build_outcome
from broker.protocol.constants import SessionState


def _row(kind: DecisionKind, **fields: str) -> DecisionRow:
    return DecisionRow(kind=kind, ts=f"t-{kind}", **fields)


def _build(state: SessionState, *rows: DecisionRow):
    return build_outcome(
        session_id="s1", title="Attach recovery", state=state, rows=list(rows)
    )


def _kinds_and_labels(history: tuple[OutcomeEvent, ...]) -> list[tuple[str, str]]:
    return [(ev.kind, ev.label) for ev in history]


def test_kept_actions_in_order_and_empty_summaries_skipped() -> None:
    out = _build(
        SessionState.COMPLETED,
        _row(DecisionKind.ANSWERED, task_summary="Checked restart behavior"),
        _row(DecisionKind.ANSWERED, task_summary=""),
        _row(DecisionKind.ANSWERED, task_summary="Pinned the socket timeout"),
        _row(DecisionKind.ANSWERED, task_summary="Reran the suite"),
    )
    assert _kinds_and_labels(out.history) == [
        ("action", "Checked restart behavior"),
        ("action", "Pinned the socket timeout"),
        ("action", "Reran the suite"),
    ]


def test_escalation_carries_reason_and_next_kept_action_as_solution() -> None:
    out = _build(
        SessionState.COMPLETED,
        _row(
            DecisionKind.ESCALATION_RAISED, task_summary="Reason X", escalation_id="e1"
        ),
        _row(DecisionKind.DISPATCHED, escalation_id="e1", detail="go with B"),
        _row(DecisionKind.ANSWERED, task_summary="Did Y"),
    )
    assert _kinds_and_labels(out.history) == [
        ("escalation", "Escalation"),
        ("action", "Did Y"),
    ]
    escalation = out.history[0]
    assert escalation.detail == "Reason X"
    assert escalation.resolution == "Did Y"
    assert "go with B" not in [ev.label for ev in out.history]  # raw prose never shown


def test_escalation_without_follow_up_gets_fallback() -> None:
    out = _build(
        SessionState.STOPPED,
        _row(
            DecisionKind.ESCALATION_RAISED, task_summary="Reason X", escalation_id="e1"
        ),
    )
    assert out.history[0].resolution == "(no recorded follow-up)"


def test_developer_prompt_result_and_fallback() -> None:
    out = _build(
        SessionState.COMPLETED,
        _row(DecisionKind.DEVELOPER_PROMPT, detail="switch to fastapi please"),
        _row(DecisionKind.ANSWERED, task_summary="Switched to fastapi"),
        _row(DecisionKind.DEVELOPER_PROMPT, detail="one more thing"),
    )
    assert _kinds_and_labels(out.history) == [
        ("developer", "Developer instruction"),
        ("action", "Switched to fastapi"),
        ("developer", "Developer instruction"),
    ]
    assert out.history[0].resolution == "Switched to fastapi"
    assert out.history[2].resolution == "(no recorded follow-up)"


def test_completion_sets_headline_and_resolves_pending_escalation() -> None:
    out = _build(
        SessionState.COMPLETED,
        _row(
            DecisionKind.ESCALATION_RAISED,
            task_summary="Asked about the schema",
            escalation_id="e1",
        ),
        _row(DecisionKind.DISPATCHED, escalation_id="e1"),
        _row(
            DecisionKind.COMPLETED,
            headline="H",
            supporting="S",
            task_summary="Wrapped up",
        ),
    )
    assert out.status == "completed"
    assert out.headline == "H"
    assert out.supporting == "S"
    assert out.history[0].resolution == "Wrapped up"
    assert (out.history[-1].kind, out.history[-1].label) == (
        "terminal",
        "Task completed",
    )


def test_error_state_uses_last_fatal_row_and_keeps_history() -> None:
    out = _build(
        SessionState.ERROR,
        _row(DecisionKind.ANSWERED, task_summary="Started the migration"),
        _row(
            DecisionKind.ERROR, reasoning="stale dispatch_decision ignored", detail="e9"
        ),
        _row(DecisionKind.ERROR, reasoning="HerdrError", detail="pane vanished"),
    )
    assert out.status == "error"
    assert out.headline == "Session ended with an error"
    assert out.supporting == "HerdrError: pane vanished"
    assert _kinds_and_labels(out.history) == [("action", "Started the migration")]


def test_error_state_without_error_row() -> None:
    out = _build(SessionState.ERROR)
    assert out.status == "error"
    assert out.supporting == "no error detail recorded"


def test_stopped_state_is_neutral() -> None:
    out = _build(
        SessionState.STOPPED,
        _row(DecisionKind.ANSWERED, task_summary="Did a thing"),
    )
    assert out.status == "stopped"
    assert out.headline == "Session ended without completing"
    assert out.supporting == ""


def test_reactivated_session_renders_both_runs_with_divider() -> None:
    out = _build(
        SessionState.COMPLETED,
        _row(DecisionKind.ANSWERED, task_summary="A1"),
        _row(
            DecisionKind.COMPLETED,
            headline="First done",
            supporting="S1",
            task_summary="A2",
        ),
        _row(DecisionKind.REACTIVATED, detail="new intent"),
        _row(DecisionKind.ANSWERED, task_summary="B1"),
        _row(
            DecisionKind.COMPLETED,
            headline="Second done",
            supporting="S2",
            task_summary="B2",
        ),
    )
    assert _kinds_and_labels(out.history) == [
        ("action", "A1"),
        ("terminal", "Task completed"),
        ("reactivated", "New task"),
        ("action", "B1"),
        ("terminal", "Task completed"),
    ]
    assert out.headline == "Second done"
    assert out.supporting == "S2"


def test_ignored_kinds_produce_no_events() -> None:
    out = _build(
        SessionState.COMPLETED,
        _row(DecisionKind.NO_ACTION),
        _row(DecisionKind.CLARIFIED, detail="because"),
        _row(DecisionKind.ASK_VERIFIED, detail="tool-1"),
        _row(DecisionKind.RETRACTED, detail="e1"),
        _row(DecisionKind.ADOPTED, detail="count=3"),
    )
    assert out.history == ()
    assert out.title == "Attach recovery"
    assert out.session_id == "s1"
