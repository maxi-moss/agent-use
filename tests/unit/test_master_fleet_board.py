"""FleetBoard harness: real registry, queue and pane stores on disk; emit =
recording list."""

from pathlib import Path
from typing import Any

import pytest

from broker.master.fleet_board import FleetBoard
from broker.master.pane_escalations import PaneEscalations
from broker.master.payload_render import render_proposal
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.master.viewmodel import Attention, FleetUpdated
from broker.protocol.constants import SessionState
from broker.protocol.schemas import (
    PANE_ESCALATION_ADAPTER,
    EscalationPayload,
    LiveStatusPayload,
    PaneEscalationPayload,
    PromptProposalPayload,
)

BUDGET_MAX = 7


def escalation(esc_id: str, session: str) -> EscalationPayload:
    return EscalationPayload.model_validate(
        {
            "escalation_id": esc_id,
            "session_id": session,
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
    )


def pane_escalation(
    kind: str, esc_id: str, session: str
) -> PaneEscalationPayload:
    fields: dict[str, Any] = (
        {
            "tool_name": "tool-name-value",
            "tool_input": {"command": "command-value"},
            "task_intent": "task-intent-value",
        }
        if kind == "permission"
        else {
            "task_context": "ctx-task-value",
            "menu": "## Question 1: question-value [header-value]",
            "first_question": "question-value",
        }
    )
    return PANE_ESCALATION_ADAPTER.validate_python(
        {
            "kind": kind,
            "escalation_id": esc_id,
            "session_id": session,
            "reason": "reason-value",
            **fields,
        }
    )


def add_session(
    board: FleetBoard,
    name: str,
    state: SessionState = SessionState.DRIVING,
    **fields: Any,
) -> None:
    board.registry.upsert(
        SessionRecord(
            name=name,
            socket_path=f"/private/tmp/{name}.sock",
            cwd="/private/tmp",
            anchor_pane="%1",
            state=state,
            **fields,
        )
    )


@pytest.fixture
def posts() -> list[Any]:
    return []


@pytest.fixture
def board(tmp_path: Path, posts: list[Any]) -> FleetBoard:
    return FleetBoard(
        Registry.load(tmp_path / "registry.json"),
        EscalationQueue.load(tmp_path / "escalation-queue.json"),
        PaneEscalations.load(tmp_path / "pane-escalations.json"),
        BUDGET_MAX,
        posts.append,
    )


def test_build_view_orders_rows_numerically_with_budgets_and_titles(
    board: FleetBoard,
) -> None:
    add_session(board, "s1")
    add_session(
        board,
        "s10",
        intent="Fix the auth bug in the checkout flow before the demo",
        title="fix auth bug",
    )
    add_session(
        board,
        "s2",
        state=SessionState.ESCALATED,
        approved_prompt="Migrate the users table",
        title="migrate users table",
        budget_count=5,
    )
    view = board.build_view()
    # Numeric order (s2 before s10), not lexical.
    assert [row.session_id for row in view.rows] == ["s1", "s2", "s10"]
    # s1 was never approved: no title is set on it, and the row shows none —
    # title never falls back to approved_prompt or intent.
    assert view.rows[0].title == ""
    s2 = view.rows[1]
    assert s2.state == SessionState.ESCALATED
    assert s2.title == "migrate users table"
    assert s2.budget_count == 5
    assert s2.budget_max == BUDGET_MAX
    assert view.rows[2].title == "fix auth bug"


def test_build_view_idle_master_and_no_sessions(board: FleetBoard) -> None:
    view = board.build_view()
    assert view.master_activity is None
    assert view.rows == ()
    assert view.queue_depth == 0
    assert view.panes == ()


def test_build_view_reports_master_activity_and_queue_state(
    board: FleetBoard, posts: list[Any]
) -> None:
    for name in ("s1", "s2", "s10"):
        add_session(board, name)
    board.note_master_activity("thinking…")
    published = [m.view for m in posts if isinstance(m, FleetUpdated)]
    assert [view.master_activity for view in published] == ["thinking…"]
    board.queue.accept(escalation("e1", "s1"))
    board.queue.accept(escalation("e2", "s2"))
    board.panes.accept(pane_escalation("permission", "p10", "s10"))
    board.panes.accept(pane_escalation("permission", "p2", "s2"))
    view = board.build_view()
    assert view.master_activity == "thinking…"
    # Only decisions count as waiting; open prompts are listed apart, in
    # numeric session order (s2 before s10) whatever order they arrived in.
    assert view.queue_depth == 2
    assert view.waiting == ("s2",)
    assert [p.session_id for p in view.panes] == ["s2", "s10"]


def test_build_view_badges_reflect_queue_proposals_and_prompts(
    board: FleetBoard,
) -> None:
    add_session(board, "s1")
    add_session(board, "s2")
    add_session(board, "s3", pane_id="w3:p2")
    # s1: a queued decision escalation AND open pane escalations from the same
    # session — held apart, so every badge carries.
    board.queue.accept(escalation("e1", "s1"))
    board.panes.accept(pane_escalation("permission", "p1", "s1"))
    board.panes.accept(pane_escalation("question", "q1", "s1"))
    # s2: a pending prompt proposal.
    board.register_proposal(
        "s2",
        PromptProposalPayload(
            proposal_id="prop-1",
            proposed_prompt="do it",
            grounding_summary="facts",
        ),
    )
    # s3: sitting on a native permission prompt.
    assert board.apply_live_status(
        "s3", LiveStatusPayload(state=SessionState.DRIVING, permission_prompt=True)
    )
    rows = {row.session_id: row for row in board.build_view().rows}
    assert rows["s1"].badges == (
        Attention.ESCALATION,
        Attention.PERMISSION,
        Attention.QUESTION,
    )
    assert rows["s2"].badges == (Attention.PROPOSAL,)
    assert rows["s3"].badges == (Attention.PERMISSION,)
    assert rows["s3"].pane_id == "w3:p2"


def test_rendered_proposals_hold_the_latest_proposal_per_session(
    board: FleetBoard,
) -> None:
    add_session(board, "s1")
    add_session(board, "s2")
    payloads: dict[str, PromptProposalPayload] = {}
    for session, proposal_id in (("s1", "p1"), ("s2", "p2"), ("s1", "p3")):
        payloads[proposal_id] = PromptProposalPayload(
            proposal_id=proposal_id,
            proposed_prompt=f"prompt {proposal_id}",
            grounding_summary="facts",
        )
        board.register_proposal(session, payloads[proposal_id])
    assert board.rendered_proposals() == {
        "s1": render_proposal(payloads["p3"]),
        "s2": render_proposal(payloads["p2"]),
    }
