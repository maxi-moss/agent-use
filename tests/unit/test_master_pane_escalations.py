"""PaneEscalations: the per-(session, kind) invariant, keyed clears, and eager
persistence through the store file."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from broker.master.pane_escalations import (
    PaneEscalations,
    PaneProtocolViolation,
    PaneStoreError,
)
from broker.protocol.schemas import (
    PermissionEscalationPayload,
    QuestionEscalationPayload,
)


def permission_pane(
    esc_id: str = "p1", session: str = "s1"
) -> PermissionEscalationPayload:
    return PermissionEscalationPayload.model_validate(
        {
            "escalation_id": esc_id,
            "session_id": session,
            "tool_name": "Bash",
            "tool_input": {"command": "git push"},
            "task_intent": "intent",
            "reason": "reason",
            "raised_at": "2026-07-29T12:00:00+00:00",
            "permission_suggestions": [],
        }
    )


def question_escalation(
    esc_id: str = "q1", session: str = "s1"
) -> QuestionEscalationPayload:
    return QuestionEscalationPayload(
        escalation_id=esc_id,
        session_id=session,
        task_context="intent",
        menu="Which layout? [Layout]",
        first_question="Which layout?",
        reason="irreversible",
    )


@pytest.fixture
def home() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        yield Path(td)


@pytest.fixture
def store_path(home: Path) -> Path:
    return home / "pane-escalations.json"


def test_one_live_pane_escalation_per_session_and_kind(store_path: Path) -> None:
    store = PaneEscalations.load(store_path)
    p1 = permission_pane("p1", "s1")
    p2 = permission_pane("p2", "s2")
    store.accept(p1)
    store.accept(p2)
    with pytest.raises(PaneProtocolViolation):
        store.accept(permission_pane("p3", "s1"))
    q1 = question_escalation("q1", "s1")
    store.accept(q1)
    with pytest.raises(PaneProtocolViolation):
        store.accept(question_escalation("q2", "s1"))
    assert store.entries == (p1, p2, q1)


def test_find_names_only_a_live_entry(store_path: Path) -> None:
    store = PaneEscalations.load(store_path)
    p1 = permission_pane("p1", "s1")
    store.accept(p1)
    assert store.find("p1") is p1
    assert store.find("ghost") is None
    store.retract("p1")
    assert store.find("p1") is None


def test_retract_clears_only_the_named_entry(store_path: Path) -> None:
    store = PaneEscalations.load(store_path)
    p1 = permission_pane("p1", "s1")
    q1 = question_escalation("q1", "s1")
    store.accept(p1)
    store.accept(q1)
    assert store.retract("p1") is p1
    assert store.retract("p1") is None
    assert store.entries == (q1,)


def test_retract_for_session_clears_both_kinds(store_path: Path) -> None:
    store = PaneEscalations.load(store_path)
    p1 = permission_pane("p1", "s1")
    q1 = question_escalation("q1", "s1")
    p2 = permission_pane("p2", "s2")
    store.accept(p1)
    store.accept(q1)
    store.accept(p2)
    assert store.retract_for_session("s1") == [p1, q1]
    assert store.retract_for_session("s1") == []
    assert store.entries == (p2,)


def test_every_mutation_is_persisted_with_its_kind(store_path: Path) -> None:
    store = PaneEscalations.load(store_path)
    p1 = permission_pane("p1", "s1")
    q1 = question_escalation("q1", "s1")
    p2 = permission_pane("p2", "s2")
    store.accept(p1)
    store.accept(q1)
    store.accept(p2)
    loaded = PaneEscalations.load(store_path).entries
    assert loaded == (p1, q1, p2)
    assert isinstance(loaded[1], QuestionEscalationPayload)
    store.retract("p1")
    assert PaneEscalations.load(store_path).entries == (q1, p2)
    store.retract_for_session("s2")
    assert PaneEscalations.load(store_path).entries == (q1,)


def test_in_session_order_is_numeric_then_kind(store_path: Path) -> None:
    store = PaneEscalations.load(store_path)
    q10 = question_escalation("q10", "s10")
    p10 = permission_pane("p10", "s10")
    p2 = permission_pane("p2", "s2")
    store.accept(q10)
    store.accept(p2)
    store.accept(p10)
    # Numeric session order (s2 before s10), not arrival order; within a
    # session, permission before question.
    assert store.in_session_order() == [p2, p10, q10]


def test_load_missing_file_starts_empty(store_path: Path) -> None:
    assert PaneEscalations.load(store_path).entries == ()


@pytest.mark.parametrize(
    "content",
    [
        "{broken",
        "[]",
        '{"pane_escalations": {}}',
        '{"pane_escalations": [{"escalation_id": "p1"}]}',
        '{"pane_escalations": [{"kind": "question", "escalation_id": "q1"}]}',
    ],
)
def test_load_refuses_an_unreadable_store(home: Path, content: str) -> None:
    broken = home / "broken.json"
    broken.write_text(content, encoding="utf-8")
    # A skipped entry is an open prompt the developer is never shown.
    with pytest.raises(PaneStoreError):
        PaneEscalations.load(broken)
