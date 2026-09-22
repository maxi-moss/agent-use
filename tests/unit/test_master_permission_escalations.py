"""PermissionEscalations: the per-session invariant, keyed clears, and eager
persistence through the store file."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from broker.master.permission_escalations import (
    PermissionEscalations,
    PermissionProtocolViolation,
    PermissionStoreError,
)
from broker.protocol.schemas import PermissionEscalationPayload


def permission_escalation(
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


@pytest.fixture
def home() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        yield Path(td)


@pytest.fixture
def store_path(home: Path) -> Path:
    return home / "permission-escalations.json"


def test_one_live_permission_escalation_per_session(store_path: Path) -> None:
    store = PermissionEscalations.load(store_path)
    p1 = permission_escalation("p1", "s1")
    p2 = permission_escalation("p2", "s2")
    store.accept(p1)
    store.accept(p2)
    with pytest.raises(PermissionProtocolViolation):
        store.accept(permission_escalation("p3", "s1"))
    assert store.entries == (p1, p2)


def test_find_names_only_a_live_entry(store_path: Path) -> None:
    store = PermissionEscalations.load(store_path)
    p1 = permission_escalation("p1", "s1")
    store.accept(p1)
    assert store.find("p1") is p1
    assert store.find("ghost") is None
    store.retract("p1")
    assert store.find("p1") is None


def test_retract_clears_only_the_named_entry(store_path: Path) -> None:
    store = PermissionEscalations.load(store_path)
    p1 = permission_escalation("p1", "s1")
    p2 = permission_escalation("p2", "s2")
    store.accept(p1)
    store.accept(p2)
    assert store.retract("p1") is p1
    assert store.retract("p1") is None
    assert store.entries == (p2,)


def test_retract_for_session_clears_that_sessions_entry(store_path: Path) -> None:
    store = PermissionEscalations.load(store_path)
    p1 = permission_escalation("p1", "s1")
    p2 = permission_escalation("p2", "s2")
    store.accept(p1)
    store.accept(p2)
    assert store.retract_for_session("s2") is p2
    assert store.retract_for_session("s2") is None
    assert store.entries == (p1,)


def test_every_mutation_is_persisted(store_path: Path) -> None:
    store = PermissionEscalations.load(store_path)
    p1 = permission_escalation("p1", "s1")
    p2 = permission_escalation("p2", "s2")
    store.accept(p1)
    store.accept(p2)
    assert PermissionEscalations.load(store_path).entries == (p1, p2)
    store.retract("p1")
    assert PermissionEscalations.load(store_path).entries == (p2,)
    store.retract_for_session("s2")
    assert PermissionEscalations.load(store_path).entries == ()


def test_load_missing_file_starts_empty(store_path: Path) -> None:
    assert PermissionEscalations.load(store_path).entries == ()


@pytest.mark.parametrize(
    "content",
    [
        "{broken",
        "[]",
        '{"permission_escalations": {}}',
        '{"permission_escalations": [{"escalation_id": "p1"}]}',
    ],
)
def test_load_refuses_an_unreadable_store(home: Path, content: str) -> None:
    broken = home / "broken.json"
    broken.write_text(content, encoding="utf-8")
    # A skipped entry is an open prompt the developer is never shown.
    with pytest.raises(PermissionStoreError):
        PermissionEscalations.load(broken)
