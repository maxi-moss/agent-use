"""EscalationQueue: FIFO order, the per-session invariant, keyed clears, and
eager persistence through the queue file."""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from broker.master.queue import (
    EscalationProtocolViolation,
    EscalationQueue,
    QueueError,
)
from broker.protocol.schemas import EscalationPayload


def escalation(esc_id: str = "e1", session: str = "s1") -> EscalationPayload:
    return EscalationPayload.model_validate(
        {
            "escalation_id": esc_id,
            "session_id": session,
            "task_context": "ctx",
            "escalation_title": "title",
            "situation": "sit",
            "what_was_asked": "asked",
            "what_is_at_stake": "stake",
            "alternatives": [{"option": "A", "pros": "pro", "cons": "con"}],
            "recommendation": "rec",
            "uncertainty": "unc",
            "what_would_change_my_mind": "change",
        }
    )


@pytest.fixture
def home() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        yield Path(td)


@pytest.fixture
def queue_path(home: Path) -> Path:
    return home / "escalation-queue.json"


def test_accept_makes_first_payload_active(queue_path: Path) -> None:
    queue = EscalationQueue.load(queue_path)
    assert queue.active is None
    e1 = escalation("e1")
    queue.accept(e1)
    assert queue.active is e1
    assert queue.depth == 1
    assert queue.waiting == ()


def test_second_accept_queues_behind_the_active(queue_path: Path) -> None:
    queue = EscalationQueue.load(queue_path)
    e1 = escalation("e1", "s1")
    e2 = escalation("e2", "s2")
    queue.accept(e1)
    queue.accept(e2)
    assert queue.active is e1
    assert queue.depth == 2
    assert queue.waiting == ("s2",)


def test_fifo_order_preserved_across_resolves(queue_path: Path) -> None:
    queue = EscalationQueue.load(queue_path)
    e1 = escalation("e1", "s1")
    e2 = escalation("e2", "s2")
    e3 = escalation("e3", "s3")
    queue.accept(e1)
    queue.accept(e2)
    queue.accept(e3)
    assert queue.active is e1
    assert queue.resolve("e1") is e1
    assert queue.active is e2
    assert queue.resolve("e2") is e2
    assert queue.active is e3
    assert queue.resolve("e3") is e3
    assert queue.active is None


def test_same_session_raise_while_queued_is_protocol_violation(
    queue_path: Path,
) -> None:
    queue = EscalationQueue.load(queue_path)
    queue.accept(escalation("e1", "s1"))
    queue.accept(escalation("e2", "s2"))
    # The session's live entry is waiting, not surfaced — still its turn to hold.
    with pytest.raises(EscalationProtocolViolation):
        queue.accept(escalation("e3", "s2"))
    assert queue.depth == 2


def test_retract_removes_a_queued_entry(queue_path: Path) -> None:
    queue = EscalationQueue.load(queue_path)
    e1 = escalation("e1", "s1")
    e2 = escalation("e2", "s2")
    e3 = escalation("e3", "s3")
    queue.accept(e1)
    queue.accept(e2)
    queue.accept(e3)
    assert queue.retract("e2") is e2
    assert queue.active is e1  # the head is untouched
    assert queue.waiting == ("s3",)


def test_retract_of_the_head_advances_active(queue_path: Path) -> None:
    queue = EscalationQueue.load(queue_path)
    e1 = escalation("e1", "s1")
    e2 = escalation("e2", "s2")
    queue.accept(e1)
    queue.accept(e2)
    assert queue.retract("e1") is e1
    assert queue.active is e2


def test_resolve_ignores_a_non_head_id(queue_path: Path) -> None:
    queue = EscalationQueue.load(queue_path)
    e1 = escalation("e1", "s1")
    e2 = escalation("e2", "s2")
    queue.accept(e1)
    queue.accept(e2)
    assert queue.resolve("e2") is None  # live, but not the head
    assert queue.resolve("ghost") is None
    assert queue.depth == 2
    assert queue.active is e1


def test_retract_for_session_clears_that_sessions_entry(queue_path: Path) -> None:
    queue = EscalationQueue.load(queue_path)
    e1 = escalation("e1", "s1")
    e2 = escalation("e2", "s2")
    queue.accept(e1)
    queue.accept(e2)
    assert queue.retract_for_session("s2") is e2
    assert queue.active is e1
    assert queue.retract_for_session("s2") is None


def test_round_trip_through_load(queue_path: Path) -> None:
    queue = EscalationQueue.load(queue_path)
    e1 = escalation("e1", "s1")
    e2 = escalation("e2", "s2")
    queue.accept(e1)
    queue.accept(e2)
    reloaded = EscalationQueue.load(queue_path)
    assert reloaded.entries == (e1, e2)


def test_load_missing_file_starts_empty(queue_path: Path) -> None:
    queue = EscalationQueue.load(queue_path)
    assert queue.active is None
    assert queue.depth == 0


def test_load_malformed_file_raises_queue_error(home: Path) -> None:
    broken = home / "broken.json"
    broken.write_text("{broken", encoding="utf-8")
    with pytest.raises(QueueError):
        EscalationQueue.load(broken)
    invalid_entry = home / "invalid-entry.json"
    invalid_entry.write_text(
        '{"queue": [{"escalation_id": "e1"}]}', encoding="utf-8"
    )
    # One bad entry fails the whole load — a skipped escalation is a silently
    # dropped decision.
    with pytest.raises(QueueError):
        EscalationQueue.load(invalid_entry)


def test_every_mutation_is_persisted(queue_path: Path) -> None:
    queue = EscalationQueue.load(queue_path)
    queue.accept(escalation("e1", "s1"))
    queue.accept(escalation("e2", "s2"))
    assert EscalationQueue.load(queue_path).depth == 2
    queue.retract("e2")
    assert EscalationQueue.load(queue_path).depth == 1
    queue.resolve("e1")
    assert EscalationQueue.load(queue_path).depth == 0
