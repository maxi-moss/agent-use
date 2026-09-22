"""Decision log: append/read/render round-trip; missing file reads empty."""

import json
from pathlib import Path

from broker.decision_log import DecisionKind, append, read_rows, render_log


def test_append_render_round_trip(tmp_path: Path) -> None:
    log = tmp_path / "sessions" / "s1" / "decisions.ndjson"
    append(
        log,
        kind=DecisionKind.ANSWERED,
        reasoning="grounded in CLAUDE.md",
        detail="use uv",
    )
    append(
        log,
        kind=DecisionKind.ESCALATION_RAISED,
        reasoning="irreversible",
        detail="drop col",
    )
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["kind"] == "answered"
    assert first["reasoning"] == "grounded in CLAUDE.md"
    assert "task_summary" not in first  # unset optional fields are not written
    text = render_log(log)
    assert "answered" in text
    assert "grounded in CLAUDE.md" in text
    assert "escalation_raised" in text
    # oldest first
    assert text.index("answered") < text.index("escalation_raised")


def test_optional_fields_survive_read_and_render(tmp_path: Path) -> None:
    log = tmp_path / "decisions.ndjson"
    append(
        log,
        kind=DecisionKind.ESCALATION_RAISED,
        reasoning="irreversible",
        detail="drop col",
        task_summary="Asked before dropping a column",
        escalation_id="e1",
        what_was_asked="drop users.legacy?",
    )
    append(
        log,
        kind=DecisionKind.COMPLETED,
        reasoning="all done",
        detail="",
        task_summary="Wrapped up",
        headline="Migrated the users table",
        supporting="Alembic revision applied and tests pass",
    )
    rows = read_rows(log)
    assert [r.kind for r in rows] == ["escalation_raised", "completed"]
    assert rows[0].task_summary == "Asked before dropping a column"
    assert rows[0].escalation_id == "e1"
    assert rows[0].what_was_asked == "drop users.legacy?"
    assert rows[0].headline is None
    assert rows[1].headline == "Migrated the users table"
    assert rows[1].supporting == "Alembic revision applied and tests pass"
    text = render_log(log)
    assert "summary: Wrapped up" in text
    assert "headline: Migrated the users table" in text
    assert "supporting: Alembic revision applied and tests pass" in text


def test_missing_file_reads_empty(tmp_path: Path) -> None:
    assert read_rows(tmp_path / "nope.ndjson") == []
    assert render_log(tmp_path / "nope.ndjson") == ""


def test_append_creates_parents(tmp_path: Path) -> None:
    log = tmp_path / "a" / "b" / "c.ndjson"
    append(log, kind=DecisionKind.ADOPTED, reasoning="", detail="count=3")
    assert log.exists()
