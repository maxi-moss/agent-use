"""Decision log: append/render round-trip; missing file renders empty."""

import json
from pathlib import Path

from broker.session.decision_log import append, render_log


def test_append_render_round_trip(tmp_path: Path) -> None:
    log = tmp_path / "sessions" / "s1" / "decisions.ndjson"
    append(log, kind="answered", reasoning="grounded in CLAUDE.md", detail="use uv")
    append(log, kind="escalation_raised", reasoning="irreversible", detail="drop col")
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["kind"] == "answered"
    assert first["reasoning"] == "grounded in CLAUDE.md"
    text = render_log(log)
    assert "answered" in text
    assert "grounded in CLAUDE.md" in text
    assert "escalation_raised" in text
    # newest last
    assert text.index("answered") < text.index("escalation_raised")


def test_render_missing_file_is_empty(tmp_path: Path) -> None:
    assert render_log(tmp_path / "nope.ndjson") == ""


def test_append_creates_parents(tmp_path: Path) -> None:
    log = tmp_path / "a" / "b" / "c.ndjson"
    append(log, kind="budget", reasoning="", detail="count=3")
    assert log.exists()
