"""Permission log: append/render round-trip; missing file renders empty."""

import json
from pathlib import Path

from broker.permission.permission_log import append, render_permission_log


def test_append_render_round_trip(tmp_path: Path) -> None:
    log = tmp_path / "sessions" / "s1" / "permissions.ndjson"
    append(
        log,
        tool_name="Read",
        tool_input={"file_path": "/repo/a.py"},
        decision="allow",
        reason="reversible read inside the working tree",
        model_id="claude-haiku-4-5",
        latency_ms=412,
    )
    append(
        log,
        tool_name="Bash",
        tool_input={"command": "git push"},
        decision="escalated",
        reason="publishes to a shared remote",
        model_id=None,
        latency_ms=0,
    )
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["tool_name"] == "Read"
    assert first["decision"] == "allow"
    assert first["latency_ms"] == 412
    assert first["tool_input"] == {"file_path": "/repo/a.py"}
    text = render_permission_log(log)
    assert "reversible read inside the working tree" in text
    assert "publishes to a shared remote" in text
    assert "git push" in text
    # newest last
    assert text.index("Read") < text.index("Bash")


def test_render_has_no_timestamp(tmp_path: Path) -> None:
    log = tmp_path / "permissions.ndjson"
    append(
        log,
        tool_name="Read",
        tool_input={},
        decision="allow",
        reason="reversible",
        model_id="claude-haiku-4-5",
        latency_ms=1,
    )
    ts = json.loads(log.read_text())["ts"]
    assert ts
    assert ts not in render_permission_log(log)


def test_render_missing_file_is_empty(tmp_path: Path) -> None:
    assert render_permission_log(tmp_path / "nope.ndjson") == ""


def test_append_creates_parents(tmp_path: Path) -> None:
    log = tmp_path / "a" / "b" / "c.ndjson"
    append(
        log,
        tool_name="Grep",
        tool_input={},
        decision="allow",
        reason="search",
        model_id="m",
        latency_ms=1,
    )
    assert log.exists()
