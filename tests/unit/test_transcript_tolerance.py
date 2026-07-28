"""Tolerance vs fail-loud split."""

from pathlib import Path
from typing import Any

import pytest

from broker.transcript import adapter, raw
from broker.transcript.adapter import (
    TranscriptParseError,
    read_cleaned,
    read_cleaned_with_report,
)
from broker.transcript.schemas import EVENT_ADAPTER

FIXTURES = Path(__file__).parent.parent / "fixtures" / "transcripts"


def test_unknown_top_level_type_skipped() -> None:
    events, report = read_cleaned_with_report(FIXTURES / "unknown-type.jsonl")
    assert report.unknown_types["future-thing"] == 1
    assert [e.kind for e in events] == ["user_prompt", "assistant_text"]


def test_unknown_kind_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A kind the union does not know is skipped + counted, never fatal.

    raw.py never emits unknown kinds today, so drift is simulated at the
    raw boundary — exactly where a format change would introduce it.
    """
    original = raw.map_line

    def drifted(obj: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        mapped = original(obj)
        if mapped and mapped[0][0] == "assistant_text":
            return [("future_kind", {"kind": "future_kind", "text": "?"})]
        return mapped

    monkeypatch.setattr(raw, "map_line", drifted)
    events, report = read_cleaned_with_report(FIXTURES / "unknown-type.jsonl")
    assert report.skipped_records == 1
    assert [e.kind for e in events] == ["user_prompt"]


def test_unknown_fields_ignored() -> None:
    """extra='ignore' is the unknown-field tolerance rule — never extra='forbid'."""
    events, report = read_cleaned_with_report(FIXTURES / "unknown-type.jsonl")
    # every fixture record carries fields the models do not declare
    # (version, uuid, isSidechain ...) and still validates
    assert len(events) == 2
    assert report.skipped_records == 0


def test_malformed_json_line_raises() -> None:
    with pytest.raises(TranscriptParseError) as exc_info:
        read_cleaned(FIXTURES / "malformed-line.jsonl")
    assert "line 2" in str(exc_info.value)


def test_nonempty_file_zero_events_raises() -> None:
    with pytest.raises(TranscriptParseError):
        read_cleaned(FIXTURES / "zero-events.jsonl")


def test_empty_file_yields_zero_events_without_raising() -> None:
    assert read_cleaned(FIXTURES / "empty-file.jsonl") == []


def test_version_mismatch_warns_not_fatal(tmp_path: Path) -> None:
    p = tmp_path / "drift.jsonl"
    p.write_text(
        '{"type": "user", "version": "9.9.9", "origin": {"kind": "human"}, '
        '"message": {"role": "user", "content": "hi"}}\n'
    )
    events, report = read_cleaned_with_report(p)
    assert len(events) == 1
    assert "9.9.9" in report.versions
    assert any("9.9.9" in w for w in report.warnings)


def test_unmatched_tool_results_are_stripped(tmp_path: Path) -> None:
    """Implementation tool results (no paired preserved call) never surface."""
    p = tmp_path / "bash-result.jsonl"
    p.write_text(
        '{"type": "user", "version": "2.1.220", "origin": {"kind": "human"}, '
        '"message": {"role": "user", "content": "hi"}}\n'
        '{"type": "user", "version": "2.1.220", "message": {"role": "user", '
        '"content": [{"type": "tool_result", "tool_use_id": "toolu_bash", '
        '"content": "ran ok"}]}}\n'
    )
    events = read_cleaned(p)
    assert [e.kind for e in events] == ["user_prompt"]


def test_render_handles_compaction_boundary() -> None:
    event = EVENT_ADAPTER.validate_python({"kind": "compaction_boundary"})
    assert adapter.render([event]) == "## [compaction boundary]\n\n"
