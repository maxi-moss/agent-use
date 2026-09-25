"""Transcript reading: read_cleaned(), render(), TranscriptParseError, ReadReport.

Three-step parse per line, deliberately:
  1. json.loads         — malformed JSON is FATAL (TranscriptParseError)
  2. raw.map_line       — unrecognised record types are discarded (counted)
  3. EVENT_ADAPTER.validate_python — shape drift is tolerated (skipped + counted)

TypeAdapter.validate_json would hide malformed JSON inside a ValidationError
(type "json_invalid") and make the fatal/tolerant split impossible.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, assert_never, cast

from pydantic import ValidationError

from broker.transcript import raw
from broker.transcript.schemas import (
    EVENT_ADAPTER,
    TRANSCRIPT_VALIDATED_AGAINST,
    AskUserQuestionUse,
    ExitPlanModeUse,
    TranscriptEvent,
)


class TranscriptParseError(Exception):
    """Malformed JSON line, or a non-empty file yielding zero events. Fail loud."""


@dataclass
class ReadReport:
    unknown_types: Counter[str] = field(default_factory=Counter[str])
    skipped_records: int = 0
    versions: set[str] = field(default_factory=set[str])
    warnings: list[str] = field(default_factory=list[str])


def read_cleaned(path: Path) -> tuple[list[TranscriptEvent], ReadReport]:
    """Read a transcript file into cleaned events, with a report of what was lost.

    Args:
        path: The session's JSONL transcript, read as UTF-8.

    Returns:
        The cleaned events in file order, plus a report of what was lost.

    Raises:
        TranscriptParseError: A line is not valid JSON, or a non-empty file
            yielded no events at all.
    """
    text = path.read_text(encoding="utf-8")
    report = ReadReport()
    events: list[TranscriptEvent] = []
    # tool_use id -> which public result event it resolves to
    pending: dict[str, str] = {}
    nonblank_lines = 0

    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        nonblank_lines += 1
        try:
            parsed: Any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TranscriptParseError(
                f"{path}: line {lineno}: malformed JSON ({exc.msg}): {line[:120]!r}"
            ) from exc
        if not isinstance(parsed, dict):
            # Valid JSON but not a record object — shape drift, tolerated.
            report.unknown_types["<non-object>"] += 1
            continue
        obj = cast(dict[str, Any], parsed)

        version = raw.record_version(obj)
        if version is not None:
            report.versions.add(version)

        mapped = raw.map_line(obj)
        if not mapped:
            rtype = raw.record_type(obj)
            if rtype is None or rtype not in raw.KNOWN_RECORD_TYPES:
                report.unknown_types[rtype or "<missing>"] += 1
            continue

        for kind, data in mapped:
            if kind == raw.KIND_TOOL_RESULT:
                resolved = _resolve_tool_result(data, pending)
                if resolved is None:
                    # Implementation tool result (Bash, Read, ...) — stripped.
                    continue
                data = resolved
            try:
                event = EVENT_ADAPTER.validate_python(data)
            except ValidationError:
                report.skipped_records += 1
                continue
            if isinstance(event, AskUserQuestionUse):
                pending[event.id] = "ask_user_answer"
            elif isinstance(event, ExitPlanModeUse):
                pending[event.id] = "exit_plan_result"
            events.append(event)

    if nonblank_lines > 0 and not events:
        raise TranscriptParseError(
            f"{path}: non-empty file ({nonblank_lines} lines) yielded zero events"
        )

    mismatched = sorted(
        v for v in report.versions if v != TRANSCRIPT_VALIDATED_AGAINST
    )
    if mismatched:
        report.warnings.append(
            f"transcript version(s) {mismatched} differ from validated "
            f"{TRANSCRIPT_VALIDATED_AGAINST!r} — adapter output may be stale"
        )
    return events, report


def _resolve_tool_result(
    data: dict[str, Any], pending: dict[str, str]
) -> dict[str, Any] | None:
    """Re-kind an internal tool result into the public event it answers.

    Args:
        data: One internal tool-result mapping from ``raw.map_line``.
        pending: Tool_use id -> public result kind, mutated by the pop.

    Returns:
        The public event dict, or ``None`` when no pending call claims this id.
    """
    tool_use_id = data["tool_use_id"]
    result_kind = pending.pop(tool_use_id, None)
    if result_kind is None:
        return None
    return {
        "kind": result_kind,
        "id": tool_use_id,
        "raw": data["raw"],
        "rejected": data["rejected"],
        "answers": data["answers"],
    }


def render(events: list[TranscriptEvent]) -> str:
    """Render cleaned events as a Markdown-ish transcript.

    Args:
        events: Cleaned events, in the order the adapter produced them.

    Returns:
        The rendered transcript, one titled section per event.
    """
    parts: list[str] = []
    for event in events:
        if event.kind == "user_prompt":
            parts.append(f"## user\n{event.text}\n\n")
        elif event.kind == "assistant_text":
            parts.append(f"## assistant\n{event.text}\n\n")
        elif event.kind == "ask_user_question":
            lines = [f"## question (id={event.id})"]
            for question in event.questions:
                lines.append(f"{question.question} [{question.header}]")
                for option in question.options:
                    lines.append(f"- {option.label}: {option.description}")
            parts.append("\n".join(lines) + "\n\n")
        elif event.kind == "ask_user_answer":
            parts.append(f"## answer (id={event.id})\n{event.raw}\n\n")
        elif event.kind == "exit_plan_mode":
            parts.append(f"## plan (id={event.id})\n{event.plan}\n\n")
        elif event.kind == "exit_plan_result":
            parts.append(f"## plan-result (id={event.id})\n{event.raw}\n\n")
        else:
            assert_never(event)
    return "".join(parts)
