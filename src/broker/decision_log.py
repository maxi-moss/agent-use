"""Append-only NDJSON decision log — one session's shared routing record.

The session broker is the only writer. The master reads it two ways: rendered
text for the master LLM's get_decision_log tool, and typed rows for the
completed-outcome modal. Neutral top-level module: broker.master may not import
broker.session, but both agree on the row format here. Timestamps are fine —
this file never enters LLM context as raw JSON."""

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class DecisionKind(StrEnum):
    """Every routing outcome the session broker records."""

    ADOPTED = "adopted"
    ANSWERED = "answered"
    ASK_ANSWERED = "ask_answered"
    ASK_SKIPPED = "ask_skipped"
    ASK_VERIFIED = "ask_verified"
    ASK_VERIFY_FAILED = "ask_verify_failed"
    CLARIFIED = "clarified"
    CLARIFY_FAILED = "clarify_failed"
    COMPACTION = "compaction"
    COMPLETED = "completed"
    DEVELOPER_PROMPT = "developer_prompt"
    DISPATCHED = "dispatched"
    DISPATCH_FAILED = "dispatch_failed"
    ERROR = "error"
    ESCALATION_RAISED = "escalation_raised"
    NO_ACTION = "no_action"
    NOTIFICATION = "notification"
    REACTIVATED = "reactivated"
    RESUMED = "resumed"
    RETRACTED = "retracted"
    SESSION_END = "session_end"
    WATCHDOG_RECONCILIATION = "watchdog_reconciliation"


class DecisionRow(BaseModel):
    """One decision-log entry parsed back from disk.

    extra="ignore" so a newer writer's fields never break an older reader."""

    model_config = ConfigDict(extra="ignore")

    ts: str = ""
    kind: DecisionKind
    reasoning: str = ""
    detail: str = ""
    task_summary: str | None = None
    escalation_id: str | None = None
    what_was_asked: str | None = None
    headline: str | None = None
    supporting: str | None = None


def append(
    path: Path,
    *,
    kind: DecisionKind,
    reasoning: str,
    detail: str,
    task_summary: str | None = None,
    escalation_id: str | None = None,
    what_was_asked: str | None = None,
    headline: str | None = None,
    supporting: str | None = None,
) -> None:
    """Append one routing decision to the NDJSON log.

    Args:
        path: Log file; created (with parent directories) on first write.
        kind: The routing outcome this entry records.
        reasoning: The triage reasoning behind that outcome.
        detail: Outcome-specific verbatim text (answer, situation, response).
        task_summary: Developer-facing one-line history summary, for a kept
            triage action or an escalation's Reason.
        escalation_id: Links an escalation's raise row to its dispatch row.
        what_was_asked: The escalation question, verbatim.
        headline: Completion outcome headline.
        supporting: Completion supporting assertion.
    """
    entry: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": kind,
        "reasoning": reasoning,
        "detail": detail,
    }
    optional = {
        "task_summary": task_summary,
        "escalation_id": escalation_id,
        "what_was_asked": what_was_asked,
        "headline": headline,
        "supporting": supporting,
    }
    entry.update({k: v for k, v in optional.items() if v is not None})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


def read_rows(path: Path) -> list[DecisionRow]:
    """Parse the decision log into typed rows, oldest first.

    Args:
        path: Log file to read.

    Returns:
        One DecisionRow per non-blank line, or [] if the file does not exist.
    """
    if not path.exists():
        return []
    rows: list[DecisionRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed: Any = json.loads(line)  # our own writes; malformed = fail loud
        rows.append(DecisionRow.model_validate(parsed))
    return rows


def render_log(path: Path) -> str:
    """Render the decision log as readable text, oldest first, no truncation.

    Args:
        path: Log file to read.

    Returns:
        One block per entry, or the empty string if the file does not exist.
    """
    parts: list[str] = []
    for row in read_rows(path):
        block = (
            f"[{row.ts or '?'}] {row.kind}\n"
            f"  reasoning: {row.reasoning}\n"
            f"  detail: {row.detail}"
        )
        for label, value in (
            ("summary", row.task_summary),
            ("headline", row.headline),
            ("supporting", row.supporting),
        ):
            if value is not None:
                block += f"\n  {label}: {value}"
        parts.append(block + "\n")
    return "\n".join(parts)
