"""Append-only NDJSON decision log — one session's shared routing record.

The session broker is the only writer. The master reads it two ways: rendered
text for the master LLM's get_decision_log tool, and typed rows for the
completed-outcome modal. Neutral top-level module: broker.master may not import
broker.session, but both agree on the row format here. The rendered text never
carries a timestamp: it enters the master LLM's context, which must assemble
byte-identically across calls."""

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class DecisionLogKind(StrEnum):
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
    DISPATCH_STALE = "dispatch_stale"
    ERROR = "error"
    ESCALATION_RAISED = "escalation_raised"
    NO_ACTION = "no_action"
    NOTIFICATION = "notification"
    QUESTION_REFUSED = "question_refused"
    REACTIVATED = "reactivated"
    RESUMED = "resumed"
    RETRACTED = "retracted"
    SESSION_END = "session_end"
    WATCHDOG_RECONCILIATION = "watchdog_reconciliation"


class DecisionLogRow(BaseModel):
    """One decision-log entry parsed back from disk.

    extra="ignore" so a newer writer's fields never break an older reader."""

    model_config = ConfigDict(extra="ignore")

    ts: str = ""
    kind: DecisionLogKind
    reasoning: str = ""
    detail: str = ""
    task_summary: str | None = None
    escalation_id: str | None = None
    tool_use_id: str | None = None
    headline: str | None = None
    supporting: str | None = None


def append(path: Path, row: DecisionLogRow) -> None:
    """Append one routing decision to the NDJSON log.

    Args:
        path: Log file; created (with parent directories) on first write.
        row: The decision to record; its ``ts`` is overwritten here.
    """
    entry = row.model_copy(
        update={"ts": datetime.now(UTC).isoformat(timespec="seconds")}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json(exclude_none=True) + "\n")
        f.flush()


def read_rows(path: Path) -> list[DecisionLogRow]:
    """Parse the decision log into typed rows, oldest first.

    Args:
        path: Log file to read.

    Returns:
        One DecisionLogRow per non-blank line, or [] if the file does not exist.
    """
    if not path.exists():
        return []
    rows: list[DecisionLogRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed: Any = json.loads(line)  # our own writes; malformed = fail loud
        rows.append(DecisionLogRow.model_validate(parsed))
    return rows


def render_decision_log(path: Path) -> str:
    """Render the decision log as readable text, oldest first, no truncation.

    Args:
        path: Log file to read.

    Returns:
        One block per entry, or the empty string if the file does not exist.
    """
    parts: list[str] = []
    for row in read_rows(path):
        block = (
            f"{row.kind}\n"
            f"  reasoning: {row.reasoning}\n"
            f"  detail: {row.detail}"
        )
        for label, value in (
            ("escalation_id", row.escalation_id),
            ("tool_use_id", row.tool_use_id),
            ("summary", row.task_summary),
            ("headline", row.headline),
            ("supporting", row.supporting),
        ):
            if value is not None:
                block += f"\n  {label}: {value}"
        parts.append(block + "\n")
    return "\n".join(parts)
