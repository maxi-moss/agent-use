"""Append-only NDJSON decision log.

The transcript is the audit trail; this log is the WHY — triage reasoning per
routing outcome. Timestamps are fine here (they never enter LLM context)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def append(path: Path, *, kind: str, reasoning: str, detail: str) -> None:
    """Append one routing decision to the NDJSON log.

    Args:
        path: Log file; created (with parent directories) on first write.
        kind: The routing outcome this entry records.
        reasoning: The triage reasoning behind that outcome.
        detail: Outcome-specific text, such as the answer or the summary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": kind,
        "reasoning": reasoning,
        "detail": detail,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


def render_log(path: Path) -> str:
    """Render the decision log as readable text, newest last, no truncation.

    Args:
        path: Log file to read.

    Returns:
        One block per entry, or the empty string if the file does not exist.
    """
    if not path.exists():
        return ""
    parts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed: Any = json.loads(line)  # our own writes; malformed = fail loud
        entry = cast(dict[str, Any], parsed)
        parts.append(
            f"[{entry.get('ts', '?')}] {entry.get('kind', '?')}\n"
            f"  reasoning: {entry.get('reasoning', '')}\n"
            f"  detail: {entry.get('detail', '')}\n"
        )
    return "\n".join(parts)
