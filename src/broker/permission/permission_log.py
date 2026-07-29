"""Append-only NDJSON permission log.

Every permission request appends one entry, approvals included, and the entry
is written before the decision leaves the module — a crash in between must not
lose the record of a command that then ran. Timestamps are fine here (they
never enter LLM context)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def append(
    path: Path,
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    decision: str,
    reason: str,
    model_id: str | None,
    latency_ms: int,
    cached: bool,
) -> None:
    """Append one permission decision to the NDJSON log.

    Args:
        path: Log file; created (with parent directories) on first write.
        tool_name: Name of the tool the session asked to run.
        tool_input: Arguments the session passed to it.
        decision: The outcome this entry records.
        reason: Why that outcome was reached.
        model_id: Model that judged the call, or ``None`` when none did.
        latency_ms: Wall-clock time this decision took.
        cached: Whether the decision replayed an earlier judgement.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool_name": tool_name,
        "tool_input": tool_input,
        "decision": decision,
        "reason": reason,
        "model_id": model_id,
        "latency_ms": latency_ms,
        "cached": cached,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()


def render_log(path: Path) -> str:
    """Render the permission log as readable text, newest last, no truncation.

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
        served = "cache" if entry.get("cached") else entry.get("model_id") or "-"
        parts.append(
            f"[{entry.get('ts', '?')}] {entry.get('tool_name', '?')} -> "
            f"{entry.get('decision', '?')} "
            f"({served}, {entry.get('latency_ms', '?')}ms)\n"
            f"  input: {json.dumps(entry.get('tool_input', {}), sort_keys=True)}\n"
            f"  reason: {entry.get('reason', '')}\n"
        )
    return "\n".join(parts)
