#!/usr/bin/env python3
"""Extract the timing of the end-to-end flow from a diagnostics NDJSON file.

Reads the shared ``diagnostics.ndjson`` written by the master and session
brokers and prints, for one run:

- a timeline of every event, offset from the first,
- every measured span (grounding, thinking, injection, …) slowest first,
- the cross-process and human-wait gaps computed from the marks,
- the end-to-end wall-clock total.

Usage:
    python scripts/flow_timings.py [path-to-diagnostics.ndjson]

With no path it looks for ``./diagnostics.ndjson`` and then
``~/.broker/logs/diagnostics.ndjson``.
"""

import json
import sys
from pathlib import Path
from typing import Any, cast


def _load(path: Path) -> list[dict[str, Any]]:
    """Read the NDJSON file into a list of records, sorted by timestamp."""
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed: Any = json.loads(line)
        if isinstance(parsed, dict):
            events.append(cast(dict[str, Any], parsed))
    events.sort(key=lambda e: e["ts"])
    return events


def _default_path() -> Path | None:
    """Pick the diagnostics file: CWD first, then the default broker home."""
    for candidate in (
        Path("diagnostics.ndjson"),
        Path.home() / ".broker" / "logs" / "diagnostics.ndjson",
    ):
        if candidate.is_file():
            return candidate
    return None


def _tag(event: dict[str, Any]) -> str:
    """Render the session/proposal/extra suffix shown after a stage name."""
    bits: list[str] = []
    if "session" in event:
        bits.append(str(event["session"]))
    extra = event.get("extra")
    if isinstance(extra, dict):
        for key, value in cast(dict[str, Any], extra).items():
            bits.append(f"{key}={value}")
    if "proposal_id" in event:
        bits.append(f"prop={str(event['proposal_id'])[:8]}")
    return f"  [{', '.join(bits)}]" if bits else ""


def _fmt(ms: float) -> str:
    """Format a duration in ms, switching to seconds past one second."""
    return f"{ms / 1000:8.2f} s" if ms >= 1000 else f"{ms:8.1f} ms"


def _find(
    events: list[dict[str, Any]],
    stage: str,
    phase: str,
    *,
    after: float | None = None,
    tool: str | None = None,
) -> dict[str, Any] | None:
    """Return the first event matching stage/phase (and optional tool/after)."""
    for event in events:
        if event["stage"] != stage or event["phase"] != phase:
            continue
        if after is not None and event["ts"] <= after:
            continue
        if tool is not None:
            extra = event.get("extra")
            if not isinstance(extra, dict):
                continue
            if cast(dict[str, Any], extra).get("tool") != tool:
                continue
        return event
    return None


def _timeline(events: list[dict[str, Any]]) -> None:
    """Print every event as an offset from the first one."""
    t0 = events[0]["ts"]
    print("=== Timeline (offset from first event) ===")
    for event in events:
        offset = (event["ts"] - t0) * 1000
        dur = event.get("duration_ms")
        dur_str = f"  ({_fmt(dur).strip()})" if dur is not None else ""
        print(
            f"+{offset / 1000:7.2f}s  {event['stage']}.{event['phase']}"
            f"{dur_str}{_tag(event)}"
        )


def _spans(events: list[dict[str, Any]]) -> None:
    """Print each measured span (an ``end`` record) slowest first."""
    ends = [e for e in events if e["phase"] == "end"]
    ends.sort(key=lambda e: e["duration_ms"], reverse=True)
    print("\n=== Measured stages (slowest first) ===")
    for event in ends:
        print(f"{_fmt(event['duration_ms'])}  {event['stage']}{_tag(event)}")


def _gaps(events: list[dict[str, Any]]) -> None:
    """Print the cross-process and human-wait gaps computed from the marks."""
    print("\n=== Transit & wait gaps ===")
    rows: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]] = []

    spawn_end = _find(events, "master.spawn_subprocess", "end")
    rows.append(
        ("session process launch", spawn_end,
         _find(events, "session.process_start", "mark"))
    )
    rows.append(
        ("proposal transit (session→master)",
         _find(events, "session.send_proposal", "begin"),
         _find(events, "master.proposal_received", "mark"))
    )
    presented = _find(events, "master.proposal_presented", "mark")
    dev_reply = (
        _find(events, "master.developer_message", "mark",
              after=presented["ts"])
        if presented is not None
        else None
    )
    rows.append(("developer read/decide (human)", presented, dev_reply))
    rows.append(
        ("approval decision (master turn)", dev_reply,
         _find(events, "master.approve_prompt", "mark"))
    )
    rows.append(
        ("approval transit (master→session)",
         _find(events, "master.send_approval", "begin"),
         _find(events, "session.approval_received", "mark"))
    )

    for label, start, end in rows:
        if start is None or end is None:
            print(f"{'    —    ':>11}  {label}  (anchors missing)")
            continue
        print(f"{_fmt((end['ts'] - start['ts']) * 1000)}  {label}")


def main() -> int:
    """Load the diagnostics file and print the timing breakdown."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        found = _default_path()
        if found is None:
            print(
                "no diagnostics.ndjson found; pass a path as the first argument",
                file=sys.stderr,
            )
            return 1
        path = found
    events = _load(path)
    if not events:
        print(f"{path}: no events", file=sys.stderr)
        return 1
    print(f"# {path}  —  {len(events)} events\n")
    _timeline(events)
    _spans(events)
    _gaps(events)
    total = (events[-1]["ts"] - events[0]["ts"]) * 1000
    print(f"\n=== End-to-end total: {_fmt(total).strip()} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
