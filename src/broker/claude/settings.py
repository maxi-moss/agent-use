"""Hook registration + verify-and-repair on ~/.claude/settings.json (spec §4.1).

Foreign entries — including Herdr's integration hook — are preserved verbatim.
Gotcha 6: settings scopes REPLACE the whole hooks array per event, they don't
merge; a project-level file with its own entry silently shadows the user-level
registration. Phase 0 structures the warning; Phase 1 supplies candidate paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from broker.claude.atomic import atomic_update_json
from broker.claude.paths import settings_path
from broker.protocol.constants import HOOK_SETTINGS_TIMEOUT

# Substring of our hook command used to identify OUR entries during repair.
BROKER_HOOK_MARKER = "broker-hook"


@dataclass
class RepairReport:
    repaired_events: list[str] = field(default_factory=list[str])
    warnings: list[str] = field(default_factory=list[str])


def hook_entry(command: str) -> dict[str, Any]:
    return {
        "hooks": [
            {
                "type": "command",
                "command": command,
                "timeout": HOOK_SETTINGS_TIMEOUT,
            }
        ]
    }


def _entry_is_ours(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    hooks = cast(dict[str, Any], entry).get("hooks")
    if not isinstance(hooks, list):
        return False
    for hook in cast(list[Any], hooks):
        if isinstance(hook, dict) and BROKER_HOOK_MARKER in str(
            cast(dict[str, Any], hook).get("command", "")
        ):
            return True
    return False


def _ensure_registered(
    data: dict[str, Any], events: list[str], command: str
) -> list[str]:
    """Append our entry for each event lacking one. Returns events repaired.
    Mutates `data` in place; every foreign entry is left untouched."""
    if BROKER_HOOK_MARKER not in command:
        raise ValueError(
            f"hook command {command!r} must contain {BROKER_HOOK_MARKER!r} "
            "or verify-and-repair cannot identify it later"
        )
    raw_hooks = data.setdefault("hooks", {})
    if not isinstance(raw_hooks, dict):
        raise ValueError("settings 'hooks' is not an object; refusing to touch it")
    hooks_by_event = cast(dict[str, Any], raw_hooks)
    added: list[str] = []
    for event in events:
        raw_entries = hooks_by_event.setdefault(event, [])
        if not isinstance(raw_entries, list):
            raise ValueError(
                f"settings hooks[{event!r}] is not a list; refusing to touch it"
            )
        entries = cast(list[Any], raw_entries)
        if any(_entry_is_ours(entry) for entry in entries):
            continue
        entries.append(hook_entry(command))
        added.append(event)
    return added


def register_hooks(
    events: list[str], command: str, path: Path | None = None
) -> None:
    target = path if path is not None else settings_path()

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        _ensure_registered(data, events, command)
        return data

    atomic_update_json(target, mutate)


def verify_and_repair(
    events: list[str], command: str, path: Path | None = None
) -> RepairReport:
    target = path if path is not None else settings_path()
    report = RepairReport()

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        report.repaired_events = _ensure_registered(data, events, command)
        return data

    atomic_update_json(target, mutate)
    if report.repaired_events:
        report.warnings.append(
            f"re-added missing broker hook entries for {report.repaired_events}"
        )
    # Gotcha 6 structural warning slot: project-level settings files REPLACE the
    # per-event hooks array and can shadow this registration. Detection needs
    # candidate project paths, which Phase 1 supplies; the report field exists now.
    return report
