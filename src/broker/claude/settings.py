"""Claude Code settings writers: hook registration and per-session rules.

Foreign entries — including Herdr's integration hook — are preserved verbatim.
Settings scopes REPLACE the whole hooks array per event, they don't merge; a
project-level file with its own entry silently shadows the user-level
registration.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from broker.claude.atomic import atomic_update_json
from broker.claude.paths import settings_path
from broker.protocol.constants import HOOK_SETTINGS_TIMEOUT

# Substring of our hook command used to identify OUR entries during repair.
BROKER_HOOK_MARKER = "broker-hook"

# The three rule lists Claude Code reads out of a settings "permissions" block.
_RULE_LISTS = ("allow", "ask", "deny")


@dataclass
class RepairReport:
    repaired_events: list[str] = field(default_factory=list[str])
    warnings: list[str] = field(default_factory=list[str])


def hook_entry(command: str) -> dict[str, Any]:
    """Build the settings.json hooks entry that runs ``command``."""
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
    """Report whether ``entry`` is one of ours, by hook command marker."""
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
    """Bring our entry for each event up to ``command``, adding it if absent.

    Args:
        data: Parsed settings object; mutated in place.
        events: Hook event names that must carry our entry.
        command: Hook command to install; must contain ``BROKER_HOOK_MARKER``.

    Returns:
        The event names that were added or rewritten.

    Raises:
        ValueError: ``command`` lacks the marker, or ``hooks``/an event's
            entries are not of the expected type.
    """
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
        desired = hook_entry(command)
        ours = [i for i, entry in enumerate(entries) if _entry_is_ours(entry)]
        if not ours:
            entries.append(desired)
            added.append(event)
            continue
        if all(entries[i] == desired for i in ours):
            continue
        for i in ours:
            entries[i] = desired
        added.append(event)
    return added


def register_hooks(
    events: list[str], command: str, path: Path | None = None
) -> None:
    """Register our hook entry for ``events`` in Claude Code's settings.

    Args:
        events: Hook event names to register under.
        command: Hook command to install; must contain ``BROKER_HOOK_MARKER``.
        path: Settings file to update; defaults to user-level ``settings.json``.
    """
    target = path if path is not None else settings_path()

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        """Add any missing entries to the settings object."""
        _ensure_registered(data, events, command)
        return data

    atomic_update_json(target, mutate)


def write_session_permissions(path: Path, rules: dict[str, Any]) -> None:
    """Write one session's native permission rules to its own settings file.

    Args:
        path: Settings file to write; every key outside ``permissions`` is
            left as it was found.
        rules: Rule lists keyed ``allow``, ``ask`` and ``deny``.

    Raises:
        ValueError: ``rules`` is missing one of the three rule lists.
    """
    missing = [key for key in _RULE_LISTS if key not in rules]
    if missing:
        # Writing the absent lists as empty would hand the session a weaker
        # rule set than the developer configured, silently.
        raise ValueError(f"permission rules missing {missing}; refusing to write")
    permissions = {key: list(rules[key]) for key in _RULE_LISTS}

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        """Replace the permissions block, leaving every other key alone."""
        data["permissions"] = permissions
        return data

    atomic_update_json(path, mutate)


def verify_and_repair(
    events: list[str],
    command: str,
    path: Path | None = None,
    *,
    shadow_candidates: list[Path] | None = None,
) -> RepairReport:
    """Re-add any broker hook entries missing from the settings file.

    Args:
        events: Hook event names that must carry our entry.
        command: Hook command to install; must contain ``BROKER_HOOK_MARKER``.
        path: Settings file to verify; defaults to user-level ``settings.json``.
        shadow_candidates: Project-level files to check for shadowing entries.

    Returns:
        The events repaired and the warnings raised.
    """
    target = path if path is not None else settings_path()
    report = RepairReport()

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        """Add any missing entries, recording which events were repaired."""
        report.repaired_events = _ensure_registered(data, events, command)
        return data

    atomic_update_json(target, mutate)
    if report.repaired_events:
        report.warnings.append(
            f"re-registered broker hook entries for {report.repaired_events}"
        )
    # Settings scopes REPLACE the whole hooks array per event, they don't
    # merge — a project-level file defining an event without our marker
    # silently shadows the user-level registration for that event.
    for candidate in shadow_candidates or []:
        _check_shadow(candidate, events, report)
    return report


def _check_shadow(
    candidate: Path, events: list[str], report: RepairReport
) -> None:
    """Warn when ``candidate`` shadows our registration for any of ``events``.

    Args:
        candidate: Project-level settings file to inspect.
        events: Hook event names to check for shadowing.
        report: Report whose warnings are appended to in place.
    """
    if not candidate.exists():
        return
    try:
        parsed: Any = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.warnings.append(
            f"{candidate}: unreadable ({exc}); cannot check for hook shadowing"
        )
        return
    if not isinstance(parsed, dict):
        return
    raw_hooks = cast(dict[str, Any], parsed).get("hooks")
    if not isinstance(raw_hooks, dict):
        return
    hooks_by_event = cast(dict[str, Any], raw_hooks)
    for event in events:
        raw_entries = hooks_by_event.get(event)
        if not isinstance(raw_entries, list):
            continue
        entries = cast(list[Any], raw_entries)
        if not any(_entry_is_ours(entry) for entry in entries):
            report.warnings.append(
                f"{candidate} defines {event} hooks without the broker "
                "marker — it SHADOWS the user-level registration for that "
                "event"
            )
