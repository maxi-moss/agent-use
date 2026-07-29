"""settings.py: idempotent registration, foreign-entry preservation, repair."""

import json
from pathlib import Path
from typing import Any

import pytest

from broker.claude.atomic import AtomicWriteError
from broker.claude.settings import (
    BROKER_HOOK_MARKER,
    hook_entry,
    register_hooks,
    verify_and_repair,
    write_session_permissions,
)
COMMAND = "/usr/bin/env python3 -m broker.hook  # broker-hook"
EVENTS = ["PreToolUse", "Stop", "SessionStart"]

RULES: dict[str, Any] = {
    "allow": ["Read", "Glob"],
    "ask": ["Bash(git push:*)"],
    "deny": [],
}

# A realistic foreign entry mimicking Herdr's integration hook.
FAKE_HERDR_ENTRY: dict[str, Any] = {
    "matcher": "*",
    "hooks": [
        {
            "type": "command",
            "command": "/opt/homebrew/bin/herdr integration report",
            "timeout": 10,
        }
    ],
}


def test_command_without_marker_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        register_hooks(EVENTS, "python3 -m something.else", tmp_path / "s.json")


def test_registration_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    register_hooks(EVENTS, COMMAND, target)
    register_hooks(EVENTS, COMMAND, target)  # double-register
    data = json.loads(target.read_text())
    for event in EVENTS:
        ours = [
            e
            for e in data["hooks"][event]
            if BROKER_HOOK_MARKER in json.dumps(e)
        ]
        assert len(ours) == 1  # one entry, not two


def test_foreign_entries_survive_byte_identical(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    before = {
        "model": "opus",
        "hooks": {
            "PreToolUse": [FAKE_HERDR_ENTRY],
            "SessionEnd": [FAKE_HERDR_ENTRY],
        },
    }
    target.write_text(json.dumps(before, indent=2))
    register_hooks(EVENTS, COMMAND, target)
    data = json.loads(target.read_text())
    # foreign entries verbatim
    assert data["hooks"]["PreToolUse"][0] == FAKE_HERDR_ENTRY
    assert data["hooks"]["SessionEnd"] == [FAKE_HERDR_ENTRY]
    assert data["model"] == "opus"
    # ours appended after the foreign one
    assert data["hooks"]["PreToolUse"][1] == hook_entry(COMMAND)


def test_verify_and_repair_readds_removed_entry(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    register_hooks(EVENTS, COMMAND, target)

    # simulate an upgrade/manual edit removing one of our entries
    data = json.loads(target.read_text())
    data["hooks"]["Stop"] = [FAKE_HERDR_ENTRY]  # ours gone, foreign remains
    target.write_text(json.dumps(data))

    report = verify_and_repair(EVENTS, COMMAND, target)
    assert report.repaired_events == ["Stop"]
    assert report.warnings  # loud

    repaired = json.loads(target.read_text())
    assert repaired["hooks"]["Stop"][0] == FAKE_HERDR_ENTRY
    assert repaired["hooks"]["Stop"][1] == hook_entry(COMMAND)


def test_verify_and_repair_clean_reports_nothing(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    register_hooks(EVENTS, COMMAND, target)
    report = verify_and_repair(EVENTS, COMMAND, target)
    assert report.repaired_events == []
    assert report.warnings == []


def test_session_permissions_creates_file_and_parents(tmp_path: Path) -> None:
    target = tmp_path / "sessions" / "s1" / "claude-settings.json"
    write_session_permissions(target, RULES)
    assert json.loads(target.read_text()) == {"permissions": RULES}
    # the temp file the atomic write renames from must not survive it
    assert [p.name for p in target.parent.iterdir()] == [target.name]


def test_session_permissions_leaves_foreign_keys_intact(tmp_path: Path) -> None:
    target = tmp_path / "claude-settings.json"
    before: dict[str, Any] = {
        "model": "opus",
        "env": {"SOMETHING": "1"},
        "permissions": {"allow": ["Write"], "ask": [], "deny": []},
    }
    target.write_text(json.dumps(before, indent=2))
    write_session_permissions(target, RULES)
    after = json.loads(target.read_text())
    assert after["model"] == "opus"
    assert after["env"] == {"SOMETHING": "1"}
    assert after["permissions"] == RULES


def test_session_permissions_refuses_to_clobber_invalid_json(
    tmp_path: Path,
) -> None:
    target = tmp_path / "claude-settings.json"
    target.write_text("{not json")
    with pytest.raises(AtomicWriteError):
        write_session_permissions(target, RULES)
    assert target.read_text() == "{not json"


def test_session_permissions_refuses_incomplete_rules(tmp_path: Path) -> None:
    """A missing list must fail loud, not be written out as an empty one."""
    target = tmp_path / "claude-settings.json"
    with pytest.raises(ValueError):
        write_session_permissions(target, {"allow": ["Read"]})
    assert not target.exists()
