"""settings.py: idempotent registration, foreign-entry preservation, repair."""

import json
from pathlib import Path
from typing import Any

import pytest

from broker.claude.settings import (
    BROKER_HOOK_MARKER,
    hook_entry,
    register_hooks,
    verify_and_repair,
)
from broker.protocol.constants import HOOK_SETTINGS_TIMEOUT

COMMAND = "/usr/bin/env python3 -m broker.hook  # broker-hook"
EVENTS = ["PreToolUse", "Stop", "SessionStart"]

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


def test_hook_entry_shape() -> None:
    entry = hook_entry(COMMAND)
    assert entry == {
        "hooks": [
            {
                "type": "command",
                "command": COMMAND,
                "timeout": HOOK_SETTINGS_TIMEOUT,
            }
        ]
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
