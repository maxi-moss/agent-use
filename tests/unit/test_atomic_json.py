"""atomic_json.py: round-trip, invalid-JSON refusal, backup, no-op skip, crash-window."""

import json
import os
from pathlib import Path
from typing import Any

import pytest

from broker.atomic_json import AtomicWriteError, atomic_update_json


def test_round_trip_missing_file_starts_empty(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        assert data == {}
        data["key"] = "value"
        return data

    written = atomic_update_json(target, mutate, backup=False)
    assert written == {"key": "value"}
    assert json.loads(target.read_text()) == {"key": "value"}


def test_round_trip_preserves_existing_keys(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text('{"existing": [1, 2], "other": {"nested": true}}')

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data["new"] = 3
        return data

    atomic_update_json(target, mutate, backup=False)
    assert json.loads(target.read_text()) == {
        "existing": [1, 2],
        "other": {"nested": True},
        "new": 3,
    }


def test_invalid_json_refused_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    corrupt = '{"broken": '
    target.write_text(corrupt)
    with pytest.raises(AtomicWriteError):
        atomic_update_json(target, lambda d: d, backup=False)
    assert target.read_text() == corrupt  # untouched


def test_non_object_json_refused(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text("[1, 2, 3]")
    with pytest.raises(AtomicWriteError):
        atomic_update_json(target, lambda d: d, backup=False)
    assert target.read_text() == "[1, 2, 3]"


def test_backup_holds_original_bytes(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    original = '{"a": 1}'
    target.write_text(original)

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data["a"] = 2
        return data

    atomic_update_json(target, mutate, backup=True)
    backup = tmp_path / "settings.json.broker-backup"
    assert backup.read_text() == original
    assert json.loads(target.read_text()) == {"a": 2}


def test_unchanged_mutate_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text('{"a": {"b": [1]}}')
    os.utime(target, ns=(1_000_000_000, 1_000_000_000))

    atomic_update_json(target, lambda d: d, backup=True)
    assert target.stat().st_mtime_ns == 1_000_000_000
    assert [p.name for p in tmp_path.iterdir()] == ["settings.json"]


def test_nested_in_place_mutation_is_written(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text('{"hooks": {"Stop": []}}')

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data["hooks"]["Stop"].append("ours")
        return data

    atomic_update_json(target, mutate, backup=False)
    assert json.loads(target.read_text()) == {"hooks": {"Stop": ["ours"]}}


def test_crash_window_leftover_temp_file_is_ignored(tmp_path: Path) -> None:
    """A temp file left by a crashed writer must not affect the next run."""
    target = tmp_path / "settings.json"
    target.write_text('{"a": 1}')
    leftover = tmp_path / ".settings.json.crashed.tmp"
    leftover.write_text('{"partial": ')

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        data["a"] = 2
        return data

    atomic_update_json(target, mutate, backup=False)
    assert json.loads(target.read_text()) == {"a": 2}
    assert leftover.exists()  # ignored, not consumed


def test_mutate_exception_leaves_target_untouched(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text('{"a": 1}')

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        atomic_update_json(target, mutate, backup=False)
    assert json.loads(target.read_text()) == {"a": 1}
    # no stray temp files holding a partial write
    temps = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert temps == []
