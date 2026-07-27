"""Atomic JSON update: read -> validate -> backup -> temp-write -> rename.

The one sanctioned write path for anything outside the repo (CLAUDE.md global
rule). Refuses to overwrite a file that fails JSON validation — a corrupted
settings file is the developer's to inspect, never ours to clobber.
"""

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast


class AtomicWriteError(Exception):
    """The existing file is unreadable as JSON, or the update cannot proceed."""


def atomic_update_json(
    path: Path,
    mutate: Callable[[dict[str, Any]], dict[str, Any]],
    backup_suffix: str = ".broker-backup",
) -> dict[str, Any]:
    """Apply `mutate` to the JSON object at `path`, atomically. Returns the
    written object. A missing file starts as {}; an invalid one is fatal."""
    original_bytes: bytes | None = None
    data: dict[str, Any] = {}
    if path.exists():
        original_bytes = path.read_bytes()
        try:
            parsed: Any = json.loads(original_bytes)
        except json.JSONDecodeError as exc:
            raise AtomicWriteError(
                f"{path} is not valid JSON ({exc.msg}); refusing to overwrite"
            ) from exc
        if not isinstance(parsed, dict):
            raise AtomicWriteError(
                f"{path} is not a JSON object; refusing to overwrite"
            )
        data = cast(dict[str, Any], parsed)

    updated = mutate(dict(data))
    if not isinstance(updated, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise AtomicWriteError("mutate() must return a dict")

    path.parent.mkdir(parents=True, exist_ok=True)
    if original_bytes is not None:
        backup_path = path.with_name(path.name + backup_suffix)
        backup_path.write_bytes(original_bytes)

    fd, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return updated
