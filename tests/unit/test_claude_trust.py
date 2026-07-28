"""trust.py: seed fresh file, preserve siblings, absolute-path key format."""

import json
from pathlib import Path

import pytest

from broker.claude.trust import seed_trust


def test_seeds_fresh_file(tmp_path: Path) -> None:
    target = tmp_path / ".claude.json"
    seed_trust(Path("/private/tmp/worktree-1"), target)
    data = json.loads(target.read_text())
    assert data == {
        "projects": {"/private/tmp/worktree-1": {"hasTrustDialogAccepted": True}}
    }


def test_preserves_existing_projects_and_siblings(tmp_path: Path) -> None:
    target = tmp_path / ".claude.json"
    before = {
        "numStartups": 42,
        "projects": {
            "/private/tmp/bspike2": {
                "hasTrustDialogAccepted": True,
                "allowedTools": ["Bash"],
            },
            "/private/tmp/worktree-1": {
                "history": ["old prompt"],
            },
        },
    }
    target.write_text(json.dumps(before))
    seed_trust(Path("/private/tmp/worktree-1"), target)
    data = json.loads(target.read_text())
    assert data["numStartups"] == 42
    assert data["projects"]["/private/tmp/bspike2"] == {
        "hasTrustDialogAccepted": True,
        "allowedTools": ["Bash"],
    }
    # sibling key inside the seeded project preserved
    assert data["projects"]["/private/tmp/worktree-1"] == {
        "history": ["old prompt"],
        "hasTrustDialogAccepted": True,
    }


def test_relative_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        seed_trust(Path("relative/worktree"), tmp_path / ".claude.json")
