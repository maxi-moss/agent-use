"""trust.py: seed fresh file, preserve siblings, absolute-path key format."""

import json
from pathlib import Path

import pytest

from broker.claude.trust import VALIDATED_AGAINST, seed_trust


def test_validated_against_pinned() -> None:
    # gotcha 7: undocumented key — re-verify on every Claude Code upgrade
    assert VALIDATED_AGAINST == "2.1.220"


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


def test_key_is_absolute_path_string(tmp_path: Path) -> None:
    target = tmp_path / ".claude.json"
    seed_trust(Path("/private/tmp/deep/nested/worktree"), target)
    data = json.loads(target.read_text())
    assert list(data["projects"]) == ["/private/tmp/deep/nested/worktree"]


def test_relative_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        seed_trust(Path("relative/worktree"), tmp_path / ".claude.json")
