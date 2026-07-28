"""Claude Code path resolution. Owns CLAUDE_CONFIG_DIR handling."""

import os
import re
from pathlib import Path


def config_dir() -> Path:
    """Resolve Claude Code's config directory.

    Returns:
        The config directory. Existence is not checked.
    """
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude"


def settings_path() -> Path:
    """Return the path to Claude Code's user settings file."""
    return config_dir() / "settings.json"


def claude_json_path() -> Path:
    """Return the path to ``~/.claude.json``, a separate global state file."""
    return Path.home() / ".claude.json"


def munge_project_path(cwd: Path) -> str:
    """Munge a path into the name Claude Code gives its project directory.

    Args:
        cwd: Path to munge, taken in its string form.

    Returns:
        The munged directory name.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def transcript_dir_for_cwd(cwd: Path) -> Path:
    """Return the transcript directory Claude Code uses for ``cwd``."""
    return config_dir() / "projects" / munge_project_path(cwd)
