"""Claude Code path resolution. Owns CLAUDE_CONFIG_DIR handling (gotcha 13)."""

import os
import re
from pathlib import Path


def config_dir() -> Path:
    """$CLAUDE_CONFIG_DIR if set, else ~/.claude. Never hardcode elsewhere."""
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".claude"


def settings_path() -> Path:
    return config_dir() / "settings.json"


def claude_json_path() -> Path:
    """~/.claude.json is a SEPARATE global state file (trust, onboarding),
    not under config_dir — it does not relocate with CLAUDE_CONFIG_DIR."""
    return Path.home() / ".claude.json"


def munge_project_path(cwd: Path) -> str:
    """EVERY non-alphanumeric char becomes '-' (gotcha 10; underscore included):
    /Users/maxi/side_projects/agent-use -> -Users-maxi-side-projects-agent-use
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(cwd))


def transcript_dir_for_cwd(cwd: Path) -> Path:
    return config_dir() / "projects" / munge_project_path(cwd)
