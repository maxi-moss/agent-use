"""Pre-seed folder trust in ~/.claude.json (spike-verified key, spec §9.1 gap).

The trust key is UNDOCUMENTED — verified only by observation on 2.1.220
(spikes/README.md §6). Re-verify on every Claude Code upgrade before trusting
VALIDATED_AGAINST here.
"""

from pathlib import Path
from typing import Any, cast

from broker.claude.atomic import AtomicWriteError, atomic_update_json
from broker.claude.paths import claude_json_path

VALIDATED_AGAINST = "2.1.220"


def seed_trust(project_path: Path, path: Path | None = None) -> None:
    """Set projects.<abs-path>.hasTrustDialogAccepted = true, creating nodes as
    needed and preserving every sibling key."""
    if not project_path.is_absolute():
        raise ValueError(f"project path must be absolute: {project_path}")
    target = path if path is not None else claude_json_path()

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        raw_projects = data.setdefault("projects", {})
        if not isinstance(raw_projects, dict):
            raise AtomicWriteError(
                f"{target}: 'projects' is not an object; refusing to touch it"
            )
        projects = cast(dict[str, Any], raw_projects)
        raw_project = projects.setdefault(str(project_path), {})
        if not isinstance(raw_project, dict):
            raise AtomicWriteError(
                f"{target}: projects[{str(project_path)!r}] is not an object; "
                "refusing to touch it"
            )
        cast(dict[str, Any], raw_project)["hasTrustDialogAccepted"] = True
        return data

    atomic_update_json(target, mutate)
