"""Pre-seed folder trust in ~/.claude.json.

The trust key is UNDOCUMENTED — verified only by observation on 2.1.220.
Re-verify on every Claude Code upgrade before trusting TRUST_KEY_VALIDATED_AGAINST.
"""

from pathlib import Path
from typing import Any, cast

from broker.claude.atomic import AtomicWriteError, atomic_update_json
from broker.claude.paths import claude_json_path

TRUST_KEY_VALIDATED_AGAINST = "2.1.220"


def seed_trust(project_path: Path, path: Path | None = None) -> None:
    """Pre-seed folder trust for ``project_path`` in ``~/.claude.json``.

    Args:
        project_path: Absolute path of the project to trust.
        path: State file to update; defaults to ``~/.claude.json``.

    Raises:
        ValueError: ``project_path`` is not absolute.
        AtomicWriteError: ``projects`` or this project's entry exists but
            isn't a JSON object.
    """
    if not project_path.is_absolute():
        raise ValueError(f"project path must be absolute: {project_path}")
    target = path if path is not None else claude_json_path()

    def mutate(data: dict[str, Any]) -> dict[str, Any]:
        """Set the trust flag on this project's entry."""
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
