"""Architectural rules that import-linter cannot express as contracts.

One test per rule, named after the rule it enforces.
"""

import re
from pathlib import Path

SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "broker"

_DRIVER_PATH = Path("herdr/driver.py")


def _modules_mentioning(needle: str) -> list[Path]:
    """Every file under src/broker containing ``needle``."""
    hits: list[Path] = []
    for path in SRC_ROOT.rglob("*"):
        if path.suffix not in {".py", ".md"} or not path.is_file():
            continue
        if needle in path.read_text(encoding="utf-8"):
            hits.append(path.relative_to(SRC_ROOT))
    return hits


def _modules_using_capability(name: str) -> list[Path]:
    """Every .py file under src/broker with a bare-word use of ``name``."""
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    hits: list[Path] = []
    for path in SRC_ROOT.rglob("*.py"):
        if pattern.search(path.read_text(encoding="utf-8")):
            hits.append(path.relative_to(SRC_ROOT))
    return hits


def _assert_capability_confined_to(name: str, package: str) -> None:
    """Every use of ``name`` lives under ``package`` or in herdr/driver.py."""
    for path in _modules_using_capability(name):
        if path == _DRIVER_PATH:
            continue
        assert path.parts[0] == package, (
            f"{name} used outside broker.{package} and {_DRIVER_PATH}: {path}"
        )


def test_model_id_pinned_in_exactly_one_module() -> None:
    """`claude-sonnet-5` lives in broker/config.py and nowhere else in src."""
    hits = _modules_mentioning("claude-sonnet-5")
    assert hits == [Path("config.py")], f"model id leaked into {hits}"


def test_classifier_model_id_pinned_in_exactly_one_module() -> None:
    """`claude-haiku-4-5` lives in broker/config.py and nowhere else in src."""
    hits = _modules_mentioning("claude-haiku-4-5")
    assert hits == [Path("config.py")], f"classifier model id leaked into {hits}"


def test_send_to_claude_capability_is_session_only() -> None:
    """agent_prompt/agent_start/pane_split/agent_get are session-only capabilities."""
    for name in ("agent_prompt", "agent_start", "pane_split", "agent_get"):
        _assert_capability_confined_to(name, "session")


def test_notify_capability_is_master_only() -> None:
    """notification_show is a master-only capability."""
    _assert_capability_confined_to("notification_show", "master")
