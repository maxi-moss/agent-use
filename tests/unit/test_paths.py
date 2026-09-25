"""BrokerPaths roots every broker-home path at the value it is given."""

from pathlib import Path

import pytest

from broker.paths import BrokerPaths
from broker.config import ClassifierConfig, EmbeddingConfig, SessionBrokerConfig


def test_escalation_queue_path() -> None:
    """The queue file lives directly under the broker home."""
    paths = BrokerPaths(Path("/private/tmp/broker-home"))
    path = paths.escalation_queue
    assert isinstance(path, Path)
    assert path.parent == paths.home
    assert path.name == "escalation-queue.json"


def test_session_paths_follow_the_configured_home_not_the_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The configured broker_home wins over $BROKER_HOME for session paths."""
    monkeypatch.setenv("BROKER_HOME", str(tmp_path / "env"))
    configured = tmp_path / "configured"
    cfg = SessionBrokerConfig(
        name="s1",
        socket_path=str(configured / "s" / "s1.sock"),
        master_socket_path=str(configured / "master.sock"),
        broker_home=configured,
        cwd=str(tmp_path),
        anchor_pane="w3:p1",
        intent="i",
        model_id="test-model",
        max_tokens=1024,
        classifier=ClassifierConfig(),
        embedding=EmbeddingConfig(),
        watchdog_seconds=300.0,
        budget_max=8,
        claude_settings_path=str(configured / "claude-settings.json"),
    )
    paths = BrokerPaths(cfg.broker_home)
    for path in (
        paths.session_decisions(cfg.name),
        paths.session_permissions(cfg.name),
        paths.session_claude_settings(cfg.name),
        paths.session_hook_log(cfg.name),
    ):
        assert path.is_relative_to(configured)
        assert not path.is_relative_to(tmp_path / "env")


def test_index_db_is_deterministic_and_under_the_home() -> None:
    """Two BrokerPaths for the same home name the same index file for a repo."""
    home = Path("/private/tmp/broker-home")
    repo = Path("/private/tmp/some-repo")
    first = BrokerPaths(home).index_db(repo)
    second = BrokerPaths(home).index_db(repo)
    assert first == second
    assert first.parent == home / "index"
    assert first.suffix == ".sqlite"
    assert first != BrokerPaths(home).index_db(Path("/private/tmp/other-repo"))
