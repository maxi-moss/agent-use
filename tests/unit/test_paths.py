"""BrokerPaths roots every broker-home path at the value it is given."""

from pathlib import Path

import pytest

from broker.paths import BrokerPaths
from broker.config import SessionBrokerConfig


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
        watchdog_seconds=300.0,
        budget_max=8,
    )
    decisions = BrokerPaths(cfg.broker_home).session_decisions(cfg.name)
    assert decisions.is_relative_to(configured)
    assert not decisions.is_relative_to(tmp_path / "env")
