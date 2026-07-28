"""BrokerConfig defaults, overlay loading, fail-loud invalid config."""

import json
from pathlib import Path

import pytest

from broker.config import BrokerConfig, ConfigError, load

SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "broker"


def test_defaults() -> None:
    cfg = BrokerConfig()
    assert cfg.model_id == "claude-opus-5"
    assert cfg.max_tokens == 8192
    assert cfg.watchdog_seconds == 300.0
    assert cfg.budget_max == 8
    assert cfg.recent_turns_window == 20


def test_broker_home_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BROKER_HOME", str(tmp_path))
    assert BrokerConfig().broker_home == tmp_path


def test_load_overlay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BROKER_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"budget_max": 2}))
    cfg = load()
    assert cfg.budget_max == 2
    assert cfg.model_id == "claude-opus-5"  # untouched default
    assert cfg.broker_home == tmp_path


def test_load_invalid_json_fails_loud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BROKER_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text("{not json")
    with pytest.raises(ConfigError):
        load()


def test_load_unknown_key_fails_loud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BROKER_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"budgetmax": 2}))
    with pytest.raises(ConfigError):
        load()


def test_load_missing_file_is_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BROKER_HOME", str(tmp_path))
    assert load() == BrokerConfig(broker_home=tmp_path)


def test_model_id_pinned_in_exactly_one_module() -> None:
    """`claude-opus-5` lives in broker/config.py and nowhere else in src."""
    hits: list[Path] = []
    for path in SRC_ROOT.rglob("*"):
        if path.suffix not in {".py", ".md"} or not path.is_file():
            continue
        if "claude-opus-5" in path.read_text(encoding="utf-8"):
            hits.append(path.relative_to(SRC_ROOT))
    assert hits == [Path("config.py")], f"model id leaked into {hits}"


def test_config_module_has_no_other_model_literal() -> None:
    """The pin is a default on BrokerConfig, present exactly once."""
    text = (SRC_ROOT / "config.py").read_text(encoding="utf-8")
    assert text.count("claude-opus-5") == 1
