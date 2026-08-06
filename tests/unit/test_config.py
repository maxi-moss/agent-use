"""BrokerConfig defaults, overlay loading, fail-loud invalid config."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from broker.config import (
    AdoptedSession,
    BrokerConfig,
    ClassifierConfig,
    ConfigError,
    PermissionRules,
    ResumedTask,
    SessionBrokerConfig,
    load,
)

SRC_ROOT = Path(__file__).parent.parent.parent / "src" / "broker"


def _modules_mentioning(needle: str) -> list[Path]:
    """Every file under src/broker containing ``needle``."""
    hits: list[Path] = []
    for path in SRC_ROOT.rglob("*"):
        if path.suffix not in {".py", ".md"} or not path.is_file():
            continue
        if needle in path.read_text(encoding="utf-8"):
            hits.append(path.relative_to(SRC_ROOT))
    return hits


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
    assert cfg.model_id == "claude-sonnet-5"  # untouched default
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
    """`claude-sonnet-5` lives in broker/config.py and nowhere else in src."""
    hits = _modules_mentioning("claude-sonnet-5")
    assert hits == [Path("config.py")], f"model id leaked into {hits}"


def test_classifier_model_id_pinned_in_exactly_one_module() -> None:
    """`claude-haiku-4-5` lives in broker/config.py and nowhere else in src."""
    hits = _modules_mentioning("claude-haiku-4-5")
    assert hits == [Path("config.py")], f"classifier model id leaked into {hits}"


def test_classifier_defaults() -> None:
    assert ClassifierConfig().model_id == "claude-haiku-4-5"
    assert ClassifierConfig().max_tokens == 1024
    assert BrokerConfig().classifier == ClassifierConfig()


@pytest.mark.parametrize("rule", ["Bash", "Bash(*)"])
def test_blanket_bash_allow_rule_is_rejected(rule: str) -> None:
    with pytest.raises(ValueError):
        PermissionRules(allow=[rule])


def test_bypass_permissions_is_rejected_in_every_list() -> None:
    banned = ["bypassPermissions"]
    with pytest.raises(ValueError):
        PermissionRules(allow=banned)
    with pytest.raises(ValueError):
        PermissionRules(ask=banned)
    with pytest.raises(ValueError):
        PermissionRules(deny=banned)


def test_blanket_bash_is_allow_only() -> None:
    """`Bash` is a legitimate ask/deny rule — only allow-ing it is the hazard."""
    assert PermissionRules(ask=["Bash"], deny=["Bash(*)"]).ask == ["Bash"]


def test_load_overlay_carries_nested_permission_keys(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BROKER_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "classifier": {"max_tokens": 256},
                "permission_rules": {
                    "allow": ["Read"],
                    "ask": ["Bash(git push:*)"],
                    "deny": ["Bash(curl:*)"],
                },
            }
        )
    )
    cfg = load()
    assert cfg.classifier.max_tokens == 256
    assert cfg.classifier.model_id == "claude-haiku-4-5"  # untouched default
    assert cfg.permission_rules.deny == ["Bash(curl:*)"]


def _session_config_kwargs() -> dict[str, Any]:
    """Every required SessionBrokerConfig field, without adopt or resume."""
    return {
        "name": "s1",
        "socket_path": "/private/tmp/s/s1.sock",
        "master_socket_path": "/private/tmp/m.sock",
        "broker_home": Path("/private/tmp/broker-home"),
        "cwd": "/private/tmp/work",
        "anchor_pane": "%1",
        "intent": "the raw intent",
        "model_id": "test-model",
        "max_tokens": 1024,
        "classifier": ClassifierConfig(),
        "watchdog_seconds": 300.0,
        "budget_max": 8,
        "claude_settings_path": "/private/tmp/claude-settings.json",
    }


def test_resume_requires_adopt() -> None:
    with pytest.raises(ValidationError) as exc:
        SessionBrokerConfig(
            **_session_config_kwargs(),
            resume=ResumedTask(approved_prompt="the first task"),
        )
    assert "adopt" in str(exc.value)


def test_resume_round_trips() -> None:
    cfg = SessionBrokerConfig(
        **_session_config_kwargs(),
        budget_count=6,
        adopt=AdoptedSession(
            pane_id="w1:p1",
            claude_session_id="cc-1",
            transcript_path="/private/tmp/t.jsonl",
        ),
        resume=ResumedTask(approved_prompt="the first task", completed=True),
    )
    round_tripped = SessionBrokerConfig.model_validate_json(
        cfg.model_dump_json()
    )
    assert round_tripped == cfg
    assert round_tripped.resume is not None
    assert round_tripped.resume.approved_prompt == "the first task"
    assert round_tripped.resume.completed is True
    assert round_tripped.budget_count == 6


def test_overlay_with_a_self_defeating_rule_fails_loud(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BROKER_HOME", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"permission_rules": {"allow": ["Bash(*)"]}})
    )
    with pytest.raises(ConfigError):
        load()
