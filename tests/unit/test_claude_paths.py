"""paths.py: CLAUDE_CONFIG_DIR honoured, munging is every-non-alphanumeric."""

from pathlib import Path

import pytest

from broker.claude import paths


def test_config_dir_default_is_home_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert paths.config_dir() == Path.home() / ".claude"


def test_config_dir_honours_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/private/tmp/claude-alt")
    assert paths.config_dir() == Path("/private/tmp/claude-alt")
    assert paths.settings_path() == Path("/private/tmp/claude-alt/settings.json")


def test_claude_json_is_not_under_config_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/private/tmp/claude-alt")
    assert paths.claude_json_path() == Path.home() / ".claude.json"


@pytest.mark.parametrize(
    "cwd,expected",
    [
        ("/private/tmp/broker-spike", "-private-tmp-broker-spike"),
        (
            # underscore becomes dash too — verified on 2.1.220
            "/Users/maxi/side_projects/agent-use",
            "-Users-maxi-side-projects-agent-use",
        ),
    ],
)
def test_munging_every_non_alphanumeric(cwd: str, expected: str) -> None:
    assert paths.munge_project_path(Path(cwd)) == expected


def test_transcript_dir_for_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/private/tmp/claude-alt")
    assert paths.transcript_dir_for_cwd(
        Path("/private/tmp/broker-spike")
    ) == Path("/private/tmp/claude-alt/projects/-private-tmp-broker-spike")
