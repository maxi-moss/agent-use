"""Herdr driver: argv construction, JSON parsing, error mapping. No live calls."""

import subprocess
from pathlib import Path
from typing import Any

import pytest

from broker.herdr import driver
from broker.herdr.driver import HerdrError

FIXTURES = Path(__file__).parent.parent / "fixtures" / "herdr"


class FakeRun:
    """Replaces subprocess.run inside the driver; records every argv."""

    def __init__(
        self, stdout: str = "", stderr: str = "", returncode: int = 0
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[list[str]] = []
        self.timeouts: list[float] = []

    def __call__(
        self,
        argv: list[str],
        capture_output: bool = False,
        text: bool = False,
        timeout: float = 0.0,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        self.timeouts.append(timeout)
        return subprocess.CompletedProcess(
            argv, self.returncode, self.stdout, self.stderr
        )


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(**kwargs: Any) -> FakeRun:
        run = FakeRun(**kwargs)
        monkeypatch.setattr(driver.subprocess, "run", run)
        return run

    return install


def test_status_parses_real_recorded_output(fake: Any) -> None:
    run = fake(stdout=(FIXTURES / "status.json").read_text())
    result = driver.status()
    assert run.calls == [["herdr", "status", "--json"]]
    assert result.client.version == "0.7.5"
    assert result.server.version == "0.7.5"
    assert result.compatible is True


def test_pane_split_argv_and_pane_id(fake: Any) -> None:
    run = fake(stdout=(FIXTURES / "pane_split.json").read_text())
    info = driver.pane_split(
        "w3:p1",
        direction="right",
        cwd=Path("/private/tmp/work"),
        env={"BROKER_SOCKET": "/private/tmp/work/b.sock"},
        focus=False,
        timeout_s=15.0,
    )
    assert run.calls == [[
        "herdr", "pane", "split",
        "--pane", "w3:p1",
        "--direction", "right",
        "--cwd", "/private/tmp/work",
        "--env", "BROKER_SOCKET=/private/tmp/work/b.sock",
        "--no-focus",
    ]]
    assert info.pane_id == "w3:p2"


def test_agent_start_with_session(fake: Any) -> None:
    run = fake(stdout=(FIXTURES / "agent_start_with_session.json").read_text())
    result = driver.agent_start(
        "sess-a1", kind="claude", pane_id="w3:p2", timeout_ms=30000
    )
    assert run.calls == [[
        "herdr", "agent", "start", "sess-a1",
        "--kind", "claude",
        "--pane", "w3:p2",
        "--timeout", "30000",
    ]]
    assert result.interactive_ready is True
    assert result.agent_session is not None
    assert result.agent_session.value == "7dfd77f8-a848-4a35-9122-d5343e019082"


def test_agent_start_without_session(fake: Any) -> None:
    """agent_session absent when the trust dialog blocked init — must parse."""
    fake(stdout=(FIXTURES / "agent_start_no_session.json").read_text())
    result = driver.agent_start(
        "sess-a1", kind="claude", pane_id="w3:p2", timeout_ms=30000
    )
    assert result.interactive_ready is True
    assert result.agent_session is None


def test_submit_prompt_is_the_two_step(fake: Any) -> None:
    """agent prompt types only; submit is prompt + pane send-keys enter."""
    run = fake(stdout="{}")
    driver.submit_prompt("sess-a1", "w3:p2", "do the thing", timeout_s=5.0)
    assert run.calls == [
        ["herdr", "agent", "prompt", "sess-a1", "do the thing"],
        ["herdr", "pane", "send-keys", "w3:p2", "enter"],
    ]


def test_agent_wait_requires_explicit_timeout_in_argv(fake: Any) -> None:
    run = fake(stdout="{}")
    driver.agent_wait("sess-a1", until=["idle", "blocked"], timeout_ms=60000)
    assert run.calls == [[
        "herdr", "agent", "wait", "sess-a1",
        "--until", "idle",
        "--until", "blocked",
        "--timeout", "60000",
    ]]


def test_agent_wait_requires_until() -> None:
    with pytest.raises(ValueError):
        driver.agent_wait("sess-a1", until=[], timeout_ms=1000)


def test_exit_1_maps_recorded_stalled_error(fake: Any) -> None:
    fake(
        stderr=(FIXTURES / "agent_prompt_stalled_stderr.json").read_text(),
        returncode=1,
    )
    with pytest.raises(HerdrError) as exc_info:
        driver.agent_prompt("sess-a1", "hello", timeout_s=5.0)
    assert exc_info.value.code == "agent_prompt_stalled"
    assert "5000 ms" in exc_info.value.message


def test_exit_2_is_our_bug(fake: Any) -> None:
    fake(stderr="usage: herdr ...", returncode=2)
    with pytest.raises(RuntimeError):
        driver.pane_close("w3:p2", timeout_s=5.0)


def test_no_public_api_emits_current(fake: Any) -> None:
    """constraint 14: --current must be rejected everywhere a target goes."""
    run = fake(stdout="{}")
    for call in (
        lambda: driver.agent_prompt("--current", "x", timeout_s=1.0),
        lambda: driver.pane_send_keys("--current", "enter", timeout_s=1.0),
        lambda: driver.pane_read("--current", timeout_s=1.0),
        lambda: driver.pane_close("--current", timeout_s=1.0),
        lambda: driver.agent_wait("--current", until=["idle"], timeout_ms=1000),
        lambda: driver.pane_split(
            "--current", direction="right", cwd=Path("/x"), env={},
            focus=False, timeout_s=1.0,
        ),
        lambda: driver.agent_start(
            "ok-name", kind="claude", pane_id="--current", timeout_ms=1000
        ),
    ):
        with pytest.raises(ValueError):
            call()
    assert run.calls == []  # nothing ever reached the CLI


@pytest.mark.parametrize(
    "name,valid",
    [
        ("a", True),
        ("sess-a1", True),
        ("a" * 32, True),
        ("", False),
        ("A", False),
        ("1abc", False),
        ("a" * 33, False),
        ("has space", False),
        ("-current", False),
    ],
)
def test_agent_name_validation(name: str, valid: bool, fake: Any) -> None:
    fake(stdout=(FIXTURES / "agent_start_no_session.json").read_text())
    if valid:
        driver.agent_start(name, kind="claude", pane_id="w3:p2", timeout_ms=1000)
    else:
        with pytest.raises(ValueError):
            driver.agent_start(
                name, kind="claude", pane_id="w3:p2", timeout_ms=1000
            )


def test_pane_read_returns_plain_text(fake: Any) -> None:
    run = fake(stdout="shell prompt $\n")
    text = driver.pane_read("w3:p2", timeout_s=5.0)
    assert text == "shell prompt $\n"
    assert run.calls == [[
        "herdr", "pane", "read", "w3:p2",
        "--source", "visible", "--format", "text",
    ]]


def test_notification_show_argv(fake: Any) -> None:
    run = fake(stdout="")
    driver.notification_show(
        "escalation", body="session 1 blocked", sound="request", timeout_s=5.0
    )
    assert run.calls == [[
        "herdr", "notification", "show", "escalation",
        "--body", "session 1 blocked",
        "--sound", "request",
    ]]
