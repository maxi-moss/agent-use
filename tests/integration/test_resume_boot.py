"""Integration: a REAL `python -m broker.session` boots from a resume config
against a stub master, binds its session socket and answers the status probe
as driving — without grounding, so no prompt proposal ever reaches the master.

Harness pieces mirror tests/integration/test_hook_wiring.py (no conftest by
project convention): a StubMaster recording envelopes behind a real
serve_unix, one real subprocess with a bounded poll and owned teardown.
"""

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest

from broker.config import (
    AdoptedSession,
    ClassifierConfig,
    EmbeddingConfig,
    ResumedTask,
    SessionBrokerConfig,
    SessionModelConfig,
)
from broker.protocol import client
from broker.protocol.constants import T_LIVE_STATUS, T_PROMPT_PROPOSAL, T_STATUS
from broker.protocol.schemas import Envelope, Response

from broker.protocol.server import serve_unix

pytestmark = pytest.mark.integration

RESUMED_PROMPT = "the approved first task"


class StubMaster:
    def __init__(self) -> None:
        self.received: list[Envelope] = []

    async def __call__(self, env: Envelope) -> Response | None:
        self.received.append(env)
        return Response(id=env.id, ok=True)

    def of_type(self, msg_type: str) -> list[Envelope]:
        return [e for e in self.received if e.type == msg_type]


def _resume_config(home: Path) -> SessionBrokerConfig:
    cwd = home / "work"
    cwd.mkdir()
    transcript = home / "t.jsonl"
    transcript.write_text("", encoding="utf-8")
    return SessionBrokerConfig(
        name="s1",
        socket_path=str(home / "s" / "s1.sock"),
        master_socket_path=str(home / "m.sock"),
        broker_home=home,
        cwd=str(cwd),
        anchor_pane="w3:p1",
        intent="the raw intent",
        budget_count=6,
        session_model=SessionModelConfig(model_id="test-model", max_tokens=1024),
        classifier=ClassifierConfig(model_id="test-classifier"),
        embedding=EmbeddingConfig(),
        watchdog_seconds=3600.0,  # never fires within the test
        budget_max=8,
        claude_settings_path=str(home / "claude-settings.json"),
        adopt=AdoptedSession(
            pane_id="w9:p9",
            claude_session_id="cc-1",
            transcript_path=str(transcript),
        ),
        resume=ResumedTask(approved_prompt=RESUMED_PROMPT, completed=False),
    )


async def _poll_status(sock: Path) -> dict[str, Any]:
    """Poll the session socket until the broker reports driving, bounded."""
    last: dict[str, Any] | None = None
    for _ in range(100):
        try:
            resp = await client.request(
                sock,
                Envelope(id=uuid.uuid4().hex, type=T_STATUS, session_id="s1"),
                timeout_s=2.0,
            )
        except (TimeoutError, OSError, ConnectionError):
            await asyncio.sleep(0.1)
            continue
        last = resp.payload
        if last.get("state") == "driving":
            return last
        await asyncio.sleep(0.1)
    raise AssertionError(f"broker never reported driving; last status: {last}")


async def _wait_live_driving(master: StubMaster) -> dict[str, Any]:
    """Wait for a driving live-status push to reach the stub master, bounded."""
    for _ in range(100):
        for env in master.of_type(T_LIVE_STATUS):
            if env.payload.get("state") == "driving":
                return env.payload
        await asyncio.sleep(0.1)
    raise AssertionError("no driving live-status push reached the master")


async def test_resumed_broker_binds_and_answers_driving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        home = Path(td)
        monkeypatch.setenv("BROKER_HOME", td)
        master = StubMaster()
        master_server = await serve_unix(home / "m.sock", master)
        cfg = _resume_config(home)
        env = dict(os.environ)
        # The broker constructs its client unconditionally; resume never
        # calls it, so a placeholder satisfies construction.
        env["ANTHROPIC_API_KEY"] = "placeholder-key"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "broker.session",
            "--config-json",
            cfg.model_dump_json(),
            env=env,
        )
        try:
            await _poll_status(Path(cfg.socket_path))
            # The resumed binding reaches the master exactly as the config
            # carried it.
            live = await _wait_live_driving(master)
            assert live["pane_id"] == "w9:p9"
            assert live["claude_session_id"] == "cc-1"
            # Resume never grounds: no proposal reached the master.
            assert master.of_type(T_PROMPT_PROPOSAL) == []
        finally:
            proc.terminate()
            await proc.wait()
            master_server.close()
            await master_server.wait_closed()
