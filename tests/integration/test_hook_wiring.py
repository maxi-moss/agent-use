"""Integration: the REAL hook subprocess against a REAL SessionBroker socket
server, plus the full in-process loop walking criteria 3-8.

Harness pieces mirror tests/unit/test_session_broker.py (no conftest by
project convention): ScriptedRun monkeypatches driver.subprocess.run, FakeLLM
feeds results through a queue, StubMaster records envelopes behind a real
serve_unix.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
import tempfile

from broker.herdr import driver
from broker.llm import ToolCall
from broker.protocol import client
from broker.protocol.constants import (
    T_APPROVE_PROMPT,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_GET_DECISION_LOG,
    T_HOOK_EVENT,
    T_PROMPT_PROPOSAL,
)
from broker.protocol.schemas import Envelope, Response
from broker.protocol.server import serve_unix
from broker.session.broker import SessionBroker
from broker.config import SessionBrokerConfig

FIXTURES = Path(__file__).parent.parent / "fixtures"
HERDR_FIXTURES = FIXTURES / "herdr"
TRANSCRIPT_FIXTURE = (
    FIXTURES / "transcripts" / "d4032982-9753-4cce-ac1a-589ee8fe7e19.jsonl"
)

ANSWER_RESULT = ToolCall(
    name="answer", input={"reasoning": "grounded", "answer": "use oauth"}
)
ESCALATE_RESULT = ToolCall(
    name="escalate",
    input={
        "reasoning": "plan looks wrong",
        "situation": "the plan contradicts the code",
        "what_was_asked": "how to proceed",
        "what_is_at_stake": "architecture drift",
        "alternatives": [{"option": "a", "pros": "p", "cons": "c"}],
        "recommendation": "stop and ask",
        "uncertainty": "whether the plan is stale",
        "what_would_change_my_mind": "a fresher plan",
    },
)
COMPLETE_RESULT = ToolCall(
    name="complete",
    input={"reasoning": "all done", "summary": "task finished cleanly"},
)


class ScriptedRun:
    """Dispatching stand-in for subprocess.run inside the herdr driver."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        argv: list[str],
        capture_output: bool = False,
        text: bool = False,
        timeout: float = 0.0,
        cwd: Path | str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        key = tuple(argv[1:3])
        if key == ("pane", "split"):
            out = (HERDR_FIXTURES / "pane_split.json").read_text()
        elif key == ("agent", "start"):
            out = (HERDR_FIXTURES / "agent_start_nested.json").read_text()
        elif key == ("agent", "get"):
            out = (HERDR_FIXTURES / "agent_get.json").read_text()
        else:
            out = "{}"
        return subprocess.CompletedProcess(list(argv), 0, out, "")

    def drive_calls(self) -> list[list[str]]:
        return [
            c
            for c in self.calls
            if c[1:3] == ["agent", "prompt"] or c[1:3] == ["pane", "send-keys"]
        ]


class FakeLLM:
    def __init__(self) -> None:
        self.results: asyncio.Queue[ToolCall] = asyncio.Queue()
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> ToolCall:
        self.calls.append(kwargs)
        return await self.results.get()


class StubMaster:
    def __init__(self) -> None:
        self.received: list[Envelope] = []

    async def __call__(self, env: Envelope) -> Response | None:
        self.received.append(env)
        return Response(id=env.id, ok=True)

    def of_type(self, msg_type: str) -> list[Envelope]:
        return [e for e in self.received if e.type == msg_type]

    async def wait_for(
        self, msg_type: str, *, count: int = 1, timeout: float = 5.0
    ) -> Envelope:
        async with asyncio.timeout(timeout):
            while len(self.of_type(msg_type)) < count:
                await asyncio.sleep(0.01)
        return self.of_type(msg_type)[count - 1]


@dataclass
class Harness:
    broker: SessionBroker
    cfg: SessionBrokerConfig
    master: StubMaster
    llm: FakeLLM
    run: ScriptedRun
    sock: Path
    transcript: Path
    run_task: "asyncio.Task[None]" = field(repr=False, kw_only=True)


def hook_env(name: str, raw_extra: dict[str, Any]) -> Envelope:
    raw: dict[str, Any] = {"hook_event_name": name, "session_id": "cc-1"}
    raw.update(raw_extra)
    return Envelope(
        id=uuid.uuid4().hex,
        type=T_HOOK_EVENT,
        session_id="cc-1",
        payload={"hook_event_name": name, "raw": raw},
    )


async def wait_state(
    broker: SessionBroker, state: str, timeout: float = 5.0
) -> None:
    async with asyncio.timeout(timeout):
        while broker.state != state:
            await asyncio.sleep(0.01)


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
        monkeypatch.setenv("BROKER_HOME", d)
        yield Path(d)


def make_cfg(home: Path, *, budget_max: int = 8) -> SessionBrokerConfig:
    cwd = home / "work"
    cwd.mkdir(exist_ok=True)
    return SessionBrokerConfig(
        name="s1",
        socket_path=str(home / "s" / "s1.sock"),
        master_socket_path=str(home / "m.sock"),
        broker_home=home,
        cwd=str(cwd),
        anchor_pane="w3:p1",
        intent="the raw intent",
        model_id="test-model",
        max_tokens=1024,
        watchdog_seconds=300.0,
        budget_max=budget_max,
    )


async def start_harness(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    budget_max: int = 8,
) -> tuple[Harness, "asyncio.Server"]:
    transcript = home / "t.jsonl"
    shutil.copy(TRANSCRIPT_FIXTURE, transcript)
    run = ScriptedRun()
    monkeypatch.setattr(driver.subprocess, "run", run)
    llm = FakeLLM()
    master = StubMaster()
    master_server = await serve_unix(home / "m.sock", master)
    cfg = make_cfg(home, budget_max=budget_max)
    broker = SessionBroker(cfg, llm_call=llm)
    run_task = asyncio.create_task(broker.run())
    sock = Path(cfg.socket_path)
    async with asyncio.timeout(5.0):
        while not sock.exists():
            await asyncio.sleep(0.01)
    harness = Harness(
        broker=broker,
        cfg=cfg,
        master=master,
        llm=llm,
        run=run,
        sock=sock,
        transcript=transcript,
        run_task=run_task,
    )
    return harness, master_server


@pytest.fixture
async def harness(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Harness]:
    h, master_server = await start_harness(home, monkeypatch)
    yield h
    await _teardown(h, master_server)


@pytest.fixture
async def tight_budget_harness(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Harness]:
    h, master_server = await start_harness(home, monkeypatch, budget_max=2)
    yield h
    await _teardown(h, master_server)


async def _teardown(h: Harness, master_server: "asyncio.Server") -> None:
    h.run_task.cancel()
    try:
        await h.run_task
    except (asyncio.CancelledError, Exception):
        pass
    master_server.close()
    await master_server.wait_closed()


async def launch(h: Harness) -> None:
    """Walk the launch sequence to the driving state."""
    await client.notify(
        h.sock,
        hook_env(
            "SessionStart",
            {"session_id": "cc-1", "transcript_path": str(h.transcript)},
        ),
    )
    await h.llm.results.put(
        ToolCall(
            name="propose_prompt",
            input={"reasoning": "grounded", "prompt": "GROUNDED PROMPT"},
        )
    )
    proposal = await h.master.wait_for(T_PROMPT_PROPOSAL)
    resp = await client.request(
        h.sock,
        Envelope(
            id=uuid.uuid4().hex,
            type=T_APPROVE_PROMPT,
            session_id="s1",
            payload={
                "proposal_id": proposal.payload["proposal_id"],
                "prompt": "APPROVED PROMPT",
            },
        ),
        timeout_s=5.0,
    )
    assert resp.ok is True
    await wait_state(h.broker, "driving")


# ── hook payloads: field shapes ───────────────────────────────────────────


def stop_payload(transcript: Path) -> dict[str, Any]:
    return {
        "session_id": "cc-1",
        "transcript_path": str(transcript),
        "cwd": "/private/tmp/work",
        "prompt_id": "p-1",
        "permission_mode": "default",
        "effort": {"level": "high"},
        "hook_event_name": "Stop",
        "last_assistant_message": "Which auth provider should I use?",
        "stop_hook_active": False,
        "background_tasks": [],
        "session_crons": [],
    }


def pretooluse_payload(transcript: Path) -> dict[str, Any]:
    return {
        "session_id": "cc-1",
        "transcript_path": str(transcript),
        "cwd": "/private/tmp/work",
        "prompt_id": "p-1",
        "permission_mode": "default",
        "effort": {"level": "high"},
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "ls", "description": "list files"},
        "tool_use_id": "toolu_int_1",
    }


async def run_hook(
    payload: dict[str, Any], sock: Path
) -> tuple[bytes, bytes, float]:
    """Run the REAL hook subprocess; returns (stdout, stderr, elapsed_s)."""
    env = dict(os.environ)
    env["BROKER_SOCKET"] = str(sock)
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "broker.hook",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate(json.dumps(payload).encode())
    elapsed = time.monotonic() - start
    assert proc.returncode == 0  # exit 0 ALWAYS
    return stdout, stderr, elapsed


async def test_hook_stop_reaches_broker_and_triggers_classify(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await launch(harness)
    resets: list[None] = []
    real_reset = harness.broker.watchdog.reset

    def spy_reset() -> None:
        resets.append(None)
        real_reset()

    monkeypatch.setattr(harness.broker.watchdog, "reset", spy_reset)
    harness.llm.calls.clear()
    await run_hook(stop_payload(harness.transcript), harness.sock)
    await harness.llm.results.put(ANSWER_RESULT)
    await harness.master.wait_for(T_BUDGET_UPDATE)
    assert resets  # watchdog.reset fired on the hook event
    # classify ran with last_assistant_message extracted from payload.raw.
    triage_call = harness.llm.calls[-1]
    content = cast(
        list[dict[str, Any]], triage_call["messages"][0]["content"]
    )
    assert "Which auth provider should I use?" in content[-1]["text"]


async def test_hook_pretooluse_stdout_empty_and_fast(
    harness: Harness,
) -> None:
    # BROKER_HOOK_TIMEOUT deliberately unset: the reply is immediate.
    stdout, stderr, elapsed = await run_hook(
        pretooluse_payload(harness.transcript), harness.sock
    )
    assert stdout == b""  # escalated → nothing on stdout → native flow
    assert stderr == b""
    assert elapsed < 1.0  # no measurable turn delay


async def test_full_loop_criteria_3_to_8(
    tight_budget_harness: Harness,
) -> None:
    """One scripted scenario: question→answer, question→escalate→dispatch,
    budget exhaustion handover, completion (budget_max=2)."""
    h = tight_budget_harness
    await launch(h)

    # ── prose question → autonomous answer, logged ────────────────────────────
    h.run.calls.clear()
    await client.notify(
        h.sock,
        hook_env("Stop", {"last_assistant_message": "Which auth provider?"}),
    )
    await h.llm.results.put(ANSWER_RESULT)
    budget = await h.master.wait_for(T_BUDGET_UPDATE)
    assert budget.payload == {"count": 1}
    assert h.run.drive_calls() == [
        ["herdr", "agent", "prompt", "s1", "use oauth"],
        ["herdr", "pane", "send-keys", "w3:p2", "enter"],
    ]
    log_resp = await client.request(
        h.sock,
        Envelope(id=uuid.uuid4().hex, type=T_GET_DECISION_LOG, payload={}),
        timeout_s=5.0,
    )
    assert "grounded" in log_resp.payload["text"]  # rationale recorded

    # ── question → full structured escalation ─────────────────────────────────
    await client.notify(
        h.sock, hook_env("Stop", {"last_assistant_message": "proceed how?"})
    )
    await h.llm.results.put(ESCALATE_RESULT)
    escalation = await h.master.wait_for(T_ESCALATION)
    for field_name in (
        "escalation_id",
        "session_id",
        "task_context",
        "situation",
        "what_was_asked",
        "what_is_at_stake",
        "alternatives",
        "recommendation",
        "uncertainty",
        "what_would_change_my_mind",
    ):
        assert field_name in escalation.payload
    await wait_state(h.broker, "escalated")

    # ── developer decision reaches the session and is acted on ────────────────
    h.run.calls.clear()
    resp = await client.request(
        h.sock,
        Envelope(
            id=uuid.uuid4().hex,
            type=T_DISPATCH_DECISION,
            session_id="s1",
            payload={
                "escalation_id": escalation.payload["escalation_id"],
                "response": "use option a, keep the old table",
            },
        ),
        timeout_s=5.0,
    )
    assert resp.ok is True
    await wait_state(h.broker, "driving")
    assert h.run.drive_calls() == [
        ["herdr", "agent", "prompt", "s1", "use option a, keep the old table"],
        ["herdr", "pane", "send-keys", "w3:p2", "enter"],
    ]
    # Budget was reset by the dispatched decision.
    budget = await h.master.wait_for(T_BUDGET_UPDATE, count=2)
    assert budget.payload == {"count": 0}

    # ── budget exhaustion converts answer into handover ───────────────────────
    for i in range(2):  # budget_max=2 autonomous answers
        await client.notify(
            h.sock,
            hook_env("Stop", {"last_assistant_message": f"question {i}?"}),
        )
        await h.llm.results.put(ANSWER_RESULT)
        await h.master.wait_for(T_BUDGET_UPDATE, count=3 + i)
    await client.notify(
        h.sock,
        hook_env("Stop", {"last_assistant_message": "one more question?"}),
    )
    h.run.calls.clear()
    await h.llm.results.put(ANSWER_RESULT)  # would answer, but budget is spent
    handover = await h.master.wait_for(T_ESCALATION, count=2)
    # The would-be answer is carried verbatim as the recommendation.
    assert handover.payload["recommendation"] == "use oauth"
    await wait_state(h.broker, "escalated")
    assert h.run.drive_calls() == []  # halted: nothing written to the pane

    # resolve the handover so the scenario can finish
    resp = await client.request(
        h.sock,
        Envelope(
            id=uuid.uuid4().hex,
            type=T_DISPATCH_DECISION,
            session_id="s1",
            payload={
                "escalation_id": handover.payload["escalation_id"],
                "response": "go with oauth",
            },
        ),
        timeout_s=5.0,
    )
    assert resp.ok is True
    await wait_state(h.broker, "driving")

    # ── completion detected; broker stops driving ─────────────────────────────
    await client.notify(
        h.sock,
        hook_env("Stop", {"last_assistant_message": "All done."}),
    )
    await h.llm.results.put(COMPLETE_RESULT)
    completion = await h.master.wait_for(T_COMPLETION)
    assert completion.payload == {"summary": "task finished cleanly"}
    await wait_state(h.broker, "completed")
    # The socket still answers status while completed.
    status = await client.request(
        h.sock,
        Envelope(id=uuid.uuid4().hex, type="status", payload={}),
        timeout_s=5.0,
    )
    assert status.payload["state"] == "completed"
