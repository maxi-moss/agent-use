"""SessionBroker in-process harness: real serve_unix sockets under /private/tmp,
fake llm_call, scripted driver.subprocess.run, stub master recording envelopes."""

import asyncio
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from broker.herdr import driver
from broker.llm import ToolCall
from broker.protocol import client
from broker.protocol.constants import (
    T_APPROVE_PROMPT,
    T_BUDGET_UPDATE,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_FATAL_ERROR,
    T_HOOK_EVENT,
    T_PERMISSION_REQUEST,
    T_PROMPT_PROPOSAL,
    T_RETRACT,
    T_STATUS,
)
from broker.protocol.schemas import Envelope, Response
from broker.protocol.server import serve_unix
from broker.session.broker import SessionBroker
from broker.session.config import SessionBrokerConfig

FIXTURES = Path(__file__).parent.parent / "fixtures"
HERDR_FIXTURES = FIXTURES / "herdr"
# Small real transcript with one answered AskUserQuestion (see SOURCES.md).
TRANSCRIPT_FIXTURE = (
    FIXTURES / "transcripts" / "d4032982-9753-4cce-ac1a-589ee8fe7e19.jsonl"
)
ANSWERED_ASK_ID = "toolu_01JkpyNV1bx66fUu8xkoanws"

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
        """Calls that write into the pane (the two-step submits)."""
        return [
            c
            for c in self.calls
            if c[1:3] == ["agent", "prompt"] or c[1:3] == ["pane", "send-keys"]
        ]


class FakeLLM:
    """Results are fed through a queue: an empty queue models a pending call."""

    def __init__(self) -> None:
        self.results: asyncio.Queue[ToolCall] = asyncio.Queue()
        self.calls: list[dict[str, Any]] = []
        self.never_resolve = False

    async def __call__(self, **kwargs: Any) -> ToolCall:
        self.calls.append(kwargs)
        if self.never_resolve:
            await asyncio.Event().wait()
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


def permission_env(
    tool_name: str, tool_use_id: str, tool_input: dict[str, Any]
) -> Envelope:
    return Envelope(
        id=uuid.uuid4().hex,
        type=T_PERMISSION_REQUEST,
        session_id="cc-1",
        payload={
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_use_id": tool_use_id,
            "cwd": "/private/tmp/x",
            "transcript_path": "/private/tmp/x/t.jsonl",
        },
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


@pytest.fixture
async def harness(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Harness]:
    transcript = home / "t.jsonl"
    shutil.copy(TRANSCRIPT_FIXTURE, transcript)
    cwd = home / "work"
    cwd.mkdir()
    run = ScriptedRun()
    monkeypatch.setattr(driver.subprocess, "run", run)
    llm = FakeLLM()
    master = StubMaster()
    master_sock = home / "m.sock"
    master_server = await serve_unix(master_sock, master)
    cfg = SessionBrokerConfig(
        name="s1",
        socket_path=str(home / "s" / "s1.sock"),
        master_socket_path=str(master_sock),
        cwd=str(cwd),
        anchor_pane="w3:p1",
        intent="the raw intent",
        model_id="test-model",
        max_tokens=1024,
        watchdog_seconds=300.0,
        budget_max=8,
    )
    broker = SessionBroker(cfg, llm_call=llm)
    run_task = asyncio.create_task(broker.run())
    # Wait for the session socket to be bound (bind happens FIRST in run()).
    sock = Path(cfg.socket_path)
    async with asyncio.timeout(5.0):
        while not sock.exists():
            await asyncio.sleep(0.01)
    yield Harness(
        broker=broker,
        cfg=cfg,
        master=master,
        llm=llm,
        run=run,
        sock=sock,
        transcript=transcript,
        run_task=run_task,
    )
    run_task.cancel()
    try:
        await run_task
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
            input={"reasoning": "grounded in cwd", "prompt": "GROUNDED PROMPT"},
        )
    )
    proposal = await h.master.wait_for(T_PROMPT_PROPOSAL)
    assert h.broker.state == "awaiting_approval"
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


async def test_permission_request_immediate_escalated_reply(
    harness: Harness,
) -> None:
    harness.llm.never_resolve = True  # any LLM work would hang forever
    start = time.monotonic()
    resp = await client.request(
        harness.sock,
        permission_env("Bash", "toolu_x1", {"command": "ls"}),
        timeout_s=5.0,
    )
    elapsed = time.monotonic() - start
    assert resp.ok is True
    assert resp.payload["decision"] == "escalated"
    assert elapsed < 0.5  # hot path


async def test_stop_triggers_triage_and_answer_submits_two_step(
    harness: Harness,
) -> None:
    await launch(harness)
    harness.run.calls.clear()
    await client.notify(
        harness.sock,
        hook_env("Stop", {"last_assistant_message": "Which auth provider?"}),
    )
    await harness.llm.results.put(ANSWER_RESULT)
    budget = await harness.master.wait_for(T_BUDGET_UPDATE)
    assert budget.payload == {"count": 1}
    assert harness.run.drive_calls() == [
        ["herdr", "agent", "prompt", "s1", "use oauth"],
        ["herdr", "pane", "send-keys", "w3:p2", "enter"],
    ]
    # Classification input is last_assistant_message, in the LAST content block.
    triage_call = harness.llm.calls[-1]
    content = cast(list[dict[str, Any]], triage_call["messages"][0]["content"])
    assert "Which auth provider?" in content[-1]["text"]
    # The approved prompt, not the raw intent, is the authoritative intent.
    assert "APPROVED PROMPT" in content[0]["text"]


async def test_escalation_sent_then_broker_is_quiescent(
    harness: Harness,
) -> None:
    await launch(harness)
    await client.notify(
        harness.sock, hook_env("Stop", {"last_assistant_message": "proceed how?"})
    )
    await harness.llm.results.put(ESCALATE_RESULT)
    escalation = await harness.master.wait_for(T_ESCALATION)
    assert escalation.payload["situation"] == "the plan contradicts the code"
    assert escalation.payload["task_context"] == "APPROVED PROMPT"
    await wait_state(harness.broker, "escalated")
    harness.run.calls.clear()
    # Further turn boundaries must not write to the pane (quiescence).
    await client.notify(
        harness.sock, hook_env("Stop", {"last_assistant_message": "still here"})
    )
    await asyncio.sleep(0.2)
    assert harness.run.drive_calls() == []
    assert harness.broker.state == "escalated"


async def test_ask_user_question_escalates_and_retracts_on_answer(
    harness: Harness,
) -> None:
    await launch(harness)
    tool_input: dict[str, Any] = {
        "questions": [
            {
                "question": "Pick a color",
                "header": "Color",
                "options": [
                    {"label": "Blue (Recommended)", "description": "calm"},
                    {"label": "Red", "description": "loud"},
                ],
                "multiSelect": False,
            }
        ]
    }
    resp = await client.request(
        harness.sock,
        permission_env("AskUserQuestion", ANSWERED_ASK_ID, tool_input),
        timeout_s=5.0,
    )
    assert resp.payload["decision"] == "escalated"  # native menu renders
    escalation = await harness.master.wait_for(T_ESCALATION)
    assert "manual input required in pane w3:p2" in escalation.payload["situation"]
    assert "Pick a color" in escalation.payload["what_was_asked"]
    assert escalation.payload["recommendation"] == "Blue (Recommended)"
    labels = [a["option"] for a in escalation.payload["alternatives"]]
    assert labels == ["Blue (Recommended)", "Red"]
    await wait_state(harness.broker, "escalated")
    # A duplicate PreToolUse for the same tool_use_id must not re-escalate.
    await client.request(
        harness.sock,
        permission_env("AskUserQuestion", ANSWERED_ASK_ID, tool_input),
        timeout_s=5.0,
    )
    await asyncio.sleep(0.1)
    assert len(harness.master.of_type(T_ESCALATION)) == 1
    # The transcript contains the paired answer -> next hook event retracts.
    await client.notify(harness.sock, hook_env("PostToolUse", {}))
    retract = await harness.master.wait_for(T_RETRACT)
    assert retract.payload["escalation_id"] == escalation.payload["escalation_id"]
    assert retract.payload["reason"] == "resolved in pane"
    await wait_state(harness.broker, "driving")


async def test_dispatch_decision_submits_and_resets_budget(
    harness: Harness,
) -> None:
    await launch(harness)
    await client.notify(
        harness.sock, hook_env("Stop", {"last_assistant_message": "proceed how?"})
    )
    await harness.llm.results.put(ESCALATE_RESULT)
    escalation = await harness.master.wait_for(T_ESCALATION)
    await wait_state(harness.broker, "escalated")
    harness.run.calls.clear()
    resp = await client.request(
        harness.sock,
        Envelope(
            id=uuid.uuid4().hex,
            type=T_DISPATCH_DECISION,
            session_id="s1",
            payload={
                "escalation_id": escalation.payload["escalation_id"],
                "response": "go with option a",
            },
        ),
        timeout_s=5.0,
    )
    assert resp.ok is True
    budget = await harness.master.wait_for(T_BUDGET_UPDATE)
    assert budget.payload == {"count": 0}
    assert harness.run.drive_calls() == [
        ["herdr", "agent", "prompt", "s1", "go with option a"],
        ["herdr", "pane", "send-keys", "w3:p2", "enter"],
    ]
    assert harness.broker.budget_count == 0
    await wait_state(harness.broker, "driving")


async def test_stale_dispatch_decision_is_ignored(harness: Harness) -> None:
    await launch(harness)
    harness.run.calls.clear()
    await client.request(
        harness.sock,
        Envelope(
            id=uuid.uuid4().hex,
            type=T_DISPATCH_DECISION,
            session_id="s1",
            payload={"escalation_id": "stale-id", "response": "too late"},
        ),
        timeout_s=5.0,
    )
    await asyncio.sleep(0.1)
    assert harness.run.drive_calls() == []


async def test_budget_exhaustion_converts_answer_to_escalation(
    harness: Harness,
) -> None:
    await launch(harness)
    harness.broker.budget_count = harness.cfg.budget_max
    harness.run.calls.clear()
    await client.notify(
        harness.sock,
        hook_env("Stop", {"last_assistant_message": "Which auth provider?"}),
    )
    await harness.llm.results.put(ANSWER_RESULT)
    escalation = await harness.master.wait_for(T_ESCALATION)
    assert "budget exhausted" in escalation.payload["situation"].lower()
    # The prepared answer becomes the recommendation, verbatim (handover).
    assert escalation.payload["recommendation"] == "use oauth"
    assert harness.run.drive_calls() == []  # the answer was NOT submitted
    await wait_state(harness.broker, "escalated")


async def test_stop_failure_sends_fatal_error(harness: Harness) -> None:
    await launch(harness)
    await client.notify(
        harness.sock,
        hook_env(
            "StopFailure",
            {"matcher": "rate_limit", "message": "429 from upstream"},
        ),
    )
    fatal = await harness.master.wait_for(T_FATAL_ERROR)
    assert fatal.payload["error_class"] == "rate_limit"
    assert "429" in fatal.payload["detail"]
    await wait_state(harness.broker, "error")


async def test_transcript_parse_error_sends_fatal_error(
    harness: Harness,
) -> None:
    await launch(harness)
    harness.transcript.write_text('{"broken json\n')
    await client.notify(
        harness.sock, hook_env("Stop", {"last_assistant_message": "hi"})
    )
    fatal = await harness.master.wait_for(T_FATAL_ERROR)
    assert fatal.payload["error_class"] == "TranscriptParseError"
    await wait_state(harness.broker, "error")


async def test_status_answers_with_bound_identifiers(harness: Harness) -> None:
    await launch(harness)
    resp = await client.request(
        harness.sock,
        Envelope(id=uuid.uuid4().hex, type=T_STATUS, session_id="s1"),
        timeout_s=5.0,
    )
    assert resp.payload["state"] == "driving"
    assert resp.payload["pane_id"] == "w3:p2"
    assert resp.payload["claude_session_id"] == "cc-1"
    assert resp.payload["transcript_path"] == str(harness.transcript)
