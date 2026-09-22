"""SessionBroker in-process harness: real serve_unix sockets under /private/tmp,
fake llm_call, scripted driver.subprocess.run, stub master recording envelopes."""

import asyncio
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from broker.herdr import driver
from broker.index.retrieval import RetrievalError
from broker.index.schemas import ContextSymbol, GroundingContext, SymbolKind
from broker.llm import LLMCallError, ToolCall
from broker.permission import PermissionModule
from broker.permission.llm import ToolCall as PermissionToolCall
from broker.protocol import client
from broker.protocol.constants import (
    NACK_WRONG_STATE,
    T_APPROVE_PROMPT,
    T_CLARIFY_ESCALATION,
    T_ASK_QUESTION,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DECISION_DELIVERED,
    T_DECISION_UNDELIVERED,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_ESCALATION_RETRACT,
    T_FATAL_ERROR,
    T_GET_DECISION_LOG,
    T_GET_PERMISSION_LOG,
    T_HOOK_EVENT,
    T_LIVE_STATUS,
    T_PERMISSION_REQUEST,
    T_PROMPT_PROPOSAL,
    T_REACTIVATE,
    T_SESSION_ENDED,
    T_SHUTDOWN,
    T_STATUS,
)
from broker.protocol.schemas import ClarifyEscalationReplyPayload, Envelope, Response
from broker.protocol.server import serve_unix
from broker.session.broker import (
    PHRASE_GROUNDING,
    PHRASE_PERMISSION,
    PHRASE_TRIAGE,
    SessionBroker,
)
from broker.config import (
    AdoptedSession,
    ClassifierConfig,
    EmbeddingConfig,
    ResumedTask,
    SessionBrokerConfig,
)
from broker.paths import BrokerPaths

FIXTURES = Path(__file__).parent.parent / "fixtures"
HERDR_FIXTURES = FIXTURES / "herdr"
# Small real transcript with one REJECTED AskUserQuestion (see SOURCES.md).
# Resolution matching is on answer-id existence, so the rejected answer still
# serves as "the pane resolved this id" in the retract tests.
TRANSCRIPT_FIXTURE = (
    FIXTURES / "transcripts" / "d4032982-9753-4cce-ac1a-589ee8fe7e19.jsonl"
)
PENDING_ASK_ID = "toolu_01JkpyNV1bx66fUu8xkoanws"
# Identifiers a reassigned broker adopts — deliberately unlike the ones the
# pane_split/agent_start fixtures return, so an adopted broker that fell back
# to starting its own session would show it.
ADOPTED_PANE = "w9:p9"
ADOPTED_SESSION = "cc-adopted"
RESUMED_PROMPT = "the approved first task"

ANSWER_RESULT = ToolCall(
    name="answer",
    input={
        "reasoning": "grounded",
        "answer": "use oauth",
        "task_activity": "wiring up oauth",
        "task_summary": "Chose oauth",
    },
)
COLOR_TOOL_INPUT: dict[str, Any] = {
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
ANSWER_QUESTIONS_RESULT = ToolCall(
    name="answer_questions",
    input={
        "reasoning": "blue matches the intent",
        "answers": [
            {
                "question": "Pick a color",
                "selected": ["Blue (Recommended)"],
                "free_text": "",
            }
        ],
    },
)
INVALID_ANSWER_QUESTIONS_RESULT = ToolCall(
    name="answer_questions",
    input={
        "reasoning": "confused",
        "answers": [
            {"question": "Pick a color", "selected": ["Green"], "free_text": ""}
        ],
    },
)
ESCALATE_RESULT = ToolCall(
    name="escalate",
    input={
        "reasoning": "plan looks wrong",
        "task_summary": "Questioned a stale plan",
        "situation": "the plan contradicts the code",
        "what_was_asked": "how to proceed",
        "what_is_at_stake": "architecture drift",
        "alternatives": [{"option": "a", "pros": "p", "cons": "c"}],
        "recommendation": "stop and ask",
        "uncertainty": "whether the plan is stale",
        "what_would_change_my_mind": "a fresher plan",
    },
)
CLARIFY_RESULT = ToolCall(
    name="answer_clarification",
    input={"reasoning": "the transcript says so", "answer": "it tried A first"},
)


class ScriptedRun:
    """Dispatching stand-in for subprocess.run inside the herdr driver."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail_prompt: Exception | None = None

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
        if key == ("agent", "prompt") and self.fail_prompt is not None:
            raise self.fail_prompt
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
        """Calls that write into the pane: the submitting `agent prompt`."""
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
        self.raise_error: Exception | None = None

    async def __call__(self, **kwargs: Any) -> ToolCall:
        self.calls.append(kwargs)
        if self.raise_error is not None:
            raise self.raise_error
        if self.never_resolve:
            await asyncio.Event().wait()
        return await self.results.get()


RETRIEVED_CONTEXT = GroundingContext(
    symbols=[
        ContextSymbol(
            qualified_name="src/x.py::do_it", path="src/x.py", kind=SymbolKind.FUNCTION,
            start_line=1, end_line=3, signature="def do_it() -> None:", fields=[], methods=[],
            score=0.7, rank=0.7,
        )
    ],
    edges=[],
    imports={},
)


class FakeRetriever:
    """Injected retrieval seam: returns a fixed context, or raises if scripted."""

    def __init__(self, context: GroundingContext = RETRIEVED_CONTEXT) -> None:
        self.context = context
        self.calls: list[tuple[str, Path]] = []
        self.raise_error: Exception | None = None

    async def __call__(self, intent: str, cwd: Path) -> GroundingContext:
        self.calls.append((intent, cwd))
        if self.raise_error is not None:
            raise self.raise_error
        return self.context


class FakePermissionLLM:
    """Scripted classifier: an empty queue models a call still in flight."""

    def __init__(self) -> None:
        self.results: asyncio.Queue[PermissionToolCall] = asyncio.Queue()
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> PermissionToolCall:
        self.calls.append(kwargs)
        return await self.results.get()

    async def script(self, name: str, reasoning: str) -> None:
        """Queue one classifier outcome."""
        await self.results.put(
            PermissionToolCall(name=name, input={"reasoning": reasoning})
        )


class SpyPermission(PermissionModule):
    """Records every signal the broker sends the module, and acts on none."""

    def __init__(self, log_path: Path, master_socket_path: str) -> None:
        super().__init__(
            ClassifierConfig(),
            session_name="s1",
            master_socket_path=master_socket_path,
            log_path=log_path,
            intent="the raw intent",
        )
        self.notes: list[str] = []
        self.intents: list[str] = []

    def note_tool_completed(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> None:
        self.notes.append(f"tool_completed:{tool_name}:{tool_input}")

    def note_developer_input(self) -> None:
        self.notes.append("developer_input")

    def note_session_ended(self) -> None:
        self.notes.append("session_ended")

    def set_intent(self, intent: str) -> None:
        self.intents.append(intent)


class StubMaster:
    def __init__(self) -> None:
        self.received: list[Envelope] = []
        # Message types whose NEXT delivery is refused (dropped connection),
        # each consumed on first use — models one failed ACK.
        self.fail_types: set[str] = set()

    async def __call__(self, env: Envelope) -> Response | None:
        if env.type in self.fail_types:
            self.fail_types.discard(env.type)
            raise ConnectionError("scripted delivery failure")
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
    retriever: FakeRetriever
    classifier: FakePermissionLLM
    run: ScriptedRun
    sock: Path
    transcript: Path
    run_task: "asyncio.Task[None]" = field(repr=False, kw_only=True)

    def spy_permission(self) -> SpyPermission:
        """Replace the broker's module with a recording stand-in."""
        spy = SpyPermission(
            self.broker.permission_log_path, self.cfg.master_socket_path
        )
        self.broker.permission = spy
        return spy


def hook_env(name: str, raw_extra: dict[str, Any]) -> Envelope:
    raw: dict[str, Any] = {"hook_event_name": name, "session_id": "cc-1"}
    raw.update(raw_extra)
    return Envelope(
        id=uuid.uuid4().hex,
        type=T_HOOK_EVENT,
        session_id="cc-1",
        payload={"hook_event_name": name, "raw": raw},
    )


def permission_env(tool_name: str, tool_input: dict[str, Any]) -> Envelope:
    return Envelope(
        id=uuid.uuid4().hex,
        type=T_PERMISSION_REQUEST,
        session_id="cc-1",
        payload={
            "tool_name": tool_name,
            "tool_input": tool_input,
            "cwd": "/private/tmp/x",
            "transcript_path": "/private/tmp/x/t.jsonl",
            "permission_mode": "auto",
            "permission_suggestions": [],
        },
    )


def ask_question_env(tool_use_id: str, tool_input: dict[str, Any]) -> Envelope:
    """A blocking ask_question request, as the hook sends it."""
    return Envelope(
        id=uuid.uuid4().hex,
        type=T_ASK_QUESTION,
        session_id="cc-1",
        payload={"tool_input": tool_input, "tool_use_id": tool_use_id},
    )


def clarify_escalation_env(escalation_id: str, question: str) -> Envelope:
    return Envelope(
        id=uuid.uuid4().hex,
        type=T_CLARIFY_ESCALATION,
        session_id="s1",
        payload={"escalation_id": escalation_id, "question": question},
    )


async def decision_log_text(h: "Harness") -> str:
    """Fetch the rendered decision log over the session socket."""
    resp = await client.request(
        h.sock,
        Envelope(id=uuid.uuid4().hex, type=T_GET_DECISION_LOG, session_id="s1"),
        timeout_s=5.0,
    )
    return cast(str, resp.payload["text"])


async def wait_state(
    broker: SessionBroker, state: str, timeout: float = 5.0
) -> None:
    async with asyncio.timeout(timeout):
        while broker.state != state:
            await asyncio.sleep(0.01)


async def wait_live(
    master: StubMaster,
    pred: Callable[[dict[str, Any]], bool],
    *,
    after: int = 0,
    timeout: float = 5.0,
) -> tuple[int, dict[str, Any]]:
    """Wait for a live-status push matching ``pred``, scanning from ``after``.

    Returns:
        The absolute index of the matching envelope in ``master.received``
        and its payload, so follow-up waits can scan past it.
    """
    async with asyncio.timeout(timeout):
        while True:
            for i, env in enumerate(master.received[after:], start=after):
                if env.type == T_LIVE_STATUS and pred(env.payload):
                    return i, env.payload
            await asyncio.sleep(0.01)


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
        monkeypatch.setenv("BROKER_HOME", d)
        yield Path(d)


@asynccontextmanager
async def _harness(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    adopt: bool,
    resume: ResumedTask | None = None,
    budget_count: int = 0,
) -> AsyncGenerator[Harness]:
    transcript = home / "t.jsonl"
    shutil.copy(TRANSCRIPT_FIXTURE, transcript)
    cwd = home / "work"
    cwd.mkdir()
    run = ScriptedRun()
    monkeypatch.setattr(driver.subprocess, "run", run)
    # ScriptedRun splits no real pane, so the driver's pane-ready delay is
    # 4 dead seconds per test here.
    monkeypatch.setattr("broker.herdr.driver._PANE_READY_DELAY_S", 0.0)
    llm = FakeLLM()
    retriever = FakeRetriever()
    classifier = FakePermissionLLM()
    master = StubMaster()
    master_sock = home / "m.sock"
    master_server = await serve_unix(master_sock, master)
    cfg = SessionBrokerConfig(
        name="s1",
        socket_path=str(home / "s" / "s1.sock"),
        master_socket_path=str(master_sock),
        broker_home=home,
        cwd=str(cwd),
        anchor_pane="w3:p1",
        intent="the raw intent",
        budget_count=budget_count,
        model_id="test-model",
        max_tokens=1024,
        classifier=ClassifierConfig(model_id="test-classifier"),
        embedding=EmbeddingConfig(),
        watchdog_seconds=300.0,
        budget_max=8,
        claude_settings_path=str(home / "claude-settings.json"),
        adopt=(
            AdoptedSession(
                pane_id=ADOPTED_PANE,
                claude_session_id=ADOPTED_SESSION,
                transcript_path=str(transcript),
            )
            if adopt
            else None
        ),
        resume=resume,
    )
    broker = SessionBroker(
        cfg,
        llm_call=llm,
        retrieve=retriever,
        permission=PermissionModule(
            cfg.classifier,
            session_name=cfg.name,
            master_socket_path=cfg.master_socket_path,
            log_path=BrokerPaths(home).session_permissions(cfg.name),
            intent=cfg.intent,
            llm_call=classifier,
        ),
    )
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
        retriever=retriever,
        classifier=classifier,
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


@pytest.fixture
async def harness(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Harness]:
    async with _harness(home, monkeypatch, adopt=False) as h:
        yield h


@pytest.fixture
async def adopted(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Harness]:
    async with _harness(home, monkeypatch, adopt=True) as h:
        yield h


@pytest.fixture
async def resumed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Harness]:
    async with _harness(
        home,
        monkeypatch,
        adopt=True,
        resume=ResumedTask(approved_prompt=RESUMED_PROMPT),
        budget_count=6,
    ) as h:
        yield h


async def ground_and_approve(h: Harness, *, count: int, prompt: str) -> None:
    """Answer the grounding call, then approve the proposal it produces.

    Args:
        h: The running harness.
        count: Which proposal to approve, 1-based — reactivation raises a
            second one.
        prompt: The text the developer approves.
    """
    await h.llm.results.put(
        ToolCall(
            name="propose_prompt",
            input={"reasoning": "grounded in cwd", "prompt": "GROUNDED PROMPT"},
        )
    )
    proposal = await h.master.wait_for(T_PROMPT_PROPOSAL, count=count)
    assert proposal.payload["retrieved"] == [{"name": "src/x.py::do_it", "score": 0.7}]
    assert h.broker.state == "awaiting_approval"
    resp = await client.request(
        h.sock,
        Envelope(
            id=uuid.uuid4().hex,
            type=T_APPROVE_PROMPT,
            session_id="s1",
            payload={
                "proposal_id": proposal.payload["proposal_id"],
                "prompt": prompt,
            },
        ),
        timeout_s=5.0,
    )
    assert resp.ok is True
    await wait_state(h.broker, "driving")


async def launch(h: Harness) -> None:
    """Walk the launch sequence to the driving state."""
    await client.notify(
        h.sock,
        hook_env(
            "SessionStart",
            {"session_id": "cc-1", "transcript_path": str(h.transcript)},
        ),
    )
    await ground_and_approve(h, count=1, prompt="APPROVED PROMPT")


async def complete(h: Harness) -> None:
    """Drive one turn boundary to the completed state."""
    await client.notify(
        h.sock, hook_env("Stop", {"last_assistant_message": "all done"})
    )
    await h.llm.results.put(
        ToolCall(
            name="complete",
            input={
                "reasoning": "task finished",
                "headline": "shipped it",
                "supporting": "all tests pass",
                "task_activity": "wrapping up",
                "task_summary": "Shipped the change",
            },
        )
    )
    await h.master.wait_for(T_COMPLETION)
    await wait_state(h.broker, "completed")


async def reactivate(h: Harness, intent: str) -> Response:
    """Ask the broker to take on a new task in the same session."""
    return await client.request(
        h.sock,
        Envelope(
            id=uuid.uuid4().hex,
            type=T_REACTIVATE,
            session_id="s1",
            payload={"intent": intent},
        ),
        timeout_s=5.0,
    )


async def never_finishes() -> None:
    """A queued job that never returns, stalling the serial queue for good."""
    await asyncio.Event().wait()


async def test_permission_request_immediate_escalated_reply(
    harness: Harness,
) -> None:
    await launch(harness)
    harness.llm.never_resolve = True  # any triage work would hang forever
    await harness.classifier.script("escalate", "rm -rf is irreversible")
    start = time.monotonic()
    resp = await client.request(
        harness.sock,
        permission_env("Bash", {"command": "rm -rf /"}),
        timeout_s=5.0,
    )
    elapsed = time.monotonic() - start
    assert resp.ok is True
    assert resp.payload["decision"] == "escalated"
    assert elapsed < 0.5  # hot path


async def test_permission_request_allow_flows_to_hook_decision(
    harness: Harness,
) -> None:
    await launch(harness)
    await harness.classifier.script("allow", "reading a file in the work tree")
    resp = await client.request(
        harness.sock,
        permission_env("Read", {"file_path": "/private/tmp/x/a.py"}),
        timeout_s=5.0,
    )
    assert resp.payload["decision"] == "allow"
    # The judged call is the one that arrived, and an approval leaves the
    # session driving — only an escalation blocks it.
    assert harness.broker.state == "driving"
    assert len(harness.classifier.calls) == 1


async def test_permission_request_bypasses_serial_queue(
    harness: Harness,
) -> None:
    await launch(harness)
    harness.broker.queue.put_nowait(never_finishes)
    await asyncio.sleep(0.05)
    assert harness.broker.queue.qsize() == 0  # the stalling job is in flight
    await harness.classifier.script("allow", "a read is reversible")
    start = time.monotonic()
    resp = await client.request(
        harness.sock,
        permission_env("Read", {"file_path": "/private/tmp/x/a.py"}),
        timeout_s=5.0,
    )
    elapsed = time.monotonic() - start
    # A decision routed through the serial queue could never reach the model
    # behind a job that never finishes, so it could not answer "allow" at all.
    assert resp.payload["decision"] == "allow"
    assert elapsed < 0.5


async def test_get_permission_log_round_trip(harness: Harness) -> None:
    await launch(harness)
    await harness.classifier.script("allow", "a listing is reversible")
    await client.request(
        harness.sock,
        permission_env("Bash", {"command": "ls"}),
        timeout_s=5.0,
    )
    resp = await client.request(
        harness.sock,
        Envelope(id=uuid.uuid4().hex, type=T_GET_PERMISSION_LOG, session_id="s1"),
        timeout_s=5.0,
    )
    assert resp.ok is True
    text = cast(str, resp.payload["text"])
    assert "Bash -> allow" in text
    assert "a listing is reversible" in text
    # The permission log is its own record; triage decisions do not leak in.
    assert "grounded in cwd" not in text


async def test_hook_events_reach_module(harness: Harness) -> None:
    spy = harness.spy_permission()
    await launch(harness)
    await client.notify(
        harness.sock,
        hook_env(
            "PostToolUse",
            {"tool_name": "Bash", "tool_input": {"command": "ls"}},
        ),
    )
    await client.notify(harness.sock, hook_env("UserPromptSubmit", {}))
    await client.notify(harness.sock, hook_env("SessionEnd", {}))
    await wait_state(harness.broker, "stopped")
    assert spy.notes == [
        "tool_completed:Bash:{'command': 'ls'}",
        "developer_input",
        "session_ended",
    ]


async def test_session_end_reports_terminal_and_exits(
    harness: Harness,
) -> None:
    await launch(harness)
    await client.notify(harness.sock, hook_env("SessionEnd", {}))
    # The developer ran /exit: the broker reports the terminal end to the
    # master so the session leaves the fleet...
    await harness.master.wait_for(T_SESSION_ENDED)
    # ...and then stops serving on its own — the run task completes without a
    # cancel, so a finished session's broker never lingers on its socket.
    async with asyncio.timeout(5.0):
        await harness.run_task


async def test_ground_and_reactivate_call_set_intent(harness: Harness) -> None:
    spy = harness.spy_permission()
    await launch(harness)
    await complete(harness)
    assert (await reactivate(harness, "now write the docs")).ok is True
    await ground_and_approve(harness, count=2, prompt="THE SECOND TASK")
    assert spy.intents == ["APPROVED PROMPT", "THE SECOND TASK"]


async def test_permission_prompt_notification_shows_up_on_status(
    harness: Harness,
) -> None:
    await launch(harness)
    await client.notify(
        harness.sock,
        hook_env("Notification", {"notification_type": "permission_prompt"}),
    )
    await wait_live(harness.master, lambda p: p["permission_prompt"] is True)
    resp = await client.request(
        harness.sock,
        Envelope(id=uuid.uuid4().hex, type=T_STATUS, session_id="s1"),
        timeout_s=5.0,
    )
    assert resp.payload["permission_prompt"] is True
    await client.notify(
        harness.sock,
        hook_env("PostToolUse", {"tool_name": "Bash", "tool_input": {}}),
    )
    await asyncio.sleep(0.1)
    resp = await client.request(
        harness.sock,
        Envelope(id=uuid.uuid4().hex, type=T_STATUS, session_id="s1"),
        timeout_s=5.0,
    )
    assert resp.payload["permission_prompt"] is False


async def test_task_activity_persists_across_pushes_and_status_probe(
    harness: Harness,
) -> None:
    await launch(harness)
    await client.notify(
        harness.sock,
        hook_env("Stop", {"last_assistant_message": "Which auth provider?"}),
    )
    await harness.llm.results.put(ANSWER_RESULT)
    idx, _ = await wait_live(
        harness.master, lambda p: p.get("task_activity") == "wiring up oauth"
    )
    resp = await client.request(
        harness.sock,
        Envelope(id=uuid.uuid4().hex, type=T_STATUS, session_id="s1"),
        timeout_s=5.0,
    )
    assert resp.payload["task_activity"] == "wiring up oauth"
    # A later, unrelated push still carries the same task description.
    await client.notify(
        harness.sock,
        hook_env("Notification", {"notification_type": "permission_prompt"}),
    )
    _, later = await wait_live(
        harness.master, lambda p: p["permission_prompt"] is True, after=idx + 1
    )
    assert later["task_activity"] == "wiring up oauth"


async def test_agent_start_forwards_this_sessions_settings_file(
    harness: Harness,
) -> None:
    await launch(harness)
    start = next(c for c in harness.run.calls if c[1:3] == ["agent", "start"])
    forwarded = start[start.index("--") + 1:]
    assert forwarded[-2:] == ["--settings", harness.cfg.claude_settings_path]


async def test_stop_triggers_triage_and_answer_submits(
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


async def test_ask_question_escalates_and_retracts_on_answer(
    harness: Harness,
) -> None:
    await launch(harness)
    await harness.llm.results.put(ESCALATE_RESULT)
    resp = await client.request(
        harness.sock,
        ask_question_env(PENDING_ASK_ID, COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert resp.ok is True
    assert resp.payload["decision"] == "escalated"
    escalation = await harness.master.wait_for(T_ESCALATION)
    # Wrap, never rewrite: the LLM's situation text survives verbatim and the
    # pane note is appended.
    assert escalation.payload["situation"].startswith(
        "the plan contradicts the code"
    )
    assert escalation.payload["situation"].endswith(
        "The native menu is on screen in pane w3:p2 — answer it directly in "
        "that pane."
    )
    assert escalation.payload["what_was_asked"] == "how to proceed"
    assert escalation.payload["uncertainty"] == "whether the plan is stale"
    await wait_state(harness.broker, "escalated")
    # A duplicate ask for the same tool_use_id gets the cached decision with
    # zero extra LLM calls, and must not re-escalate.
    calls_before = len(harness.llm.calls)
    dup = await client.request(
        harness.sock,
        ask_question_env(PENDING_ASK_ID, COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert dup.payload == resp.payload
    assert len(harness.llm.calls) == calls_before
    await asyncio.sleep(0.1)
    assert len(harness.master.of_type(T_ESCALATION)) == 1
    # The transcript contains the paired answer -> next hook event retracts.
    await client.notify(harness.sock, hook_env("PostToolUse", {}))
    retract = await harness.master.wait_for(T_ESCALATION_RETRACT)
    assert retract.payload["escalation_id"] == escalation.payload["escalation_id"]
    assert retract.payload["reason"] == "resolved in pane"
    await wait_state(harness.broker, "driving")


async def test_ask_question_answered_injects_and_updates_budget(
    harness: Harness,
) -> None:
    await launch(harness)
    await harness.llm.results.put(ANSWER_QUESTIONS_RESULT)
    resp = await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert resp.ok is True
    assert resp.payload["decision"] == "answer"
    assert resp.payload["updated_input"] == {
        **COLOR_TOOL_INPUT,
        "answers": {"Pick a color": "Blue (Recommended)"},
    }
    log_text = await decision_log_text(harness)
    assert "ask_answered" in log_text
    assert "blue matches the intent" in log_text
    assert harness.broker.budget_count == 1
    budget = await harness.master.wait_for(T_BUDGET_UPDATE)
    assert budget.payload == {"count": 1}
    assert harness.master.of_type(T_ESCALATION) == []
    assert harness.broker.state == "driving"


async def test_ask_question_duplicate_returns_cached_answer(
    harness: Harness,
) -> None:
    await launch(harness)
    await harness.llm.results.put(ANSWER_QUESTIONS_RESULT)
    first = await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    calls_before = len(harness.llm.calls)
    second = await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert second.payload == first.payload
    assert len(harness.llm.calls) == calls_before
    assert harness.broker.budget_count == 1  # not double-counted


async def test_ask_question_invalid_then_valid_retries_once(
    harness: Harness,
) -> None:
    await launch(harness)
    calls_before = len(harness.llm.calls)
    await harness.llm.results.put(INVALID_ANSWER_QUESTIONS_RESULT)
    await harness.llm.results.put(ANSWER_QUESTIONS_RESULT)
    resp = await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert resp.payload["decision"] == "answer"
    assert len(harness.llm.calls) == calls_before + 2


async def test_ask_question_invalid_twice_escalates(harness: Harness) -> None:
    await launch(harness)
    await harness.llm.results.put(INVALID_ANSWER_QUESTIONS_RESULT)
    await harness.llm.results.put(INVALID_ANSWER_QUESTIONS_RESULT)
    resp = await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert resp.payload["decision"] == "escalated"
    escalation = await harness.master.wait_for(T_ESCALATION)
    assert "AnswerValidationError" in escalation.payload["uncertainty"]


async def test_ask_question_llm_failure_escalates(harness: Harness) -> None:
    await launch(harness)
    harness.llm.raise_error = LLMCallError("scripted failure")
    resp = await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert resp.payload["decision"] == "escalated"
    escalation = await harness.master.wait_for(T_ESCALATION)
    assert "LLMCallError" in escalation.payload["uncertainty"]


async def test_ask_question_malformed_payload_escalates(
    harness: Harness,
) -> None:
    await launch(harness)
    calls_before = len(harness.llm.calls)
    resp = await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", {"questions": []}),
        timeout_s=5.0,
    )
    assert resp.payload["decision"] == "escalated"
    escalation = await harness.master.wait_for(T_ESCALATION)
    assert "unusable question payload" in escalation.payload["uncertainty"]
    assert len(harness.llm.calls) == calls_before  # no LLM call


async def test_ask_question_budget_exhausted_escalates_without_llm(
    harness: Harness,
) -> None:
    await launch(harness)
    harness.broker.budget_count = harness.cfg.budget_max
    calls_before = len(harness.llm.calls)
    resp = await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert resp.payload["decision"] == "escalated"
    escalation = await harness.master.wait_for(T_ESCALATION)
    assert "budget exhausted" in escalation.payload["uncertainty"]
    assert len(harness.llm.calls) == calls_before  # no LLM call


async def test_ask_question_in_non_driving_state_skips_escalation(
    harness: Harness,
) -> None:
    await launch(harness)
    await complete(harness)
    calls_before = len(harness.llm.calls)
    resp = await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert resp.payload["decision"] == "escalated"
    assert len(harness.llm.calls) == calls_before
    await asyncio.sleep(0.1)
    assert harness.master.of_type(T_ESCALATION) == []  # no second raise
    assert "ask_skipped" in await decision_log_text(harness)


async def test_ask_verified_on_matching_post_tool_use(harness: Harness) -> None:
    await launch(harness)
    await harness.llm.results.put(ANSWER_QUESTIONS_RESULT)
    await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    await client.notify(
        harness.sock,
        hook_env(
            "PostToolUse",
            {
                "tool_name": "AskUserQuestion",
                "tool_use_id": "toolu_new_1",
                "tool_input": COLOR_TOOL_INPUT,
                "tool_response": {
                    "questions": COLOR_TOOL_INPUT["questions"],
                    "answers": {"Pick a color": "Blue (Recommended)"},
                },
            },
        ),
    )
    async with asyncio.timeout(5.0):
        while "ask_verified" not in await decision_log_text(harness):
            await asyncio.sleep(0.01)
    assert harness.master.of_type(T_ESCALATION) == []
    assert harness.broker.state == "driving"


async def test_ask_verify_mismatch_escalates_without_auto_retract(
    harness: Harness,
) -> None:
    await launch(harness)
    await harness.llm.results.put(ANSWER_QUESTIONS_RESULT)
    await client.request(
        harness.sock,
        ask_question_env("toolu_new_1", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    await client.notify(
        harness.sock,
        hook_env(
            "PostToolUse",
            {
                "tool_name": "AskUserQuestion",
                "tool_use_id": "toolu_new_1",
                "tool_input": COLOR_TOOL_INPUT,
                "tool_response": {
                    "questions": COLOR_TOOL_INPUT["questions"],
                    "answers": {"Pick a color": "Red"},
                },
            },
        ),
    )
    escalation = await harness.master.wait_for(T_ESCALATION)
    assert "different answers" in escalation.payload["situation"]
    await wait_state(harness.broker, "escalated")
    # The answer already exists in the session; id-existence resolution would
    # self-retract this before the developer saw it. It must stay raised.
    await client.notify(harness.sock, hook_env("PostToolUse", {}))
    await client.notify(
        harness.sock, hook_env("Stop", {"last_assistant_message": "moving on"})
    )
    await asyncio.sleep(0.2)
    assert harness.master.of_type(T_ESCALATION_RETRACT) == []
    assert harness.broker.state == "escalated"


async def test_ask_verify_backstop_escalates_then_retracts_on_answer(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("broker.session.broker.ASK_VERIFY_TIMEOUT_S", 0.1)
    await launch(harness)
    await harness.llm.results.put(ANSWER_QUESTIONS_RESULT)
    await client.request(
        harness.sock,
        ask_question_env("toolu_backstop", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    # No PostToolUse arrives and the transcript never records the answer.
    escalation = await harness.master.wait_for(T_ESCALATION)
    assert "no answer was recorded" in escalation.payload["situation"]
    await wait_state(harness.broker, "escalated")
    # The developer answers the still-open menu in the pane: the answer id
    # appears in the transcript and the escalation auto-retracts.
    question_record = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "AskUserQuestion",
                    "id": "toolu_backstop",
                    "input": COLOR_TOOL_INPUT,
                }
            ]
        },
    }
    answer_record = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_backstop",
                    "content": "answered in the pane",
                }
            ],
        },
    }
    with harness.transcript.open("a") as f:
        f.write(json.dumps(question_record) + "\n")
        f.write(json.dumps(answer_record) + "\n")
    await client.notify(harness.sock, hook_env("PostToolUse", {}))
    retract = await harness.master.wait_for(T_ESCALATION_RETRACT)
    assert retract.payload["escalation_id"] == escalation.payload["escalation_id"]
    await wait_state(harness.broker, "driving")


async def test_ask_verify_backstop_read_failure_escalates_without_auto_retract(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("broker.session.broker.ASK_VERIFY_TIMEOUT_S", 0.1)
    await launch(harness)
    await harness.llm.results.put(ANSWER_QUESTIONS_RESULT)
    await client.request(
        harness.sock,
        ask_question_env("toolu_read_fail", COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    # The backstop's transcript read blows up: verification must escalate,
    # never vanish into the task.
    harness.transcript.unlink()
    escalation = await harness.master.wait_for(T_ESCALATION)
    assert "verification itself failed" in escalation.payload["situation"]
    await wait_state(harness.broker, "escalated")
    # The broker is flying blind, so an answer id in the transcript must not
    # self-retract this before the developer saw it.
    shutil.copy(TRANSCRIPT_FIXTURE, harness.transcript)
    with harness.transcript.open("a") as f:
        f.write(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "AskUserQuestion",
                                "id": "toolu_read_fail",
                                "input": COLOR_TOOL_INPUT,
                            }
                        ]
                    },
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_read_fail",
                                "content": "answered in the pane",
                            }
                        ],
                    },
                }
            )
            + "\n"
        )
    await client.notify(harness.sock, hook_env("PostToolUse", {}))
    await client.notify(
        harness.sock, hook_env("Stop", {"last_assistant_message": "moving on"})
    )
    await asyncio.sleep(0.2)
    assert harness.master.of_type(T_ESCALATION_RETRACT) == []
    assert harness.broker.state == "escalated"


async def test_stop_that_resolves_an_escalation_still_triages_its_turn(
    harness: Harness,
) -> None:
    # The Stop ending the turn the developer answered in the pane is the same
    # Stop carrying the question they left open. Retracting without triaging
    # it strands the pane: no further hook is coming to trigger one.
    await launch(harness)
    await harness.llm.results.put(ESCALATE_RESULT)
    resp = await client.request(
        harness.sock,
        ask_question_env(PENDING_ASK_ID, COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert resp.payload["decision"] == "escalated"
    escalation = await harness.master.wait_for(T_ESCALATION)
    await wait_state(harness.broker, "escalated")
    harness.run.calls.clear()

    await harness.llm.results.put(ANSWER_RESULT)
    await client.notify(
        harness.sock,
        hook_env("Stop", {"last_assistant_message": "Which auth provider?"}),
    )
    retract = await harness.master.wait_for(T_ESCALATION_RETRACT)
    assert retract.payload["escalation_id"] == escalation.payload["escalation_id"]
    await harness.master.wait_for(T_BUDGET_UPDATE)
    assert harness.run.drive_calls() == [
        ["herdr", "agent", "prompt", "s1", "use oauth"],
    ]
    triage_call = harness.llm.calls[-1]
    content = cast(list[dict[str, Any]], triage_call["messages"][0]["content"])
    assert "Which auth provider?" in content[-1]["text"]


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
    ]
    assert harness.broker.budget_count == 0
    await wait_state(harness.broker, "driving")
    # Resolution is bound to real delivery: the broker confirms the decision
    # reached the pane so the master resolves the escalation, not on the ACK.
    delivered = await harness.master.wait_for(T_DECISION_DELIVERED)
    assert delivered.payload["escalation_id"] == escalation.payload["escalation_id"]


async def escalate_via_stop(h: Harness) -> str:
    """Drive one Stop through triage to a raised escalation; return its id."""
    await client.notify(
        h.sock, hook_env("Stop", {"last_assistant_message": "proceed how?"})
    )
    await h.llm.results.put(ESCALATE_RESULT)
    escalation = await h.master.wait_for(T_ESCALATION)
    await wait_state(h.broker, "escalated")
    return cast(str, escalation.payload["escalation_id"])


async def test_clarify_escalation_answers(harness: Harness) -> None:
    await launch(harness)
    escalation_id = await escalate_via_stop(harness)
    harness.run.calls.clear()
    await harness.llm.results.put(CLARIFY_RESULT)
    resp = await client.request(
        harness.sock,
        clarify_escalation_env(escalation_id, "what did it already try?"),
        timeout_s=5.0,
    )
    assert resp.ok is True
    assert (
        ClarifyEscalationReplyPayload.model_validate(resp.payload).answer
        == "it tried A first"
    )
    call = harness.llm.calls[-1]
    assert call["model"] == "test-model"
    content = cast(list[dict[str, Any]], call["messages"][0]["content"])
    assert "what did it already try?" in content[-1]["text"]
    assert "the plan contradicts the code" in content[-1]["text"]
    # The escalation is untouched: still pending, nothing typed, nothing sent.
    assert harness.broker.state == "escalated"
    assert harness.run.drive_calls() == []
    assert len(harness.master.of_type(T_ESCALATION)) == 1
    assert harness.master.of_type(T_ESCALATION_RETRACT) == []
    assert "clarified" in await decision_log_text(harness)


async def test_clarify_escalation_wrong_id_refused(harness: Harness) -> None:
    await launch(harness)
    await escalate_via_stop(harness)
    calls_before = len(harness.llm.calls)
    resp = await client.request(
        harness.sock, clarify_escalation_env("bogus", "anything?"), timeout_s=5.0
    )
    assert resp.ok is False
    assert resp.payload["reason_code"] == NACK_WRONG_STATE
    assert len(harness.llm.calls) == calls_before
    assert harness.broker.state == "escalated"


async def test_clarify_escalation_not_escalated_refused(harness: Harness) -> None:
    await launch(harness)
    calls_before = len(harness.llm.calls)
    resp = await client.request(
        harness.sock, clarify_escalation_env("e1", "anything?"), timeout_s=5.0
    )
    assert resp.ok is False
    assert resp.payload["reason_code"] == NACK_WRONG_STATE
    assert len(harness.llm.calls) == calls_before


async def test_clarify_escalation_cancelled_by_retract(harness: Harness) -> None:
    await launch(harness)
    await harness.llm.results.put(ESCALATE_RESULT)
    resp = await client.request(
        harness.sock,
        ask_question_env(PENDING_ASK_ID, COLOR_TOOL_INPUT),
        timeout_s=5.0,
    )
    assert resp.payload["decision"] == "escalated"
    escalation = await harness.master.wait_for(T_ESCALATION)
    await wait_state(harness.broker, "escalated")
    calls_before = len(harness.llm.calls)
    harness.llm.never_resolve = True
    asking = asyncio.create_task(
        client.request(
            harness.sock,
            clarify_escalation_env(
                escalation.payload["escalation_id"], "what did it try?"
            ),
            timeout_s=5.0,
        )
    )
    async with asyncio.timeout(5.0):
        while len(harness.llm.calls) == calls_before:
            await asyncio.sleep(0.01)
    # The transcript already holds the paired answer: this hook retracts.
    await client.notify(harness.sock, hook_env("PostToolUse", {}))
    retract = await harness.master.wait_for(T_ESCALATION_RETRACT)
    assert retract.payload["escalation_id"] == escalation.payload["escalation_id"]
    answer = await asking
    assert answer.ok is False
    assert answer.payload["error"] == "escalation resolved in the pane"
    assert answer.payload["reason_code"] == NACK_WRONG_STATE
    assert harness.broker._clarify_tasks == set()  # pyright: ignore[reportPrivateUsage]
    await wait_state(harness.broker, "driving")


async def test_clarify_escalation_timeout_fails_loud_and_cancels_the_call(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("broker.session.broker.CLARIFY_TIMEOUT_S", 0.1)
    await launch(harness)
    escalation_id = await escalate_via_stop(harness)
    harness.llm.never_resolve = True
    asking = asyncio.create_task(
        client.request(
            harness.sock,
            clarify_escalation_env(escalation_id, "what did it try?"),
            timeout_s=5.0,
        )
    )
    in_flight = harness.broker._clarify_tasks  # pyright: ignore[reportPrivateUsage]
    async with asyncio.timeout(5.0):
        while not in_flight:
            await asyncio.sleep(0.01)
    llm_task = next(iter(in_flight))
    resp = await asking
    assert resp.ok is False
    assert "TimeoutError" in resp.payload["error"]
    # The timed-out LLM call is cancelled with the connection's wait, so
    # nothing is left for a later dispatch or retract to chase.
    assert llm_task.cancelled()
    assert in_flight == set()
    assert harness.broker.state == "escalated"
    assert harness.master.of_type(T_ESCALATION_RETRACT) == []
    assert "clarify_failed" in await decision_log_text(harness)


async def test_stale_dispatch_decision_is_reported_not_submitted(
    harness: Harness,
) -> None:
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
    # Nothing reaches the pane, and the miss is reported as no-longer-held so
    # the master drops its orphaned queue entry.
    undelivered = await harness.master.wait_for(T_DECISION_UNDELIVERED)
    assert undelivered.payload["escalation_id"] == "stale-id"
    assert undelivered.payload["still_live"] is False
    assert harness.run.drive_calls() == []


async def test_dispatch_decision_submit_failure_reports_undelivered(
    harness: Harness,
) -> None:
    await launch(harness)
    escalation_id = await escalate_via_stop(harness)
    harness.run.calls.clear()
    harness.run.fail_prompt = RuntimeError("pane gone")
    await client.request(
        harness.sock,
        Envelope(
            id=uuid.uuid4().hex,
            type=T_DISPATCH_DECISION,
            session_id="s1",
            payload={"escalation_id": escalation_id, "response": "go with a"},
        ),
        timeout_s=5.0,
    )
    # A failed pane write is a loud miss (still_live), never confirmed as
    # delivered: the broker holds the escalation and stays ESCALATED so the
    # master leaves it surfaced for a re-decide, with no fatal.
    undelivered = await harness.master.wait_for(T_DECISION_UNDELIVERED)
    assert undelivered.payload["escalation_id"] == escalation_id
    assert "pane gone" in undelivered.payload["detail"]
    assert undelivered.payload["still_live"] is True
    assert harness.master.of_type(T_DECISION_DELIVERED) == []
    assert harness.broker.state == "escalated"  # still held, awaiting re-decide
    assert "dispatch_failed" in await decision_log_text(harness)


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


async def test_retrieval_failure_is_fatal_not_a_proposal(harness: Harness) -> None:
    harness.retriever.raise_error = RetrievalError("no code index for /x; run …")
    await client.notify(
        harness.sock,
        hook_env(
            "SessionStart",
            {"session_id": "cc-1", "transcript_path": str(harness.transcript)},
        ),
    )
    fatal = await harness.master.wait_for(T_FATAL_ERROR)
    assert fatal.payload["error_class"] == "RetrievalError"
    assert "no code index" in fatal.payload["detail"]
    assert harness.master.of_type(T_PROMPT_PROPOSAL) == []
    assert harness.broker.state == "error"


async def test_adopted_broker_takes_over_without_touching_the_pane(
    adopted: Harness,
) -> None:
    # No SessionStart is sent: the hook fired for the outgoing broker and will
    # not fire again. An adopting broker that waited for it would hang, and one
    # that split a pane would strand the developer's live chat.
    await ground_and_approve(adopted, count=1, prompt="THE HANDOVER TASK")
    assert [c for c in adopted.run.calls if c[1:3] == ["pane", "split"]] == []
    assert [c for c in adopted.run.calls if c[1:3] == ["agent", "start"]] == []
    assert adopted.run.drive_calls() == [
        ["herdr", "agent", "prompt", "s1", "THE HANDOVER TASK"],
    ]
    resp = await client.request(
        adopted.sock,
        Envelope(id=uuid.uuid4().hex, type=T_STATUS, session_id="s1"),
        timeout_s=5.0,
    )
    assert resp.payload["pane_id"] == ADOPTED_PANE
    assert resp.payload["claude_session_id"] == ADOPTED_SESSION
    assert resp.payload["transcript_path"] == str(adopted.transcript)


async def test_resume_skips_grounding(resumed: Harness) -> None:
    # No grounding, no proposal, no pane write: there is no new task, and the
    # developer already approved this prompt once.
    await wait_state(resumed.broker, "driving")
    assert resumed.llm.calls == []
    assert resumed.run.calls == []
    assert resumed.master.of_type(T_PROMPT_PROPOSAL) == []
    assert resumed.broker.approved_prompt == RESUMED_PROMPT
    # The permission module judges against the resumed task, not the raw
    # intent the config also carries.
    assert resumed.broker.permission.intent == RESUMED_PROMPT


async def test_resume_restores_completed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _harness(
        home,
        monkeypatch,
        adopt=True,
        resume=ResumedTask(approved_prompt=RESUMED_PROMPT, completed=True),
    ) as h:
        await wait_state(h.broker, "completed")
        assert h.llm.calls == []
        # A resumed-completed session still takes a new task the normal way.
        assert (await reactivate(h, "now write the docs")).ok is True
        await ground_and_approve(h, count=1, prompt="THE SECOND TASK")
        assert h.run.drive_calls() == [
            ["herdr", "agent", "prompt", "s1", "THE SECOND TASK"],
        ]


async def test_resume_budget_continues(resumed: Harness) -> None:
    await wait_state(resumed.broker, "driving")
    await client.notify(
        resumed.sock,
        hook_env("Stop", {"last_assistant_message": "Which auth provider?"}),
    )
    await resumed.llm.results.put(ANSWER_RESULT)
    budget = await resumed.master.wait_for(T_BUDGET_UPDATE)
    # The persisted count continues; a reattach must not refill the budget.
    assert budget.payload == {"count": 7}
    assert resumed.broker.budget_count == 7


async def test_reactivate_grounds_new_task_and_supersedes_the_old_intent(
    harness: Harness,
) -> None:
    await launch(harness)
    await complete(harness)
    harness.broker.budget_count = 5
    harness.run.calls.clear()
    assert (await reactivate(harness, "now write the docs")).ok is True
    await ground_and_approve(harness, count=2, prompt="THE SECOND TASK")
    assert harness.run.drive_calls() == [
        ["herdr", "agent", "prompt", "s1", "THE SECOND TASK"],
    ]
    # Developer contact resets the autonomous answer budget.
    assert harness.broker.budget_count == 0
    assert harness.master.of_type(T_BUDGET_UPDATE)[-1].payload == {"count": 0}
    # The new approved prompt is the authoritative intent from here on;
    # triaging the second task against the first one's would misclassify it.
    await client.notify(
        harness.sock, hook_env("Stop", {"last_assistant_message": "which format?"})
    )
    await harness.llm.results.put(ANSWER_RESULT)
    await harness.master.wait_for(T_BUDGET_UPDATE, count=2)
    content = cast(
        list[dict[str, Any]], harness.llm.calls[-1]["messages"][0]["content"]
    )
    assert "THE SECOND TASK" in content[0]["text"]
    assert "APPROVED PROMPT" not in content[0]["text"]


async def test_reactivate_refused_while_a_task_is_still_running(
    harness: Harness,
) -> None:
    await launch(harness)  # driving, not completed
    harness.run.calls.clear()
    resp = await reactivate(harness, "do something else instead")
    assert resp.ok is False
    assert "driving" in resp.payload["error"]
    await asyncio.sleep(0.1)
    # A refused reactivation must not displace the running task.
    assert harness.run.drive_calls() == []
    assert harness.broker.state == "driving"


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


async def test_state_changes_push_live_status_with_activity(
    harness: Harness,
) -> None:
    h = harness
    await client.notify(
        h.sock,
        hook_env(
            "SessionStart",
            {"session_id": "cc-1", "transcript_path": str(h.transcript)},
        ),
    )
    # The grounding call is parked on the empty fake LLM, so the push carries
    # the state and the phrase together.
    _, p = await wait_live(h.master, lambda p: p["state"] == "grounding")
    assert PHRASE_GROUNDING in p["activity"]
    await ground_and_approve(h, count=1, prompt="APPROVED PROMPT")
    # Grounding done: the phrase was discarded and the state moved on.
    _, p = await wait_live(
        h.master,
        lambda p: p["state"] == "driving" and p["activity"] == "",
    )
    assert p["permission_prompt"] is False


async def test_concurrent_phrases_both_appear_and_clear_independently(
    harness: Harness,
) -> None:
    h = harness
    await launch(h)
    # Turn triage in flight on the serial event loop (empty LLM queue).
    await client.notify(
        h.sock, hook_env("Stop", {"last_assistant_message": "which way?"})
    )
    await wait_live(h.master, lambda p: PHRASE_TRIAGE in p["activity"])
    # A permission decision in flight on a socket-handler task, concurrently.
    req = asyncio.create_task(
        client.request(
            h.sock, permission_env("Bash", {"command": "ls"}), timeout_s=5.0
        )
    )
    both, _ = await wait_live(
        h.master,
        lambda p: PHRASE_TRIAGE in p["activity"]
        and PHRASE_PERMISSION in p["activity"],
    )
    # The permission decision resolves and clears ONLY its own phrase.
    await h.classifier.script("allow", "a listing is reversible")
    resp = await req
    assert resp.payload["decision"] == "allow"
    cleared, _ = await wait_live(
        h.master,
        lambda p: PHRASE_TRIAGE in p["activity"]
        and PHRASE_PERMISSION not in p["activity"],
        after=both + 1,
    )
    # The triage resolves and the activity goes idle.
    await h.llm.results.put(ANSWER_RESULT)
    await wait_live(
        h.master, lambda p: p["activity"] == "", after=cleared + 1
    )


async def test_permission_prompt_notification_is_pushed(
    harness: Harness,
) -> None:
    h = harness
    await launch(h)
    mark = len(h.master.received)
    await client.notify(
        h.sock,
        hook_env("Notification", {"notification_type": "permission_prompt"}),
    )
    _, p = await wait_live(
        h.master, lambda p: p["permission_prompt"] is True, after=mark
    )
    # The prompt is a flag, not a state: the broker keeps driving.
    assert p["state"] == "driving"


async def test_failed_push_is_retried(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("broker.session.broker.STATUS_RETRY_S", 0.05)
    h = harness
    await launch(h)
    await wait_live(h.master, lambda p: p["state"] == "driving")
    mark = len(h.master.received)
    h.master.fail_types.add(T_LIVE_STATUS)
    # Exactly one change after the scripted failure: only the retry can
    # deliver it.
    await client.notify(
        h.sock,
        hook_env("Notification", {"notification_type": "permission_prompt"}),
    )
    await wait_live(
        h.master, lambda p: p["permission_prompt"] is True, after=mark
    )
    assert h.master.fail_types == set()  # the first attempt really failed


async def test_shutdown_cancels_the_status_sender_cleanly(
    harness: Harness,
) -> None:
    h = harness
    await launch(h)
    resp = await client.request(
        h.sock,
        Envelope(id=uuid.uuid4().hex, type=T_SHUTDOWN, session_id="s1"),
        timeout_s=5.0,
    )
    assert resp.ok is True
    # run() returns only once its finally cancelled and awaited the sender;
    # a sender parked in its wait() would hang this forever.
    async with asyncio.timeout(5.0):
        await h.run_task
