"""MasterRuntime harness: fake broker = protocol.client against the real
runtime server; driver.subprocess.run monkeypatched; app_post = recording list."""

import asyncio
import contextlib
import json
import logging
import subprocess
import tempfile
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from broker.config import BrokerConfig
from broker.herdr import driver
from broker.master.messages import (
    CompletionArrived,
    EscalationArrived,
    Notice,
    ProposalArrived,
)
from broker.master.registry import Registry, SessionRecord
from broker.master.runtime import MasterRuntime, render_escalation
from broker.protocol import client
from broker.protocol.constants import (
    SessionState,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_FATAL_ERROR,
    T_PROMPT_PROPOSAL,
    T_REACTIVATE,
    T_RETRACT,
)
from broker.protocol.schemas import Envelope, EscalationPayload, Response
from broker.protocol.server import serve_unix


class RecordingRun:
    """Stand-in for subprocess.run inside the herdr driver."""

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
        return subprocess.CompletedProcess(list(argv), 0, "{}", "")


class StubSession:
    """Recording session-socket handler behind a real serve_unix."""

    def __init__(self) -> None:
        self.envelopes: list[Envelope] = []

    async def handler(self, env: Envelope) -> Response:
        self.envelopes.append(env)
        return Response(id=env.id, ok=True)


class NackingSession:
    """Session-socket handler that always rejects."""

    async def handler(self, env: Envelope) -> Response:
        return Response(id=env.id, ok=False)


class ReasoningNackSession:
    """Session-socket handler that rejects and says why."""

    reason = "session is 'driving', not 'completed'"

    async def handler(self, env: Envelope) -> Response:
        return Response(id=env.id, ok=False, payload={"error": self.reason})


class FakeProcess:
    """Stand-in for the session-broker subprocess."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -15


class RecordingSpawn:
    """Stand-in for asyncio.create_subprocess_exec that records each argv."""

    def __init__(self) -> None:
        self.argvs: list[tuple[str, ...]] = []

    async def __call__(self, *argv: str) -> FakeProcess:
        self.argvs.append(argv)
        return FakeProcess(4242 + len(self.argvs))

    def config(self) -> dict[str, Any]:
        """Decode --config-json from the most recent spawn."""
        argv = self.argvs[-1]
        return cast(
            dict[str, Any], json.loads(argv[argv.index("--config-json") + 1])
        )


class MalformedLogSession:
    """Session-socket handler answering get_decision_log with no text field."""

    async def handler(self, env: Envelope) -> Response:
        return Response(id=env.id, ok=True, payload={"oops": "not text"})


def escalation_dict(esc_id: str = "e1") -> dict[str, Any]:
    return {
        "escalation_id": esc_id,
        "session_id": "s1",
        "task_context": "ctx-task-value",
        "situation": "situation-value",
        "what_was_asked": "asked-value",
        "what_is_at_stake": "stake-value",
        "alternatives": [
            {"option": "opt-a", "pros": "pros-a", "cons": "cons-a"},
            {"option": "opt-b", "pros": "pros-b", "cons": "cons-b"},
        ],
        "recommendation": "recommendation-value",
        "uncertainty": "uncertainty-value",
        "what_would_change_my_mind": "change-mind-value",
    }


@pytest.fixture
def recording_run(monkeypatch: pytest.MonkeyPatch) -> RecordingRun:
    rec = RecordingRun()
    monkeypatch.setattr(driver.subprocess, "run", rec)
    return rec


@pytest.fixture
def spawn(monkeypatch: pytest.MonkeyPatch) -> RecordingSpawn:
    rec = RecordingSpawn()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", rec)
    return rec


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        monkeypatch.setenv("BROKER_HOME", td)
        yield Path(td)


@pytest.fixture
async def rt(
    home: Path, recording_run: RecordingRun
) -> AsyncIterator[tuple[MasterRuntime, list[Any]]]:
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    registry.upsert(
        SessionRecord(
            name="s1",
            socket_path=str(home / "s" / "s1.sock"),
            cwd="/private/tmp",
            anchor_pane="%1",
            state=SessionState.DRIVING,
        )
    )
    posts: list[Any] = []
    runtime = MasterRuntime(posts.append, registry, cfg, anchor_pane="%1")
    task = asyncio.create_task(runtime.serve())
    for _ in range(200):
        if runtime.master_socket_path.exists():
            break
        await asyncio.sleep(0.01)
    else:
        raise TimeoutError("master socket never bound")
    yield runtime, posts
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def send(
    runtime: MasterRuntime,
    msg_type: str,
    payload: dict[str, Any],
    session: str = "s1",
) -> Response:
    env = Envelope(
        id=uuid.uuid4().hex, type=msg_type, session_id=session, payload=payload
    )
    return await client.request(
        runtime.master_socket_path, env, timeout_s=5.0
    )


def _leaf_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:  # pyright: ignore[reportUnknownVariableType]
            yield from _leaf_values(item)
    elif isinstance(value, dict):
        for item in value.values():  # pyright: ignore[reportUnknownVariableType]
            yield from _leaf_values(item)


async def test_escalation_rendered_verbatim(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    payload = escalation_dict()
    resp = await send(runtime, T_ESCALATION, payload)
    assert resp.ok
    arrived = [m for m in posts if isinstance(m, EscalationArrived)]
    assert len(arrived) == 1
    rendered = arrived[0].rendered
    # Every payload field value appears byte-for-byte — no paraphrase.
    for value in _leaf_values(payload):
        assert value in rendered
    assert rendered == render_escalation(
        EscalationPayload.model_validate(payload)
    )
    assert runtime.slot.active is not None
    assert runtime.slot.active.escalation_id == "e1"


async def test_second_escalation_while_active_is_protocol_violation(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    resp = await send(runtime, T_ESCALATION, escalation_dict("e2"))
    assert not resp.ok
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("PROTOCOL VIOLATION" in t for t in notices)
    # The first escalation stays active; the second is never surfaced.
    assert runtime.slot.active is not None
    assert runtime.slot.active.escalation_id == "e1"
    assert (
        len([m for m in posts if isinstance(m, EscalationArrived)]) == 1
    )


async def test_retract_clears_slot_and_informs(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    resp = await send(
        runtime,
        T_RETRACT,
        {"escalation_id": "e1", "reason": "resolved in pane"},
    )
    assert resp.ok
    assert runtime.slot.active is None
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("resolved in pane" in t for t in notices)


async def test_dispatch_aborts_on_stale_escalation(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        # No active escalation at all.
        result = await runtime.dispatch("ghost", "option B")
        assert "NOT dispatched" in result
        # Mismatched id while another escalation is active.
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        result = await runtime.dispatch("e2", "option B")
        assert "NOT dispatched" in result
        assert stub.envelopes == []  # nothing ever reached the session
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("NOT dispatched" in t for t in notices)
    finally:
        server.close()
        await server.wait_closed()


async def test_dispatch_delivers_decision_when_live(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        result = await runtime.dispatch("e1", "use option B")
        assert "dispatched" in result
        assert len(stub.envelopes) == 1
        env = stub.envelopes[0]
        assert env.type == T_DISPATCH_DECISION
        assert env.payload["escalation_id"] == "e1"
        assert env.payload["response"] == "use option B"
        assert runtime.slot.active is None
        assert runtime.registry.get("s1").state == "driving"
    finally:
        server.close()
        await server.wait_closed()


async def test_budget_update_persists_to_registry(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    resp = await send(runtime, T_BUDGET_UPDATE, {"count": 5})
    assert resp.ok
    assert runtime.registry.get("s1").budget_count == 5
    reloaded = Registry.load(home / "registry.json")
    assert reloaded.get("s1").budget_count == 5


async def test_completion_notifies_done(
    rt: tuple[MasterRuntime, list[Any]], recording_run: RecordingRun
) -> None:
    runtime, posts = rt
    resp = await send(runtime, T_COMPLETION, {"summary": "did the thing"})
    assert resp.ok
    arrived = [m for m in posts if isinstance(m, CompletionArrived)]
    assert len(arrived) == 1
    assert arrived[0].summary == "did the thing"  # verbatim
    notify_calls = [
        c for c in recording_run.calls if c[1:3] == ["notification", "show"]
    ]
    assert len(notify_calls) == 1
    argv = notify_calls[0]
    assert argv[argv.index("--sound") + 1] == "done"
    assert runtime.registry.get("s1").state == "completed"


async def test_thin_escalation_rejected(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    thin = escalation_dict()
    del thin["recommendation"]
    resp = await send(runtime, T_ESCALATION, thin)
    assert not resp.ok
    # Warning path — NEVER surfaced as a complete escalation.
    assert [m for m in posts if isinstance(m, EscalationArrived)] == []
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("MALFORMED" in t for t in notices)
    assert runtime.slot.active is None


async def test_every_broker_message_type_is_acked(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    # Brokers deliver via client.request and fail loud without a reply —
    # every upward type must get an ok=True ack.
    runtime, _ = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    assert (
        await send(
            runtime, T_RETRACT, {"escalation_id": "e1", "reason": "r"}
        )
    ).ok
    assert (
        await send(
            runtime,
            T_PROMPT_PROPOSAL,
            {
                "proposal_id": "p1",
                "proposed_prompt": "do the task",
                "grounding_summary": "repo facts",
            },
        )
    ).ok
    assert (await send(runtime, T_BUDGET_UPDATE, {"count": 1})).ok
    assert (await send(runtime, T_COMPLETION, {"summary": "s"})).ok
    assert (
        await send(
            runtime, T_FATAL_ERROR, {"error_class": "X", "detail": "d"}
        )
    ).ok


async def test_proposal_rendered_verbatim_and_tracked(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    resp = await send(
        runtime,
        T_PROMPT_PROPOSAL,
        {
            "proposal_id": "p1",
            "proposed_prompt": "the exact proposed prompt",
            "grounding_summary": "the exact grounding summary",
        },
    )
    assert resp.ok
    arrived = [m for m in posts if isinstance(m, ProposalArrived)]
    assert len(arrived) == 1
    assert "the exact proposed prompt" in arrived[0].rendered
    assert "the exact grounding summary" in arrived[0].rendered
    assert runtime.registry.get("s1").state == "awaiting_approval"


async def test_dispatch_reports_rejection_when_session_nacks(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = NackingSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        result = await runtime.dispatch("e1", "use option B")
        assert "rejected" in result
        # A NACKed dispatch must not be treated as delivered.
        assert runtime.slot.active is not None
        assert runtime.registry.get("s1").state != "driving"
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("rejected" in t for t in notices)
    finally:
        server.close()
        await server.wait_closed()


async def test_send_prompt_reports_rejection_when_session_nacks(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    record = runtime.registry.get("s1")
    record.budget_count = 5
    runtime.registry.upsert(record)
    stub = NackingSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        result = await runtime.send_prompt("s1", "hello")
        assert "rejected" in result
        # A NACKed send must not reset the budget as if it were delivered.
        assert runtime.registry.get("s1").budget_count == 5
    finally:
        server.close()
        await server.wait_closed()


async def test_get_decision_log_warns_and_returns_empty_on_malformed_reply(
    rt: tuple[MasterRuntime, list[Any]],
    home: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runtime, _ = rt
    stub = MalformedLogSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        with caplog.at_level(logging.WARNING):
            result = await runtime.get_decision_log("s1")
        assert result == ""
        assert any("malformed" in r.message.lower() for r in caplog.records)
    finally:
        server.close()
        await server.wait_closed()


def _bind_session(runtime: MasterRuntime) -> SessionRecord:
    """Give s1 the identifiers a reassignment has to carry over."""
    record = runtime.registry.get("s1")
    record.pane_id = "w3:p2"
    record.claude_session_id = "cc-1"
    record.transcript_path = "/private/tmp/t.jsonl"
    record.approved_prompt = "the first task"
    record.budget_count = 6
    runtime.registry.upsert(record)
    return record


async def test_reassign_spawns_a_broker_that_adopts_the_live_session(
    rt: tuple[MasterRuntime, list[Any]], home: Path, spawn: RecordingSpawn
) -> None:
    runtime, _ = rt
    record = _bind_session(runtime)
    result = await runtime.reassign_session("s1", "take it from here")
    assert "reassigned" in result
    config = spawn.config()
    assert config["name"] == "s1"
    # The SAME socket path: BROKER_SOCKET was baked into the pane's
    # environment at split time, so a replacement broker bound anywhere else
    # would never see another hook event from this session.
    assert config["socket_path"] == record.socket_path
    assert config["intent"] == "take it from here"
    assert config["budget_count"] == 0
    assert config["adopt"] == {
        "pane_id": "w3:p2",
        "claude_session_id": "cc-1",
        "transcript_path": "/private/tmp/t.jsonl",
    }
    reloaded = Registry.load(home / "registry.json").get("s1")
    assert reloaded.intent == "take it from here"
    assert reloaded.approved_prompt is None  # superseded until re-approved
    assert reloaded.budget_count == 0
    assert reloaded.state == "spawning"
    assert reloaded.pid is not None  # the new broker, not the dead one


async def test_reassign_refuses_a_session_it_only_partly_knows(
    rt: tuple[MasterRuntime, list[Any]], spawn: RecordingSpawn
) -> None:
    runtime, _ = rt
    record = runtime.registry.get("s1")
    record.pane_id = "w3:p2"  # no claude session id, no transcript path
    runtime.registry.upsert(record)
    with pytest.raises(ValueError) as exc:
        await runtime.reassign_session("s1", "take it from here")
    assert "claude_session_id" in str(exc.value)
    assert "transcript_path" in str(exc.value)
    # Refusing AFTER the teardown would leave the session with no broker at
    # all — the check has to come first.
    assert spawn.argvs == []
    assert runtime.registry.get("s1").state == "driving"


async def test_reassign_refuses_while_a_broker_still_answers(
    rt: tuple[MasterRuntime, list[Any]],
    home: Path,
    spawn: RecordingSpawn,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("broker.master.runtime.STOP_WAIT_S", 0.3)
    runtime, _ = rt
    _bind_session(runtime)
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        with pytest.raises(RuntimeError) as exc:
            await runtime.reassign_session("s1", "take it from here")
        assert "refusing to reassign" in str(exc.value)
        # A unix rebind over a live listener succeeds silently; two brokers
        # would then split this pane's hook traffic between them.
        assert spawn.argvs == []
    finally:
        server.close()
        await server.wait_closed()


async def test_reactivate_relays_the_intent_and_supersedes_the_old_one(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    record = runtime.registry.get("s1")
    record.state = SessionState.COMPLETED
    record.approved_prompt = "the first task"
    record.budget_count = 4
    runtime.registry.upsert(record)
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        result = await runtime.reactivate_session("s1", "now write the docs")
        assert "reactivated" in result
        assert len(stub.envelopes) == 1
        assert stub.envelopes[0].type == T_REACTIVATE
        assert stub.envelopes[0].payload == {"intent": "now write the docs"}
        reloaded = Registry.load(home / "registry.json").get("s1")
        assert reloaded.intent == "now write the docs"
        assert reloaded.approved_prompt is None  # superseded until re-approved
        assert reloaded.budget_count == 0
        assert reloaded.state == "grounding"
    finally:
        server.close()
        await server.wait_closed()


async def test_reactivate_rejection_surfaces_the_reason_and_changes_nothing(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    record = runtime.registry.get("s1")
    record.approved_prompt = "the first task"
    record.budget_count = 4
    runtime.registry.upsert(record)
    stub = ReasoningNackSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        result = await runtime.reactivate_session("s1", "displace it")
        # The broker's own reason reaches the developer; "refused" alone would
        # not say which task is still running.
        assert stub.reason in result
        assert any(stub.reason in m.text for m in posts if isinstance(m, Notice))
        # A refused reactivation must not read as if the new task took.
        reloaded = runtime.registry.get("s1")
        assert reloaded.intent != "displace it"
        assert reloaded.approved_prompt == "the first task"
        assert reloaded.budget_count == 4
        assert reloaded.state == "driving"
    finally:
        server.close()
        await server.wait_closed()
