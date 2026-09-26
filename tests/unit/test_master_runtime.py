"""MasterRuntime harness: fake broker = protocol.client against the real
runtime server; driver.subprocess.run monkeypatched; emit = recording list."""

import asyncio
import json
import logging
import subprocess
import tempfile
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from broker import decision_log
from broker import logging_setup
from broker.decision_log import DecisionLogKind, DecisionLogRow
from broker.config import BrokerConfig
from broker.herdr import driver
from broker.master.__main__ import log_notice
from broker.master.pane_escalations import PaneEscalations
from broker.master.payload_render import (
    PANE_UNKNOWN,
    render_escalation,
    render_question_escalation,
)
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.master.runtime import MasterRuntime
from broker.master.viewmodel import (
    Attention,
    CompletionArrived,
    EscalationArrived,
    FleetUpdated,
    Notice,
    PaneEscalationArrived,
    PaneRequest,
    ProposalArrived,
    SessionStateChanged,
)
from broker.protocol import client
from broker.protocol.constants import (
    NackCode,
    PaneKind,
    SessionState,
    T_APPROVE_PROMPT,
    T_CLARIFY_ESCALATION,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DECISION_DELIVERED,
    T_DECISION_UNDELIVERED,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_ESCALATION_RETRACT,
    T_FATAL_ERROR,
    T_GET_PERMISSION_LOG,
    T_LIVE_STATUS,
    T_PANE_ESCALATION,
    T_PANE_RETRACT,
    T_PROMPT_PROPOSAL,
    T_PROMPT_UNDELIVERED,
    T_REACTIVATE,
    T_SESSION_ENDED,
    T_STATUS,
)
from broker.protocol.schemas import (
    MASTER_SOCKET_PAYLOADS,
    Envelope,
    EscalationPayload,
    PermissionEscalationPayload,
    QuestionEscalationPayload,
    Response,
)
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
        return Response(id=env.id, ok=False, payload={"error": "refused"})


class ReasoningNackSession:
    """Session-socket handler that rejects and says why."""

    reason = "session is 'driving', not 'completed'"

    async def handler(self, env: Envelope) -> Response:
        return Response(id=env.id, ok=False, payload={"error": self.reason})


class ClarifyingSession:
    """Session-socket handler answering clarify_escalation with a fixed answer."""

    answer = "it tried A"

    def __init__(self) -> None:
        self.envelopes: list[Envelope] = []

    async def handler(self, env: Envelope) -> Response:
        self.envelopes.append(env)
        return Response(id=env.id, ok=True, payload={"answer": self.answer})


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


class PermissionLogSession:
    """Session-socket handler answering get_permission_log with fixed text."""

    text = "ts=1 Bash allow — matches the stated task"

    def __init__(self) -> None:
        self.envelopes: list[Envelope] = []

    async def handler(self, env: Envelope) -> Response:
        self.envelopes.append(env)
        return Response(id=env.id, ok=True, payload={"text": self.text})


class StatusSession:
    """Session-socket handler answering status with a scripted payload."""

    def __init__(
        self,
        *,
        permission_prompt: bool,
        task_activity: str = "",
        state: str = "driving",
    ) -> None:
        self.permission_prompt = permission_prompt
        self.task_activity = task_activity
        self.state = state

    async def handler(self, env: Envelope) -> Response:
        if env.type != T_STATUS:
            return Response(
                id=env.id, ok=False, payload={"error": f"unexpected {env.type}"}
            )
        return Response(
            id=env.id,
            ok=True,
            payload={
                "state": self.state,
                "permission_prompt": self.permission_prompt,
                "task_activity": self.task_activity,
            },
        )


class ProposalStatusSession:
    """Session-socket handler reporting a pending proposal on status."""

    def __init__(self, proposal: dict[str, Any] | None) -> None:
        self.proposal = proposal

    async def handler(self, env: Envelope) -> Response:
        if env.type != T_STATUS:
            return Response(
                id=env.id, ok=False, payload={"error": f"unexpected {env.type}"}
            )
        return Response(
            id=env.id,
            ok=True,
            payload={
                "state": "awaiting_approval",
                "pane_id": "w3:p2",
                "permission_prompt": False,
                "task_activity": "",
                "pending_proposal": self.proposal,
            },
        )


def escalation_dict(esc_id: str = "e1", session: str = "s1") -> dict[str, Any]:
    return {
        "escalation_id": esc_id,
        "session_id": session,
        "task_context": "ctx-task-value",
        "disclosure": {
            "escalation_title": "title-value",
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
        },
    }


def permission_pane_dict(
    esc_id: str = "p1", session: str = "s1"
) -> dict[str, Any]:
    return {
        "kind": "permission",
        "escalation_id": esc_id,
        "session_id": session,
        "tool_name": "tool-name-value",
        "tool_input": {"command": "command-value"},
        "task_intent": "task-intent-value",
        "reason": "reason-value",
        "permission_suggestions": [
            {"type": "setMode", "mode": "mode-value"}
        ],
    }


def question_escalation_dict(
    esc_id: str = "q1", session: str = "s1"
) -> dict[str, Any]:
    return {
        "kind": "question",
        "escalation_id": esc_id,
        "session_id": session,
        "task_context": "ctx-task-value",
        "menu": (
            "## Question 1 (multi-select): question-value [header-value]\n"
            "- label-a: description-a\n- label-b: description-b"
        ),
        "first_question": "question-value",
        "reason": "reason-value",
        "analysis": escalation_dict()["disclosure"],
    }


def broker_messages(session: str) -> list[tuple[str, dict[str, Any]]]:
    """One valid message of every master-socket type, in an order a live
    broker could send them."""
    return [
        (T_ESCALATION, escalation_dict("e1", session)),
        (T_ESCALATION_RETRACT, {"escalation_id": "e1", "reason": "r"}),
        (T_PANE_ESCALATION, permission_pane_dict("p1", session)),
        (T_PANE_RETRACT, {"escalation_id": "p1", "reason": "r"}),
        (T_PROMPT_UNDELIVERED, {"detail": "d"}),
        (
            T_PROMPT_PROPOSAL,
            {
                "proposal_id": "p1",
                "proposed_prompt": "do the task",
                "grounding_summary": "repo facts",
            },
        ),
        (T_BUDGET_UPDATE, {"count": 1}),
        (T_DECISION_DELIVERED, {"escalation_id": "e1"}),
        (T_DECISION_UNDELIVERED, {"escalation_id": "e1", "still_live": False}),
        (T_LIVE_STATUS, {"state": "driving"}),
        (T_COMPLETION, {"headline": "h", "supporting": "s"}),
        (T_FATAL_ERROR, {"error_class": "X", "detail": "d"}),
        (T_SESSION_ENDED, {}),
    ]


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
    queue = EscalationQueue.load(home / "escalation-queue.json")
    panes = PaneEscalations.load(home / "pane-escalations.json")
    runtime = MasterRuntime(
        posts.append,
        registry,
        queue,
        panes,
        cfg,
        anchor_pane="%1",
        claude_json=home / "claude.json",
    )
    runtime.start()
    for _ in range(200):
        if runtime.master_socket_path.exists():
            break
        await asyncio.sleep(0.01)
    else:
        raise TimeoutError("master socket never bound")
    yield runtime, posts
    await runtime.aclose()


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


def _add_session(runtime: MasterRuntime, name: str) -> None:
    runtime.registry.upsert(
        SessionRecord(
            name=name,
            socket_path=f"/private/tmp/{name}.sock",
            cwd="/private/tmp",
            anchor_pane="%1",
            state=SessionState.DRIVING,
        )
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


async def test_second_escalation_while_active_is_protocol_violation(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    resp = await send(runtime, T_ESCALATION, escalation_dict("e2"))
    assert not resp.ok
    # Same session: the broker was told to hold one at a time and did not.
    assert resp.payload["reason_code"] == NackCode.PROTOCOL_VIOLATION
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("PROTOCOL VIOLATION" in t for t in notices)
    # The first escalation stays active; the second is never surfaced.
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "e1"
    assert (
        len([m for m in posts if isinstance(m, EscalationArrived)]) == 1
    )


async def test_second_permission_pane_from_a_session_supersedes_the_first(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (
        await send(runtime, T_PANE_ESCALATION, permission_pane_dict("p1"))
    ).ok
    # No retract of p1 arrived first: a lost retract must not wedge the slot.
    assert (
        await send(runtime, T_PANE_ESCALATION, permission_pane_dict("p2"))
    ).ok
    assert [p.escalation_id for p in runtime.panes.entries] == ["p2"]
    arrived = [m for m in posts if isinstance(m, PaneEscalationArrived)]
    assert [m.escalation_id for m in arrived] == ["p1", "p2"]
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert "permission escalation p1 from session s1 superseded by p2" in notices


async def test_permission_pane_never_waits_behind_a_decision(
    rt: tuple[MasterRuntime, list[Any]], recording_run: RecordingRun
) -> None:
    runtime, posts = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    resp = await send(
        runtime, T_PANE_ESCALATION, permission_pane_dict("p1")
    )
    # The native prompt is already blocking the session, so the developer is
    # told about it now — it takes no place in the decision queue.
    assert resp.ok
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "e1"
    arrived = [m for m in posts if isinstance(m, PaneEscalationArrived)]
    assert [m.escalation_id for m in arrived] == ["p1"]
    notify_calls = [
        c for c in recording_run.calls if c[1:3] == ["notification", "show"]
    ]
    assert len(notify_calls) == 2  # the decision head and the prompt
    view = [m for m in posts if isinstance(m, FleetUpdated)][-1].view
    assert view.queue_depth == 1
    assert view.waiting == ()
    assert view.panes == (
        PaneRequest(PaneKind.PERMISSION, "s1", "p1", "tool-name-value"),
    )


async def test_question_escalation_never_queued_and_announced_at_once(
    rt: tuple[MasterRuntime, list[Any]], recording_run: RecordingRun
) -> None:
    runtime, posts = rt
    _add_session(runtime, "s2")
    record = runtime.registry.get("s2")
    record.pane_id = "w3:p4"
    runtime.registry.upsert(record)
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    payload = question_escalation_dict("q1", "s2")
    assert (await send(runtime, T_PANE_ESCALATION, payload, session="s2")).ok
    # The menu is already blocking s2, so the developer hears of it now; it
    # never takes a place in the decision queue or moves its head.
    assert runtime.queue.depth == 1
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "e1"
    assert runtime.registry.get("s2").state == SessionState.DRIVING
    arrived = [m for m in posts if isinstance(m, PaneEscalationArrived)]
    assert [(m.kind, m.escalation_id) for m in arrived] == [
        (PaneKind.QUESTION, "q1")
    ]
    rendered = arrived[0].rendered
    assert rendered == render_question_escalation(
        QuestionEscalationPayload.model_validate(payload), "w3:p4"
    )
    for value in _leaf_values(payload):
        if value not in ("question", "s2"):
            assert value in rendered
    assert "(multi-select)" in rendered
    assert "cannot be answered here" in rendered
    notify_calls = [
        c for c in recording_run.calls if c[1:3] == ["notification", "show"]
    ]
    assert len(notify_calls) == 2  # the decision head and the menu
    view = [m for m in posts if isinstance(m, FleetUpdated)][-1].view
    assert view.queue_depth == 1
    assert view.head is not None and view.head.escalation_id == "e1"
    assert view.panes == (
        PaneRequest(PaneKind.QUESTION, "s2", "q1", "question-value"),
    )


async def test_unreadable_menu_still_surfaces(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    payload = question_escalation_dict()
    payload["menu"] = ""
    payload["first_question"] = ""
    del payload["analysis"]
    assert (await send(runtime, T_PANE_ESCALATION, payload)).ok
    arrived = [m for m in posts if isinstance(m, PaneEscalationArrived)]
    assert "the menu could not be read" in arrived[0].rendered
    assert "reason-value" in arrived[0].rendered


async def test_dispatch_and_clarify_refuse_a_question_naming_the_pane(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    record = runtime.registry.get("s1")
    record.pane_id = "w3:p2"
    runtime.registry.upsert(record)
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (
            await send(runtime, T_PANE_ESCALATION, question_escalation_dict("q1"))
        ).ok
        dispatched = await runtime.dispatch("q1", "label-a")
        clarified = await runtime.clarify_escalation("q1", "why?")
        for result in (dispatched, clarified):
            assert "an AskUserQuestion menu" in result
            assert "pane w3:p2" in result
        assert dispatched.startswith("decision NOT dispatched")
        assert clarified.startswith("question NOT sent")
        assert stub.envelopes == []
        assert runtime.panes.find("q1") is not None
    finally:
        server.close()
        await server.wait_closed()


async def test_prompt_undelivered_becomes_a_notice(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    resp = await send(
        runtime,
        T_PROMPT_UNDELIVERED,
        {"detail": "an AskUserQuestion menu is open in pane w3:p2"},
    )
    assert resp.ok
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any(
        "s1" in t and "an AskUserQuestion menu is open in pane w3:p2" in t
        for t in notices
    )


async def test_malformed_permission_pane_nacked_never_surfaced(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    thin = permission_pane_dict()
    del thin["reason"]
    resp = await send(runtime, T_PANE_ESCALATION, thin)
    assert not resp.ok
    assert resp.payload["reason_code"] == NackCode.MALFORMED
    # A prompt the developer cannot act on is worse than none at all.
    assert [m for m in posts if isinstance(m, PaneEscalationArrived)] == []
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("MALFORMED" in t for t in notices)
    assert runtime.panes.entries == ()


async def test_escalation_from_an_unknown_session_never_enters_the_queue(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    """A queued entry for a session the master cannot route to would surface
    eventually and block every session it can."""
    runtime, posts = rt
    resp = await send(
        runtime, T_ESCALATION, escalation_dict("e1", "ghost"), session="ghost"
    )
    assert not resp.ok
    assert resp.payload["reason_code"] == NackCode.UNKNOWN_SESSION
    resp = await send(
        runtime,
        T_PANE_ESCALATION,
        permission_pane_dict("p1", "ghost"),
        session="ghost",
    )
    assert not resp.ok
    assert resp.payload["reason_code"] == NackCode.UNKNOWN_SESSION
    assert runtime.queue.active is None
    assert runtime.panes.entries == ()
    assert [m for m in posts if isinstance(m, EscalationArrived)] == []
    assert [m for m in posts if isinstance(m, PaneEscalationArrived)] == []


async def test_dispatch_refuses_permission_pane(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert (
            await send(
                runtime, T_PANE_ESCALATION, permission_pane_dict("p1")
            )
        ).ok
        result = await runtime.dispatch("p1", "yes, go ahead")
        # Refused as a permission prompt, not as a stale decision: the
        # developer is told where the answer belongs.
        assert "NOT dispatched" in result
        assert "permission prompt" in result
        # The registry has no pane for s1 here; the refusal still has to say
        # where the answer belongs rather than go silent.
        assert PANE_UNKNOWN in result
        assert stub.envelopes == []  # nothing reached the session
        # Nothing was resolved either.
        assert runtime.panes.find("p1") is not None
        assert runtime.queue.active is not None
        assert runtime.queue.active.escalation_id == "e1"
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("NOT dispatched" in t for t in notices)
    finally:
        server.close()
        await server.wait_closed()


async def test_reassign_retracts_that_sessions_permission_pane(
    rt: tuple[MasterRuntime, list[Any]], home: Path, spawn: RecordingSpawn
) -> None:
    runtime, posts = rt
    _bind_session(runtime)
    # Another session's prompt must survive: reassigning s1 says nothing
    # about it.
    other = PermissionEscalationPayload.model_validate(
        permission_pane_dict("p9", "s9")
    )
    runtime.panes.accept(other)
    assert (
        await send(
            runtime, T_PANE_ESCALATION, permission_pane_dict("p1")
        )
    ).ok
    await runtime.reassign_session("s1", "take it from here")
    # Its permission module died with the broker, so nothing else would ever
    # clear it.
    assert runtime.panes.entries == (other,)
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("p1" in t and "retracted" in t for t in notices)


async def test_stop_session_retracts_that_sessions_permission_pane(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    assert (
        await send(
            runtime, T_PANE_ESCALATION, permission_pane_dict("p1")
        )
    ).ok
    assert (
        await send(runtime, T_PANE_ESCALATION, question_escalation_dict("q1"))
    ).ok
    await runtime.stop_session("s1")
    # Both raisers died with the broker and the native prompts are still on
    # screen, so no retraction is ever coming.
    assert runtime.panes.entries == ()
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("p1" in t and "retracted" in t for t in notices)
    assert any("q1" in t and "retracted" in t for t in notices)
    # The decision escalation from the same session is kept on purpose.
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "e1"


async def test_get_permission_log_round_trip(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    stub = PermissionLogSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert await runtime.get_permission_log("s1") == stub.text
        assert [e.type for e in stub.envelopes] == [T_GET_PERMISSION_LOG]
    finally:
        server.close()
        await server.wait_closed()


async def test_list_sessions_reports_permission_prompt_flag(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    assert (
        await send(runtime, T_LIVE_STATUS, {"state": "driving", "pane_id": "w3:p2"})
    ).ok
    stub = StatusSession(permission_prompt=True)
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        listing = await runtime.list_sessions()
        assert "s1" in listing
        assert "sitting on a permission prompt" in listing
        assert "w3:p2" in listing
        # The flag is read on demand and must NOT reach the summary the master
        # carries into every turn.
        assert "permission prompt" not in runtime.registry_summary()
    finally:
        server.close()
        await server.wait_closed()


async def test_list_sessions_degrades_when_a_session_is_unreachable(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    """One dead session costs a line of the listing, never the whole listing."""
    runtime, _ = rt
    listing = await runtime.list_sessions()
    assert "s1" in listing
    assert "unreachable" in listing


async def test_retract_clears_head_and_informs(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    resp = await send(
        runtime,
        T_ESCALATION_RETRACT,
        {"escalation_id": "e1", "reason": "resolved in pane"},
    )
    assert resp.ok
    assert runtime.queue.active is None
    # Raising it set ESCALATED; leaving it there outlives the escalation and
    # every later read of the registry is wrong about the session.
    assert runtime.registry.get("s1").state == SessionState.DRIVING
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("resolved in pane" in t for t in notices)


async def test_permission_retract_leaves_session_state_alone(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    # A permission escalation never set the state, so withdrawing it must not
    # claim the session is driving when its broker knows otherwise — its own
    # escalation may still be live.
    runtime, posts = rt
    assert (
        await send(
            runtime, T_PANE_ESCALATION, permission_pane_dict("p1")
        )
    ).ok
    record = runtime.registry.get("s1")
    record.state = SessionState.ESCALATED
    runtime.registry.upsert(record)
    assert (
        await send(
            runtime,
            T_PANE_RETRACT,
            {"escalation_id": "p1", "reason": "answered"},
        )
    ).ok
    assert runtime.registry.get("s1").state == SessionState.ESCALATED
    assert runtime.panes.entries == ()
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("p1" in t and "retracted: answered" in t for t in notices)


async def test_queued_escalation_surfaces_after_resolve(
    rt: tuple[MasterRuntime, list[Any]],
    home: Path,
    recording_run: RecordingRun,
) -> None:
    runtime, posts = rt
    _add_session(runtime, "s2")
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)

    def surfaced() -> list[str]:
        return [m.escalation_id for m in posts if isinstance(m, EscalationArrived)]

    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert (
            await send(runtime, T_ESCALATION, escalation_dict("e2", "s2"), "s2")
        ).ok
        assert surfaced() == ["e1"]
        result = await runtime.dispatch("e1", "use option B")
        assert "dispatched" in result
        # Resolution waits for confirmed delivery: e1 is still the head and
        # nothing new surfaces until the broker confirms it reached the pane.
        assert surfaced() == ["e1"]
        assert runtime.queue.active is not None
        assert runtime.queue.active.escalation_id == "e1"
        assert (await send(runtime, T_DECISION_DELIVERED, {"escalation_id": "e1"})).ok
        assert surfaced() == ["e1", "e2"]
        assert runtime.queue.active is not None
        assert runtime.queue.active.escalation_id == "e2"
        notify_calls = [
            c for c in recording_run.calls if c[1:3] == ["notification", "show"]
        ]
        assert len(notify_calls) == 2  # one per surfacing, none at accept
    finally:
        server.close()
        await server.wait_closed()


async def test_queued_escalation_surfaces_after_retract(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    _add_session(runtime, "s2")
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    assert (await send(runtime, T_ESCALATION, escalation_dict("e2", "s2"), "s2")).ok
    assert (
        await send(
            runtime,
            T_ESCALATION_RETRACT,
            {"escalation_id": "e1", "reason": "answered"},
        )
    ).ok
    arrived = [m.escalation_id for m in posts if isinstance(m, EscalationArrived)]
    assert arrived == ["e1", "e2"]
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "e2"


async def test_retracted_queued_escalation_is_never_surfaced(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    _add_session(runtime, "s2")
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    assert (await send(runtime, T_ESCALATION, escalation_dict("e2", "s2"), "s2")).ok
    assert (
        await send(
            runtime,
            T_ESCALATION_RETRACT,
            {"escalation_id": "e2", "reason": "resolved in pane"},
            "s2",
        )
    ).ok
    assert (
        await send(
            runtime,
            T_ESCALATION_RETRACT,
            {"escalation_id": "e1", "reason": "answered"},
        )
    ).ok
    # The queued escalation was withdrawn before its turn; announcing it would
    # hand the developer a decision nobody is waiting on.
    arrived = [m.escalation_id for m in posts if isinstance(m, EscalationArrived)]
    assert arrived == ["e1"]
    assert runtime.queue.active is None


async def test_notification_fires_at_surface_time_not_accept(
    rt: tuple[MasterRuntime, list[Any]],
    home: Path,
    recording_run: RecordingRun,
) -> None:
    runtime, _ = rt
    _add_session(runtime, "s2")
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)

    def notify_count() -> int:
        return len(
            [
                c
                for c in recording_run.calls
                if c[1:3] == ["notification", "show"]
            ]
        )

    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert (
            await send(runtime, T_ESCALATION, escalation_dict("e2", "s2"), "s2")
        ).ok
        assert notify_count() == 1  # only the surfaced head is announced
        await runtime.dispatch("e1", "use option B")
        assert notify_count() == 1  # nothing new surfaces before delivery lands
        assert (await send(runtime, T_DECISION_DELIVERED, {"escalation_id": "e1"})).ok
        assert notify_count() == 2  # the next head announces when it surfaces
    finally:
        server.close()
        await server.wait_closed()


async def test_startup_resurfaces_the_persisted_head_and_open_prompts(
    home: Path, recording_run: RecordingRun
) -> None:
    queue_path = home / "escalation-queue.json"
    panes_path = home / "pane-escalations.json"
    seeded = EscalationQueue.load(queue_path)
    seeded.accept(EscalationPayload.model_validate(escalation_dict("e1")))
    seeded.accept(EscalationPayload.model_validate(escalation_dict("e2", "s2")))
    PaneEscalations.load(panes_path).accept(
        PermissionEscalationPayload.model_validate(
            permission_pane_dict("p1")
        )
    )
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    for name in ("s1", "s2"):
        registry.upsert(
            SessionRecord(
                name=name,
                socket_path=str(home / "s" / f"{name}.sock"),
                cwd="/private/tmp",
                anchor_pane="%1",
                state=SessionState.DRIVING,
            )
        )
    posts: list[Any] = []
    runtime = MasterRuntime(
        posts.append,
        registry,
        EscalationQueue.load(queue_path),
        PaneEscalations.load(panes_path),
        cfg,
        anchor_pane="%1",
        claude_json=home / "claude.json",
    )
    runtime.start()
    for _ in range(200):
        if runtime.master_socket_path.exists():
            break
        await asyncio.sleep(0.01)
    else:
        raise TimeoutError("master socket never bound")
    await runtime.aclose()
    arrived = [m for m in posts if isinstance(m, EscalationArrived)]
    assert len(arrived) == 1
    assert arrived[0].escalation_id == "e1"
    assert arrived[0].rendered == render_escalation(
        EscalationPayload.model_validate(escalation_dict("e1"))
    )
    # An open prompt persisted across the restart is announced again: the
    # native prompt is still blocking its session.
    prompts = [m for m in posts if isinstance(m, PaneEscalationArrived)]
    assert [m.escalation_id for m in prompts] == ["p1"]
    # The waiting decision stays unannounced; the fleet view carries it.
    fleets = [m for m in posts if isinstance(m, FleetUpdated)]
    assert fleets[0].view.queue_depth == 2
    assert fleets[0].view.waiting == ("s2",)
    assert [p.escalation_id for p in fleets[0].view.panes] == ["p1"]


async def test_probe_of_a_settled_session_keeps_it_settled(home: Path) -> None:
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
    # A completed broker keeps serving and still reports its last turn's
    # task activity.
    stub = StatusSession(
        permission_prompt=False, task_activity="wrapping up", state="completed"
    )
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    posts: list[Any] = []
    runtime = MasterRuntime(
        posts.append,
        registry,
        EscalationQueue.load(home / "escalation-queue.json"),
        PaneEscalations.load(home / "pane-escalations.json"),
        cfg,
        anchor_pane="%1",
        claude_json=home / "claude.json",
    )
    try:
        await runtime.probe_status("s1")
    finally:
        server.close()
    assert registry.get("s1").state is SessionState.COMPLETED
    assert [
        (m.session_id, m.state) for m in posts if isinstance(m, SessionStateChanged)
    ] == [("s1", SessionState.COMPLETED)]
    row = runtime.build_fleet_view().rows[0]
    assert row.task_activity == ""


async def test_repopulate_from_brokers_fills_task_activity_at_startup(
    home: Path,
) -> None:
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
    stub = StatusSession(
        permission_prompt=False, task_activity="reviewing the login flow"
    )
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    posts: list[Any] = []
    runtime = MasterRuntime(
        posts.append,
        registry,
        EscalationQueue.load(home / "escalation-queue.json"),
        PaneEscalations.load(home / "pane-escalations.json"),
        cfg,
        anchor_pane="%1",
        claude_json=home / "claude.json",
    )
    try:
        runtime.start()

        def repopulated() -> bool:
            fleets = [m for m in posts if isinstance(m, FleetUpdated)]
            return bool(fleets) and any(
                row.task_activity == "reviewing the login flow"
                for row in fleets[-1].view.rows
            )

        for _ in range(200):
            if repopulated():
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError(
                "startup never repopulated task_activity from the broker"
            )
        await runtime.aclose()
    finally:
        server.close()
        await server.wait_closed()


async def test_repopulate_from_brokers_recovers_pending_proposal(
    home: Path,
) -> None:
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    registry.upsert(
        SessionRecord(
            name="s1",
            socket_path=str(home / "s" / "s1.sock"),
            cwd="/private/tmp",
            anchor_pane="%1",
            state=SessionState.AWAITING_APPROVAL,
        )
    )
    # A broker still awaiting approval hands its proposal back on the probe, so
    # a restarted master recovers what its memory-only map had lost.
    stub = ProposalStatusSession(
        {
            "proposal_id": "pr1",
            "proposed_prompt": "add a health endpoint",
            "grounding_summary": "grounded against app.main",
        }
    )
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    posts: list[Any] = []
    runtime = MasterRuntime(
        posts.append,
        registry,
        EscalationQueue.load(home / "escalation-queue.json"),
        PaneEscalations.load(home / "pane-escalations.json"),
        cfg,
        anchor_pane="%1",
        claude_json=home / "claude.json",
    )
    try:
        runtime.start()
        for _ in range(200):
            if "pr1" in runtime.proposals:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("startup never recovered the pending proposal")
        await runtime.aclose()
    finally:
        server.close()
        await server.wait_closed()

    assert runtime.proposals["pr1"].session_id == "s1"
    arrived = [m for m in posts if isinstance(m, ProposalArrived)]
    assert [(m.session_id, m.proposal_id) for m in arrived] == [("s1", "pr1")]
    assert "add a health endpoint" in arrived[0].rendered


async def test_repopulate_from_brokers_registers_nothing_when_no_proposal(
    home: Path,
) -> None:
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
    # No proposal pending: the probe defaults pending_proposal to None and the
    # guard registers nothing rather than crashing on the absent payload.
    stub = StatusSession(permission_prompt=False, task_activity="working")
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    posts: list[Any] = []
    runtime = MasterRuntime(
        posts.append,
        registry,
        EscalationQueue.load(home / "escalation-queue.json"),
        PaneEscalations.load(home / "pane-escalations.json"),
        cfg,
        anchor_pane="%1",
        claude_json=home / "claude.json",
    )
    try:
        runtime.start()
        for _ in range(200):
            if any(
                isinstance(m, FleetUpdated)
                and any(row.task_activity == "working" for row in m.view.rows)
                for m in posts
            ):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("startup never probed the broker")
        await runtime.aclose()
    finally:
        server.close()
        await server.wait_closed()

    assert runtime.proposals == {}
    assert [m for m in posts if isinstance(m, ProposalArrived)] == []


async def test_fleet_view_tracks_the_queue_and_open_prompts_separately(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)

    def state() -> tuple[int, tuple[str, ...], tuple[str, ...]]:
        view = [m for m in posts if isinstance(m, FleetUpdated)][-1].view
        prompts = tuple(p.escalation_id for p in view.panes)
        return (view.queue_depth, view.waiting, prompts)

    try:
        assert state() == (0, (), ())  # start() announces the loaded stores
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert state() == (1, (), ())
        assert (
            await send(
                runtime, T_PANE_ESCALATION, permission_pane_dict("p1")
            )
        ).ok
        assert state() == (1, (), ("p1",))  # counted apart from the queue
        await runtime.dispatch("e1", "use option B")
        assert state() == (1, (), ("p1",))  # unresolved until delivery lands
        assert (await send(runtime, T_DECISION_DELIVERED, {"escalation_id": "e1"})).ok
        assert state() == (0, (), ("p1",))
        assert (
            await send(
                runtime,
                T_PANE_RETRACT,
                {"escalation_id": "p1", "reason": "answered"},
            )
        ).ok
        assert state() == (0, (), ())
    finally:
        server.close()
        await server.wait_closed()


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
        # The ACK only accepts it for processing: e1 stays the live head until
        # the broker confirms delivery, then it resolves.
        assert runtime.queue.active is not None
        assert runtime.queue.active.escalation_id == "e1"
        assert (await send(runtime, T_DECISION_DELIVERED, {"escalation_id": "e1"})).ok
        assert runtime.queue.active is None
        # Single authority: the master never invents DRIVING — the state stays
        # until the broker reports its own transition.
        assert runtime.registry.get("s1").state == "escalated"
        assert (
            await send(runtime, T_LIVE_STATUS, {"state": "driving"})
        ).ok
        assert runtime.registry.get("s1").state == "driving"
    finally:
        server.close()
        await server.wait_closed()


async def test_undelivered_still_live_keeps_escalation_for_redecide(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert "dispatched" in await runtime.dispatch("e1", "use option B")
        assert (
            await send(
                runtime,
                T_DECISION_UNDELIVERED,
                {
                    "escalation_id": "e1",
                    "detail": "SubmitTimeout: pane wedged",
                    "still_live": True,
                },
            )
        ).ok
        # A failed pane write never resolved e1: it is reported loudly and
        # stays the live head so the developer can dispatch again.
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("did NOT reach" in t and "e1" in t for t in notices)
        assert runtime.queue.active is not None
        assert runtime.queue.active.escalation_id == "e1"
        # The in-flight lock cleared, so a re-decide dispatches rather than
        # bouncing off a decision that is supposedly still being delivered.
        assert "dispatched" in await runtime.dispatch("e1", "retry")
    finally:
        server.close()
        await server.wait_closed()


async def test_undelivered_stale_retracts_orphaned_entry(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert "dispatched" in await runtime.dispatch("e1", "use option B")
        assert (
            await send(
                runtime,
                T_DECISION_UNDELIVERED,
                {
                    "escalation_id": "e1",
                    "detail": "session had already moved past this escalation",
                    "still_live": False,
                },
            )
        ).ok
        # The broker no longer holds e1 (e.g. answered before a master
        # restart): the orphaned entry is dropped so the queue cannot wedge.
        assert runtime.queue.active is None
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("cleared" in t and "e1" in t for t in notices)
        # The in-flight lock cleared with it: a fresh escalation dispatches.
        assert (await send(runtime, T_ESCALATION, escalation_dict("e2"))).ok
        assert "dispatched" in await runtime.dispatch("e2", "go")
    finally:
        server.close()
        await server.wait_closed()


async def test_stop_session_clears_inflight_for_its_own_head(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert "dispatched" in await runtime.dispatch("e1", "go")
        # Stopped mid-delivery: no delivered/undelivered reply ever clears the
        # marker, so the stop must, or a re-dispatch is wrongly refused.
        await runtime.stop_session("s1")
        assert (
            "already being delivered"
            not in await runtime.dispatch("e1", "retry")
        )
    finally:
        server.close()
        await server.wait_closed()


async def test_second_dispatch_refused_while_inflight(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert "dispatched" in await runtime.dispatch("e1", "first")
        # A decision is already on its way to the pane; a second would
        # double-submit the same escalation.
        result = await runtime.dispatch("e1", "second")
        assert "already being delivered" in result
        assert len(stub.envelopes) == 1  # the second never reached the session
    finally:
        server.close()
        await server.wait_closed()


async def test_fatal_error_retracts_a_live_escalation(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert (
            await send(
                runtime, T_PANE_ESCALATION, permission_pane_dict("p1")
            )
        ).ok
        assert (
            await send(runtime, T_PANE_ESCALATION, question_escalation_dict("q1"))
        ).ok
        # A session that dies with a live escalation would otherwise wedge the
        # head forever, undispatchable to a dead session, and its open prompts
        # would stay shown with no raiser left to retract them.
        assert (
            await send(
                runtime,
                T_FATAL_ERROR,
                {"error_class": "SubmitTimeout", "detail": "pane unresponsive"},
            )
        ).ok
        assert runtime.registry.get("s1").state == "error"
        assert runtime.queue.active is None
        assert runtime.panes.entries == ()
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("e1" in t and "retracted" in t for t in notices)
        assert any("p1" in t and "retracted" in t for t in notices)
        assert any("q1" in t and "retracted" in t for t in notices)
    finally:
        server.close()
        await server.wait_closed()


async def test_clarify_escalation_relays_answer(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = ClarifyingSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        result = await runtime.clarify_escalation("e1", "what did it try?")
        assert len(stub.envelopes) == 1
        env = stub.envelopes[0]
        assert env.type == T_CLARIFY_ESCALATION
        assert env.payload == {"escalation_id": "e1", "question": "what did it try?"}
        # The answer reaches the developer verbatim; the tool loop gets only
        # an acknowledgement it cannot paraphrase from.
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("it tried A" in t and "e1" in t for t in notices)
        assert "it tried A" not in result
        assert "shown to the developer" in result
        active = runtime.queue.active
        assert active is not None and active.escalation_id == "e1"
    finally:
        server.close()
        await server.wait_closed()


async def test_clarify_escalation_wrong_id(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = ClarifyingSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        result = await runtime.clarify_escalation("ghost", "q")
        assert "NOT sent" in result
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        result = await runtime.clarify_escalation("e2", "q")
        assert "NOT sent" in result
        assert stub.envelopes == []
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("NOT sent" in t for t in notices)
    finally:
        server.close()
        await server.wait_closed()


async def test_clarify_escalation_permission_refused(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = ClarifyingSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert (
            await send(
                runtime, T_PANE_ESCALATION, permission_pane_dict("p1")
            )
        ).ok
        result = await runtime.clarify_escalation("p1", "q")
        assert "NOT sent" in result
        assert "permission prompt" in result
        assert PANE_UNKNOWN in result
        assert stub.envelopes == []
        assert runtime.panes.find("p1") is not None
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("NOT sent" in t for t in notices)
    finally:
        server.close()
        await server.wait_closed()


async def test_clarify_escalation_broker_nack(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = ReasoningNackSession()
    stub.reason = "escalation resolved in the pane"
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        result = await runtime.clarify_escalation("e1", "q")
        assert result.startswith("no clarification from session s1")
        assert result.endswith(": escalation resolved in the pane")
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert result in notices
        # The master never resolves on the broker's behalf.
        assert runtime.queue.active is not None
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
    resp = await send(
        runtime,
        T_COMPLETION,
        {"headline": "Recovered the session", "supporting": "budget survived"},
    )
    assert resp.ok
    arrived = [m for m in posts if isinstance(m, CompletionArrived)]
    assert len(arrived) == 1
    assert arrived[0].headline == "Recovered the session"  # verbatim
    assert arrived[0].supporting == "budget survived"
    notify_calls = [
        c for c in recording_run.calls if c[1:3] == ["notification", "show"]
    ]
    assert len(notify_calls) == 1
    argv = notify_calls[0]
    assert argv[argv.index("--sound") + 1] == "done"
    assert runtime.registry.get("s1").state == "completed"


async def test_build_session_outcome_reads_log_off_disk(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    # No broker socket is ever contacted: the log file and registry suffice,
    # which is what an error/stopped session (dead broker) relies on.
    runtime, _ = rt
    log = runtime.paths.session_decisions("s1")
    decision_log.append(
        log,
        DecisionLogRow(
            kind=DecisionLogKind.ANSWERED,
            reasoning="r",
            detail="a",
            task_summary="Did A",
        ),
    )
    decision_log.append(
        log,
        DecisionLogRow(
            kind=DecisionLogKind.COMPLETED,
            reasoning="r",
            detail="",
            task_summary="Wrapped up",
            headline="Recovered the session",
            supporting="budget survived",
        ),
    )
    assert (
        await send(
            runtime,
            T_COMPLETION,
            {"headline": "Recovered the session", "supporting": "budget survived"},
        )
    ).ok
    outcome = runtime.build_session_outcome("s1")
    assert outcome.status == "completed"
    assert outcome.headline == "Recovered the session"
    assert [ev.label for ev in outcome.history] == ["Did A", "Task completed"]
    with pytest.raises(KeyError):
        runtime.build_session_outcome("ghost")


async def test_session_ended_removes_from_fleet(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    # s1 is seeded driving and visible in the summary the master carries.
    assert "s1" in runtime.registry_summary()
    resp = await send(runtime, T_SESSION_ENDED, {})
    assert resp.ok
    # Removed everywhere a router looks: the summary, the fleet, and the
    # registry itself — so it can never be handed a new task or a decision.
    assert "s1" not in runtime.registry.records
    assert runtime.registry_summary() == "(no sessions)"
    last_view = [m for m in posts if isinstance(m, FleetUpdated)][-1].view
    assert not any(row.session_id == "s1" for row in last_view.rows)
    # The removal is durable, and a late live-status push cannot resurrect it.
    assert "s1" not in Registry.load(home / "registry.json").records
    resp = await send(runtime, T_LIVE_STATUS, {"state": "driving"})
    assert resp.payload["reason_code"] == NackCode.UNKNOWN_SESSION
    assert "s1" not in runtime.registry.records


async def test_session_ended_retracts_its_stranded_escalation(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    # s1 is the surfaced FIFO head, waiting on a decision the developer never
    # gave before running /exit; a bystander waits behind it and must survive.
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1", "s1"))).ok
    assert (
        await send(
            runtime, T_PANE_ESCALATION, permission_pane_dict("p1")
        )
    ).ok
    assert (
        await send(runtime, T_PANE_ESCALATION, question_escalation_dict("q1"))
    ).ok
    other = EscalationPayload.model_validate(escalation_dict("e9", "s9"))
    runtime.queue.accept(other)
    assert runtime.queue.active is not other
    resp = await send(runtime, T_SESSION_ENDED, {})
    assert resp.ok
    # The broker exits without withdrawing either, so ending must retract
    # both — otherwise the head wedges the queue forever, undispatchable to a
    # gone session, and the open prompt stays shown for a session that is gone.
    assert runtime.queue.depth == 1
    assert runtime.queue.active is other
    assert runtime.panes.entries == ()
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("e1" in t and "retracted" in t for t in notices)
    assert any("p1" in t and "retracted" in t for t in notices)
    assert any("q1" in t and "retracted" in t for t in notices)


async def test_thin_escalation_rejected(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    thin = escalation_dict()
    del thin["disclosure"]["recommendation"]
    resp = await send(runtime, T_ESCALATION, thin)
    assert not resp.ok
    # Warning path — NEVER surfaced as a complete escalation.
    assert [m for m in posts if isinstance(m, EscalationArrived)] == []
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("MALFORMED" in t for t in notices)
    assert runtime.queue.active is None


async def test_every_broker_message_type_is_acked(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    # Brokers deliver via client.request and fail loud without a reply —
    # every upward type must get an ok=True ack.
    runtime, _ = rt
    messages = broker_messages("s1")
    assert {t for t, _ in messages} == MASTER_SOCKET_PAYLOADS.keys()
    for msg_type, payload in messages:
        assert (await send(runtime, msg_type, payload)).ok, msg_type


async def test_proposal_awaits_approval_and_badges_on_arrival(
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
    assert len([m for m in posts if isinstance(m, ProposalArrived)]) == 1
    assert list(runtime.proposals) == ["p1"]
    assert runtime.registry.get("s1").state == SessionState.AWAITING_APPROVAL
    # The badge reaches the sidebar on this push, not on some later unrelated
    # one — the developer needs to see it the moment it arrives.
    row = [m for m in posts if isinstance(m, FleetUpdated)][-1].view.rows[0]
    assert row.badges == (Attention.PROPOSAL,)


async def test_second_proposal_from_a_session_replaces_its_first(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, _ = rt
    assert (
        await send(
            runtime,
            T_PROMPT_PROPOSAL,
            {
                "proposal_id": "p1",
                "proposed_prompt": "first attempt",
                "grounding_summary": "g1",
            },
        )
    ).ok
    assert (
        await send(
            runtime,
            T_PROMPT_PROPOSAL,
            {
                "proposal_id": "p2",
                "proposed_prompt": "second attempt",
                "grounding_summary": "g2",
            },
        )
    ).ok
    assert list(runtime.proposals) == ["p2"]
    row = [row for row in runtime.build_fleet_view().rows if row.session_id == "s1"][
        0
    ]
    assert row.badges == (Attention.PROPOSAL,)


async def test_approve_prompt_stores_the_title_and_it_reaches_the_fleet_row(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    assert (
        await send(
            runtime,
            T_PROMPT_PROPOSAL,
            {
                "proposal_id": "p1",
                "proposed_prompt": "the proposed prompt",
                "grounding_summary": "grounding",
            },
        )
    ).ok
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        result = await runtime.approve_prompt(
            "p1", "the approved prompt", "fix the login bug"
        )
        assert "approved" in result
        assert len(stub.envelopes) == 1
        assert stub.envelopes[0].type == T_APPROVE_PROMPT
        assert stub.envelopes[0].payload == {
            "proposal_id": "p1",
            "prompt": "the approved prompt",
        }
        assert runtime.registry.get("s1").title == "fix the login bug"
        row = [m for m in posts if isinstance(m, FleetUpdated)][-1].view.rows[0]
        assert row.title == "fix the login bug"
    finally:
        server.close()
        await server.wait_closed()


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
        assert runtime.queue.active is not None
        assert runtime.registry.get("s1").state != "driving"
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("rejected" in t for t in notices)
    finally:
        server.close()
        await server.wait_closed()


async def test_send_prompt_leaves_the_budget_to_the_broker(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    record = runtime.registry.get("s1")
    record.budget_count = 5
    runtime.registry.upsert(record)
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        result = await runtime.send_prompt("s1", "hello")
        assert result == "prompt accepted by session s1"
        # Acceptance is not delivery: only the broker's budget update resets it.
        assert runtime.registry.get("s1").budget_count == 5
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


async def test_get_decision_log_raises_on_malformed_reply(
    rt: tuple[MasterRuntime, list[Any]],
    home: Path,
) -> None:
    """A malformed log reply must fail loud, never read as an empty log."""
    runtime, _ = rt
    stub = MalformedLogSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        with pytest.raises(ValidationError):
            await runtime.get_decision_log("s1")
    finally:
        server.close()
        await server.wait_closed()


async def test_spawn_session_seeds_trust_and_registers(
    home: Path, spawn: RecordingSpawn
) -> None:
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    queue = EscalationQueue.load(home / "escalation-queue.json")
    panes = PaneEscalations.load(home / "pane-escalations.json")
    runtime = MasterRuntime(
        lambda _event: None,
        registry,
        queue,
        panes,
        cfg,
        anchor_pane="%1",
        claude_json=home / "claude.json",
    )
    result = await runtime.spawn_session("do the thing", str(home))
    assert "spawned session" in result
    assert len(spawn.argvs) == 1
    assert registry.get("s1").cwd == str(home)
    assert registry.get("s1").intent == "do the thing"
    trusted = cast(
        dict[str, Any],
        json.loads((home / "claude.json").read_text(encoding="utf-8")),
    )
    assert trusted["projects"][str(home)]["hasTrustDialogAccepted"] is True


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
    # -I isolates the subprocess from the developer's PYTHONPATH and cwd.
    assert spawn.argvs[-1][1:4] == ("-I", "-m", "broker.session")
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


async def test_reassign_does_not_resurrect_a_session_ended_during_its_stop_wait(
    rt: tuple[MasterRuntime, list[Any]], spawn: RecordingSpawn
) -> None:
    runtime, _ = rt
    record = _bind_session(runtime)
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingShutdown:
        async def handler(self, env: Envelope) -> Response:
            entered.set()
            await release.wait()
            return Response(id=env.id, ok=True, payload={})

    stub = BlockingShutdown()
    server = await serve_unix(Path(record.socket_path), stub.handler)
    try:
        reassign = asyncio.create_task(runtime.reassign_session("s1", "new task"))
        await asyncio.wait_for(entered.wait(), timeout=2)
        # The broker reports its own end while reassign is still waiting on
        # the shutdown reply — the session must not come back from that.
        assert (await send(runtime, T_SESSION_ENDED, {})).ok
        assert "s1" not in runtime.registry.records
        release.set()
        with pytest.raises(KeyError):
            await reassign
    finally:
        server.close()
        await server.wait_closed()
    assert "s1" not in runtime.registry.records


class SpawnWatchingSettings(RecordingSpawn):
    """Spawn stand-in that notes whether the rules file was already on disk."""

    def __init__(self, settings_path: Path) -> None:
        super().__init__()
        self.settings_path = settings_path
        self.existed_at_spawn: list[bool] = []

    async def __call__(self, *argv: str) -> FakeProcess:
        self.existed_at_spawn.append(self.settings_path.exists())
        return await super().__call__(*argv)


async def test_spawn_writes_the_session_rules_and_passes_them_on(
    rt: tuple[MasterRuntime, list[Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _ = rt
    settings_path = runtime.paths.session_claude_settings("s1")
    spawn = SpawnWatchingSettings(settings_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    _bind_session(runtime)
    await runtime.reassign_session("s1", "take it from here")
    # A broker started before its rules exist runs the session unconfigured.
    assert spawn.existed_at_spawn == [True]
    written = cast(
        dict[str, Any], json.loads(settings_path.read_text(encoding="utf-8"))
    )
    assert written["permissions"] == runtime.cfg.permission_rules.model_dump()
    config = spawn.config()
    assert config["claude_settings_path"] == str(settings_path)
    assert config["classifier"] == runtime.cfg.classifier.model_dump()


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


async def test_attach_spawns_a_resuming_broker(
    rt: tuple[MasterRuntime, list[Any]], home: Path, spawn: RecordingSpawn
) -> None:
    runtime, _ = rt
    record = _bind_session(runtime)
    intent_before = record.intent
    result = await runtime.attach_session("s1")
    assert "reattached" in result
    config = spawn.config()
    assert config["name"] == "s1"
    # The SAME socket path: BROKER_SOCKET was baked into the pane's
    # environment at split time.
    assert config["socket_path"] == record.socket_path
    assert config["budget_count"] == 6
    assert config["adopt"] == {
        "pane_id": "w3:p2",
        "claude_session_id": "cc-1",
        "transcript_path": "/private/tmp/t.jsonl",
    }
    assert config["resume"] == {
        "approved_prompt": "the first task",
        "completed": False,
    }
    # Attach touches none of intent, approved prompt or budget — the diff
    # against reassignment IS the feature.
    reloaded = Registry.load(home / "registry.json").get("s1")
    assert reloaded.intent == intent_before
    assert reloaded.approved_prompt == "the first task"
    assert reloaded.budget_count == 6
    assert reloaded.state == "spawning"
    assert reloaded.pid is not None


async def test_attach_refuses_while_a_broker_still_answers(
    rt: tuple[MasterRuntime, list[Any]], home: Path, spawn: RecordingSpawn
) -> None:
    runtime, _ = rt
    _bind_session(runtime)
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        with pytest.raises(RuntimeError) as exc:
            await runtime.attach_session("s1")
        assert "refusing to attach" in str(exc.value)
        assert spawn.argvs == []
    finally:
        server.close()
        await server.wait_closed()


async def test_attach_refuses_a_session_it_only_partly_knows(
    rt: tuple[MasterRuntime, list[Any]], spawn: RecordingSpawn
) -> None:
    runtime, _ = rt
    record = runtime.registry.get("s1")
    record.pane_id = "w3:p2"  # no claude session id, no transcript path
    runtime.registry.upsert(record)
    with pytest.raises(ValueError) as exc:
        await runtime.attach_session("s1")
    assert "claude_session_id" in str(exc.value)
    assert "transcript_path" in str(exc.value)
    assert spawn.argvs == []
    assert runtime.registry.get("s1").state == "driving"


async def test_attach_refuses_without_an_approved_prompt(
    rt: tuple[MasterRuntime, list[Any]], spawn: RecordingSpawn
) -> None:
    runtime, _ = rt
    record = _bind_session(runtime)
    record.approved_prompt = None
    runtime.registry.upsert(record)
    with pytest.raises(ValueError) as exc:
        await runtime.attach_session("s1")
    # A broker that died before approval left nothing to resume; the refusal
    # names the route that takes a new task.
    assert "reassign_session" in str(exc.value)
    assert spawn.argvs == []


async def test_attach_refuses_a_gone_session(
    rt: tuple[MasterRuntime, list[Any]], spawn: RecordingSpawn
) -> None:
    runtime, _ = rt
    _bind_session(runtime)
    # A session whose pane was found gone is removed, not marked: it is no
    # longer a routing candidate, so attach cannot even resolve it.
    runtime.registry.remove("s1")
    with pytest.raises(KeyError):
        await runtime.attach_session("s1")
    assert spawn.argvs == []


async def test_attach_resumes_completed(
    rt: tuple[MasterRuntime, list[Any]], spawn: RecordingSpawn
) -> None:
    runtime, _ = rt
    record = _bind_session(runtime)
    record.state = SessionState.COMPLETED
    runtime.registry.upsert(record)
    await runtime.attach_session("s1")
    # Resumed as completed, so reactivate_session still applies afterwards.
    assert spawn.config()["resume"] == {
        "approved_prompt": "the first task",
        "completed": True,
    }


async def test_attach_retracts_stranded_escalations(
    rt: tuple[MasterRuntime, list[Any]], spawn: RecordingSpawn
) -> None:
    runtime, posts = rt
    _bind_session(runtime)
    # Another session's escalations must survive: attaching s1 says nothing
    # about them.
    other_decision = EscalationPayload.model_validate(escalation_dict("e9", "s9"))
    other_prompt = PermissionEscalationPayload.model_validate(
        permission_pane_dict("p9", "s9")
    )
    runtime.queue.accept(other_decision)
    runtime.panes.accept(other_prompt)
    runtime.queue.accept(
        EscalationPayload.model_validate(escalation_dict("e1", "s1"))
    )
    runtime.panes.accept(
        PermissionEscalationPayload.model_validate(
            permission_pane_dict("p1", "s1")
        )
    )
    runtime.panes.accept(
        QuestionEscalationPayload.model_validate(question_escalation_dict("q1"))
    )
    await runtime.attach_session("s1")
    assert spawn.argvs != []  # the refusals all passed and a broker spawned
    # Both kinds retracted: a live entry would refuse the resumed broker's
    # first raise, and a dispatched decision would be discarded to its log.
    assert runtime.queue.entries == (other_decision,)
    assert runtime.panes.entries == (other_prompt,)
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("e1" in t and "retracted" in t for t in notices)
    assert any("p1" in t and "retracted" in t for t in notices)
    assert any("q1" in t and "retracted" in t for t in notices)


async def test_build_fleet_view_orders_rows_numerically_with_budgets_and_titles(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, _ = rt
    runtime.registry.upsert(
        SessionRecord(
            name="s10",
            socket_path="/private/tmp/s10.sock",
            cwd="/private/tmp",
            anchor_pane="%1",
            state=SessionState.DRIVING,
            intent="Fix the auth bug in the checkout flow before the demo",
            title="fix auth bug",
        )
    )
    runtime.registry.upsert(
        SessionRecord(
            name="s2",
            socket_path="/private/tmp/s2.sock",
            cwd="/private/tmp",
            anchor_pane="%1",
            state=SessionState.ESCALATED,
            approved_prompt="Migrate the users table",
            title="migrate users table",
            budget_count=5,
        )
    )
    view = runtime.build_fleet_view()
    # Numeric order (s2 before s10), not lexical.
    assert [row.session_id for row in view.rows] == ["s1", "s2", "s10"]
    # s1 was never approved: no title is set on it, and the row shows none —
    # title no longer falls back to approved_prompt or intent.
    assert view.rows[0].title == ""
    s2 = view.rows[1]
    assert s2.state == SessionState.ESCALATED
    assert s2.title == "migrate users table"
    assert s2.budget_count == 5
    assert s2.budget_max == runtime.cfg.budget_max
    s10 = view.rows[2]
    assert s10.title == "fix auth bug"


async def test_build_fleet_view_idle_master_and_no_sessions(home: Path) -> None:
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    queue = EscalationQueue.load(home / "escalation-queue.json")
    panes = PaneEscalations.load(home / "pane-escalations.json")
    runtime = MasterRuntime(
        lambda _event: None,
        registry,
        queue,
        panes,
        cfg,
        anchor_pane="%1",
        claude_json=home / "claude.json",
    )
    view = runtime.build_fleet_view()
    assert view.master_activity is None
    assert view.rows == ()
    assert view.queue_depth == 0
    assert view.panes == ()


async def test_build_fleet_view_reports_master_activity_and_queue_state(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, _ = rt
    for name in ("s2", "s10"):
        _add_session(runtime, name)
    runtime.note_master_activity("thinking…")
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    assert (await send(runtime, T_ESCALATION, escalation_dict("e2", "s2"), "s2")).ok
    for name, esc_id in (("s10", "p10"), ("s2", "p2")):
        assert (
            await send(
                runtime,
                T_PANE_ESCALATION,
                permission_pane_dict(esc_id, name),
                session=name,
            )
        ).ok
    view = runtime.build_fleet_view()
    assert view.master_activity == "thinking…"
    # Only decisions count as waiting; open prompts are listed apart, in
    # numeric session order (s2 before s10) whatever order they arrived in.
    assert view.queue_depth == 2
    assert view.waiting == ("s2",)
    assert [p.session_id for p in view.panes] == ["s2", "s10"]


async def test_build_fleet_view_badges_reflect_queue_proposals_and_prompts(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, _ = rt
    runtime.registry.upsert(
        SessionRecord(
            name="s2",
            socket_path="/private/tmp/s2.sock",
            cwd="/private/tmp",
            anchor_pane="%1",
            state=SessionState.DRIVING,
        )
    )
    runtime.registry.upsert(
        SessionRecord(
            name="s3",
            socket_path="/private/tmp/s3.sock",
            cwd="/private/tmp",
            anchor_pane="%1",
            state=SessionState.DRIVING,
            pane_id="w3:p2",
        )
    )
    # s1: a queued decision escalation AND an open permission escalation from
    # the same session — held apart, so both badges carry.
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1", "s1"))).ok
    assert (
        await send(
            runtime, T_PANE_ESCALATION, permission_pane_dict("p1", "s1")
        )
    ).ok
    assert (
        await send(
            runtime, T_PANE_ESCALATION, question_escalation_dict("q1", "s1")
        )
    ).ok
    # s2: a pending prompt proposal.
    assert (
        await send(
            runtime,
            T_PROMPT_PROPOSAL,
            {
                "proposal_id": "prop-1",
                "proposed_prompt": "do it",
                "grounding_summary": "facts",
            },
            session="s2",
        )
    ).ok
    # s3: sitting on a native permission prompt.
    assert (
        await send(
            runtime,
            T_LIVE_STATUS,
            {"state": "driving", "permission_prompt": True},
            session="s3",
        )
    ).ok
    rows = {row.session_id: row for row in runtime.build_fleet_view().rows}
    assert rows["s1"].badges == (
        Attention.ESCALATION,
        Attention.PERMISSION,
        Attention.QUESTION,
    )
    assert rows["s2"].badges == (Attention.PROPOSAL,)
    assert rows["s3"].badges == (Attention.PERMISSION,)
    assert rows["s3"].pane_id == "w3:p2"


async def test_live_status_updates_state_activity_and_perm(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    resp = await send(
        runtime,
        T_LIVE_STATUS,
        {
            "state": "escalated",
            "activity": "reviewing a permission request…",
            "permission_prompt": True,
            "task_activity": "reviewing the login flow",
        },
    )
    assert resp.ok  # every push is ACKed so the sender never spins
    assert runtime.registry.get("s1").state == "escalated"
    fleets = [m for m in posts if isinstance(m, FleetUpdated)]
    assert fleets  # the push published a fresh snapshot
    row = fleets[-1].view.rows[0]
    assert row.state == SessionState.ESCALATED
    assert Attention.PERMISSION in row.badges
    assert row.broker_activity == "reviewing a permission request…"
    assert row.task_activity == "reviewing the login flow"
    # A follow-up clearing push empties activity, task activity, and the
    # PERMISSION badge.
    resp = await send(
        runtime,
        T_LIVE_STATUS,
        {"state": "driving", "activity": "", "permission_prompt": False},
    )
    assert resp.ok
    row = [m for m in posts if isinstance(m, FleetUpdated)][-1].view.rows[0]
    assert row.state == SessionState.DRIVING
    assert row.task_activity == ""
    assert Attention.PERMISSION not in row.badges
    assert row.broker_activity == ""


async def test_live_status_persists_the_session_identity(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    identity = {
        "pane_id": "w3:p2",
        "claude_session_id": "cc-1",
        "transcript_path": "/private/tmp/cc-1.jsonl",
    }
    assert (await send(runtime, T_LIVE_STATUS, {"state": "driving", **identity})).ok
    # Persisted, not just held: a master restarted after its brokers died
    # has only the registry file to adopt or reconcile from.
    record = Registry.load(home / "registry.json").get("s1")
    assert (record.pane_id, record.claude_session_id, record.transcript_path) == (
        "w3:p2",
        "cc-1",
        "/private/tmp/cc-1.jsonl",
    )
    # A push that has not learned a field never erases what the registry holds.
    assert (await send(runtime, T_LIVE_STATUS, {"state": "driving"})).ok
    assert Registry.load(home / "registry.json").get("s1").pane_id == "w3:p2"
    # /clear in a settled session binds a new Claude session: the absorbing
    # guard that drops late state must not drop the new identity with it.
    assert (
        await send(runtime, T_COMPLETION, {"headline": "done", "supporting": "s"})
    ).ok
    assert (
        await send(
            runtime,
            T_LIVE_STATUS,
            {
                "state": "completed",
                **identity,
                "claude_session_id": "cc-2",
                "transcript_path": "/private/tmp/cc-2.jsonl",
            },
        )
    ).ok
    record = Registry.load(home / "registry.json").get("s1")
    assert record.claude_session_id == "cc-2"
    assert record.transcript_path == "/private/tmp/cc-2.jsonl"


async def test_set_state_is_idempotent(
    rt: tuple[MasterRuntime, list[Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, posts = rt
    saves: list[int] = []
    orig_save = runtime.registry.save

    def counting_save() -> None:
        saves.append(1)
        orig_save()

    monkeypatch.setattr(runtime.registry, "save", counting_save)
    assert (await send(runtime, T_LIVE_STATUS, {"state": "escalated"})).ok
    changed = [m for m in posts if isinstance(m, SessionStateChanged)]
    assert len(changed) == 1
    assert len(saves) == 1
    fleet_count = len([m for m in posts if isinstance(m, FleetUpdated)])
    # The same state again: no second announcement, no second save — but the
    # snapshot still publishes for the activity/perm side of the push.
    assert (await send(runtime, T_LIVE_STATUS, {"state": "escalated"})).ok
    changed = [m for m in posts if isinstance(m, SessionStateChanged)]
    assert len(changed) == 1
    assert len(saves) == 1
    assert (
        len([m for m in posts if isinstance(m, FleetUpdated)])
        == fleet_count + 1
    )


async def test_absorbing_state_ignores_late_pushes(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    assert (
        await send(runtime, T_COMPLETION, {"headline": "done", "supporting": "s"})
    ).ok
    assert runtime.registry.get("s1").state == "completed"
    # A stale in-flight push carrying an older state arrives late: it must
    # not resurrect the settled session — but it is still ACKed.
    resp = await send(
        runtime,
        T_LIVE_STATUS,
        {"state": "driving", "activity": "reviewing the latest turn…"},
    )
    assert resp.ok
    assert runtime.registry.get("s1").state == "completed"
    # The sanctioned exit is a master-initiated boundary write: reactivate
    # lifts the session back into work, after which pushes apply again.
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        result = await runtime.reactivate_session("s1", "next task")
        assert "reactivated" in result
        assert runtime.registry.get("s1").state == "grounding"
        assert (await send(runtime, T_LIVE_STATUS, {"state": "driving"})).ok
        assert runtime.registry.get("s1").state == "driving"
    finally:
        server.close()
        await server.wait_closed()


async def test_absorbing_transition_clears_live_status(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (
        await send(
            runtime,
            T_LIVE_STATUS,
            {
                "state": "driving",
                "activity": "reviewing a permission request…",
                "permission_prompt": True,
                "task_activity": "reviewing the login flow",
            },
        )
    ).ok
    # The master stops the session mid-activity. The broker is gone, so no
    # clearing push is coming and the guard would refuse it anyway — the
    # boundary write itself must retire the phrase and the ⚠.
    await runtime.stop_session("s1")
    row = [m for m in posts if isinstance(m, FleetUpdated)][-1].view.rows[0]
    assert row.state == SessionState.STOPPED
    assert row.broker_activity == ""
    assert row.task_activity == ""
    assert row.badges == ()


async def test_list_sessions_probes_rather_than_reading_the_pushed_map(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, _ = rt
    # The last push said no prompt was pending — as after a master restart,
    # where the transient map is empty and no re-seeding push is coming for a
    # session already sitting on its prompt.
    assert (
        await send(
            runtime,
            T_LIVE_STATUS,
            {"state": "driving", "permission_prompt": False},
        )
    ).ok
    stub = StatusSession(permission_prompt=True)
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        listing = await runtime.list_sessions()
        # The tool's authoritative probe wins over the stale pushed state.
        assert "sitting on a permission prompt" in listing
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
    record.title = "the first task title"
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
        assert reloaded.title == ""  # cleared alongside the approved prompt
        # Acceptance is not delivery: only the broker's budget update resets it.
        assert reloaded.budget_count == 4
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


async def test_every_message_from_an_unknown_session_is_refused(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    """A message from a session the master cannot route to changes nothing:
    a completion, proposal or budget it applied would describe no session."""
    runtime, posts = rt
    before = len(posts)
    for msg_type, payload in broker_messages("ghost"):
        resp = await send(runtime, msg_type, payload, session="ghost")
        assert resp.payload["reason_code"] == NackCode.UNKNOWN_SESSION, msg_type
    assert list(runtime.registry.records) == ["s1"]
    assert runtime.queue.active is None
    assert runtime.panes.entries == ()
    assert runtime.proposals == {}
    assert all(isinstance(m, Notice) for m in posts[before:])


async def test_escalation_naming_another_session_is_refused(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, _ = rt
    _add_session(runtime, "s2")
    resp = await send(runtime, T_ESCALATION, escalation_dict("e1", "s2"))
    assert resp.payload["reason_code"] == NackCode.UNKNOWN_SESSION
    resp = await send(
        runtime, T_PANE_ESCALATION, permission_pane_dict("p1", "s2")
    )
    assert resp.payload["reason_code"] == NackCode.UNKNOWN_SESSION
    assert runtime.queue.active is None
    assert runtime.panes.entries == ()
    assert runtime.registry.get("s2").state == SessionState.DRIVING


async def test_unknown_or_malformed_message_is_refused_as_malformed(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    resp = await send(runtime, "no_such_type", {})
    assert resp.payload["reason_code"] == NackCode.MALFORMED
    resp = await send(runtime, T_BUDGET_UPDATE, {"count": "many"})
    assert resp.payload["reason_code"] == NackCode.MALFORMED
    assert runtime.registry.get("s1").budget_count == 0
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("no_such_type" in t for t in notices)
    assert any("MALFORMED" in t and T_BUDGET_UPDATE in t for t in notices)


def test_log_notice_writes_every_notice_to_the_configured_log_file(
    tmp_path: Path,
) -> None:
    root = logging.getLogger()
    saved = list(root.handlers)
    saved_level = root.level
    for handler in saved:
        root.removeHandler(handler)
    try:
        log_path = tmp_path / "master.log"
        logging_setup.configure(log_path)
        log_notice(Notice("a stranded escalation"))
        log_notice(SessionStateChanged("s1", SessionState.DRIVING))
        text = log_path.read_text(encoding="utf-8")
        assert "a stranded escalation" in text
        assert text.count("\n") == 1
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in saved:
            root.addHandler(handler)
        root.setLevel(saved_level)
