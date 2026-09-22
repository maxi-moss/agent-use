"""MasterRuntime harness: fake broker = protocol.client against the real
runtime server; driver.subprocess.run monkeypatched; emit = recording list."""

import asyncio
import contextlib
import json
import subprocess
import tempfile
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from broker.config import BrokerConfig
from broker.herdr import driver
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.master.runtime import (
    PANE_UNKNOWN,
    MasterRuntime,
    render_escalation,
    render_permission_escalation,
)
from broker.master.viewmodel import (
    Attention,
    CompletionArrived,
    EscalationArrived,
    FleetUpdated,
    Notice,
    PermissionEscalationArrived,
    ProposalArrived,
    SessionStatusChanged,
)
from broker.protocol import client
from broker.protocol.constants import (
    NACK_MALFORMED,
    NACK_PROTOCOL_VIOLATION,
    NACK_UNKNOWN_SESSION,
    SessionState,
    T_APPROVE_PROMPT,
    T_CLARIFY_ESCALATION,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_FATAL_ERROR,
    T_GET_PERMISSION_LOG,
    T_LIVE_STATUS,
    T_PERMISSION_ESCALATION,
    T_PROMPT_PROPOSAL,
    T_REACTIVATE,
    T_RETRACT,
    T_SESSION_ENDED,
    T_STATUS,
)
from broker.protocol.schemas import (
    Envelope,
    EscalationPayload,
    PermissionEscalationPayload,
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
        return Response(id=env.id, ok=False)


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
            return Response(id=env.id, ok=False)
        return Response(
            id=env.id,
            ok=True,
            payload={
                "state": self.state,
                "pane_id": "w3:p2",
                "permission_prompt": self.permission_prompt,
                "task_activity": self.task_activity,
            },
        )


def escalation_dict(esc_id: str = "e1", session: str = "s1") -> dict[str, Any]:
    return {
        "escalation_id": esc_id,
        "session_id": session,
        "raiser": {"component": "broker", "session_id": session},
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


def permission_escalation_dict(
    esc_id: str = "p1", session: str = "s1"
) -> dict[str, Any]:
    return {
        "escalation_id": esc_id,
        "session_id": session,
        "raiser": {"component": "permission", "session_id": session},
        "tool_name": "tool-name-value",
        "tool_input": {"command": "command-value"},
        "task_intent": "task-intent-value",
        "reason": "reason-value",
        "raised_at": "2026-07-29T12:00:00+00:00",
        "permission_suggestions": [
            {"type": "setMode", "mode": "mode-value"}
        ],
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
    queue = EscalationQueue.load(home / "escalation-queue.json")
    runtime = MasterRuntime(posts.append, registry, queue, cfg, anchor_pane="%1")
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
    # Every value the developer decides on appears byte-for-byte — no
    # paraphrase. The raiser is routing identity, not something they read.
    for value in _leaf_values({k: v for k, v in payload.items() if k != "raiser"}):
        assert value in rendered
    assert rendered == render_escalation(
        EscalationPayload.model_validate(payload)
    )
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "e1"


async def test_second_escalation_while_active_is_protocol_violation(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    resp = await send(runtime, T_ESCALATION, escalation_dict("e2"))
    assert not resp.ok
    # Same raiser: the broker was told to hold one at a time and did not.
    assert resp.payload["reason_code"] == NACK_PROTOCOL_VIOLATION
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("PROTOCOL VIOLATION" in t for t in notices)
    # The first escalation stays active; the second is never surfaced.
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "e1"
    assert (
        len([m for m in posts if isinstance(m, EscalationArrived)]) == 1
    )


async def test_permission_escalation_rendered_names_pane_and_offers_no_dispatch(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    record = runtime.registry.get("s1")
    record.pane_id = "w3:p2"
    runtime.registry.upsert(record)
    payload = permission_escalation_dict()
    resp = await send(runtime, T_PERMISSION_ESCALATION, payload)
    assert resp.ok
    arrived = [m for m in posts if isinstance(m, PermissionEscalationArrived)]
    assert len(arrived) == 1
    rendered = arrived[0].rendered
    assert rendered == render_permission_escalation(
        PermissionEscalationPayload.model_validate(payload), "w3:p2"
    )
    # Every field the developer judges the prompt on appears byte-for-byte.
    # The raiser is routing identity and raised_at is the resolution baseline;
    # neither is something they read.
    judged = {
        k: v
        for k, v in payload.items()
        if k not in ("raiser", "raised_at")
    }
    for value in _leaf_values(judged):
        assert value in rendered
    # No timestamp reaches the block: it is carried into the master's LLM
    # context, where a clock reading is only ever something to reason from.
    assert payload["raised_at"] not in rendered
    assert "w3:p2" in rendered  # the pane the native prompt is waiting in
    assert "cannot be answered here" in rendered
    assert "dispatch" not in rendered.lower()  # no affordance to answer it here
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "p1"


async def test_same_raiser_double_raise_is_protocol_violation(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (
        await send(runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1"))
    ).ok
    resp = await send(
        runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p2")
    )
    assert not resp.ok
    assert resp.payload["reason_code"] == NACK_PROTOCOL_VIOLATION
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "p1"
    assert (
        len([m for m in posts if isinstance(m, PermissionEscalationArrived)]) == 1
    )


async def test_cross_raiser_escalation_queues_behind_the_active_one(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    resp = await send(
        runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
    )
    # Two distinct raisers: the second one broke no rule, so it waits its
    # turn rather than being refused.
    assert resp.ok
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "e1"
    # Not surfaced yet: the developer sees the head and a depth line only.
    assert [m for m in posts if isinstance(m, PermissionEscalationArrived)] == []
    fleets = [m for m in posts if isinstance(m, FleetUpdated)]
    assert fleets[-1].view.queue_depth == 2
    assert fleets[-1].view.waiting == ("s1",)


async def test_malformed_permission_escalation_nacked_never_surfaced(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    thin = permission_escalation_dict()
    del thin["reason"]
    resp = await send(runtime, T_PERMISSION_ESCALATION, thin)
    assert not resp.ok
    assert resp.payload["reason_code"] == NACK_MALFORMED
    # A prompt the developer cannot act on is worse than none at all.
    assert [m for m in posts if isinstance(m, PermissionEscalationArrived)] == []
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("MALFORMED" in t for t in notices)
    assert runtime.queue.active is None


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
    assert resp.payload["reason_code"] == NACK_UNKNOWN_SESSION
    resp = await send(
        runtime,
        T_PERMISSION_ESCALATION,
        permission_escalation_dict("p1", "ghost"),
        session="ghost",
    )
    assert not resp.ok
    assert resp.payload["reason_code"] == NACK_UNKNOWN_SESSION
    assert runtime.queue.active is None
    assert [m for m in posts if isinstance(m, EscalationArrived)] == []
    assert [m for m in posts if isinstance(m, PermissionEscalationArrived)] == []


async def test_dispatch_refuses_permission_escalation(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (
            await send(
                runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
            )
        ).ok
        result = await runtime.dispatch("p1", "yes, go ahead")
        assert "NOT dispatched" in result
        # The registry has no pane for s1 here; the refusal still has to say
        # where the answer belongs rather than go silent.
        assert PANE_UNKNOWN in result
        assert stub.envelopes == []  # nothing reached the session
        assert runtime.queue.active is not None  # nothing was resolved either
        notices = [m.text for m in posts if isinstance(m, Notice)]
        assert any("NOT dispatched" in t for t in notices)
    finally:
        server.close()
        await server.wait_closed()


async def test_reassign_retracts_that_sessions_permission_escalation(
    rt: tuple[MasterRuntime, list[Any]], home: Path, spawn: RecordingSpawn
) -> None:
    runtime, posts = rt
    _bind_session(runtime)
    # Another session's escalation must survive: reassigning s1 says nothing
    # about it.
    other = PermissionEscalationPayload.model_validate(
        permission_escalation_dict("p9", "s9")
    )
    runtime.queue.accept(other)
    await runtime.reassign_session("s1", "take it from here")
    assert runtime.queue.active is other

    runtime.queue.retract("p9")
    assert (
        await send(
            runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
        )
    ).ok
    await runtime.reassign_session("s1", "and again")
    # Its raiser died with the broker, so nothing else would ever clear it.
    assert runtime.queue.active is None
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("p1" in t and "retracted" in t for t in notices)


async def test_stop_session_retracts_that_sessions_permission_escalation(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    # A broker-raised escalation from the same session must survive the stop:
    # raiser identity is the only thing telling the two kinds apart.
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    await runtime.stop_session("s1")
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "e1"

    runtime.queue.retract("e1")
    assert (
        await send(
            runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
        )
    ).ok
    assert runtime.queue.active is not None
    await runtime.stop_session("s1")
    # Its raiser died with the broker and the native prompt is still on screen,
    # so no retraction is ever coming and the slot would stay held for every
    # other session too.
    assert runtime.queue.active is None
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("p1" in t and "retracted" in t for t in notices)


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
    stub = StatusSession(permission_prompt=True)
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        listing = await runtime.render_sessions_with_permission_prompts()
        assert "s1" in listing
        assert "sitting on a permission prompt" in listing
        assert "w3:p2" in listing
        # The flag is read on demand and must NOT reach the summary the master
        # carries into every turn.
        assert "permission prompt" not in runtime.render_registry_summary()
    finally:
        server.close()
        await server.wait_closed()


async def test_list_sessions_degrades_when_a_session_is_unreachable(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    """One dead session costs a line of the listing, never the whole listing."""
    runtime, _ = rt
    listing = await runtime.render_sessions_with_permission_prompts()
    assert "s1" in listing
    assert "unreachable" in listing


async def test_retract_clears_head_and_informs(
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
    runtime, _ = rt
    assert (
        await send(
            runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
        )
    ).ok
    record = runtime.registry.get("s1")
    record.state = SessionState.ESCALATED
    runtime.registry.upsert(record)
    assert (
        await send(
            runtime, T_RETRACT, {"escalation_id": "p1", "reason": "answered"}
        )
    ).ok
    assert runtime.registry.get("s1").state == SessionState.ESCALATED


async def test_queued_escalation_surfaces_after_resolve(
    rt: tuple[MasterRuntime, list[Any]],
    home: Path,
    recording_run: RecordingRun,
) -> None:
    runtime, posts = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)
    try:
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert (
            await send(
                runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
            )
        ).ok
        assert (
            [m for m in posts if isinstance(m, PermissionEscalationArrived)]
            == []
        )
        result = await runtime.dispatch("e1", "use option B")
        assert "dispatched" in result
        arrived = [
            m for m in posts if isinstance(m, PermissionEscalationArrived)
        ]
        assert len(arrived) == 1
        assert arrived[0].escalation_id == "p1"
        assert runtime.queue.active is not None
        assert runtime.queue.active.escalation_id == "p1"
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
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    assert (
        await send(
            runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
        )
    ).ok
    assert (
        await send(
            runtime, T_RETRACT, {"escalation_id": "e1", "reason": "answered"}
        )
    ).ok
    arrived = [m for m in posts if isinstance(m, PermissionEscalationArrived)]
    assert len(arrived) == 1
    assert arrived[0].escalation_id == "p1"
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "p1"


async def test_retracted_queued_escalation_is_never_surfaced(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    assert (
        await send(
            runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
        )
    ).ok
    assert (
        await send(
            runtime, T_RETRACT, {"escalation_id": "p1", "reason": "superseded"}
        )
    ).ok
    assert (
        await send(
            runtime, T_RETRACT, {"escalation_id": "e1", "reason": "answered"}
        )
    ).ok
    # The queued escalation was withdrawn before its turn; announcing it would
    # hand the developer a decision nobody is waiting on.
    assert [m for m in posts if isinstance(m, PermissionEscalationArrived)] == []
    assert runtime.queue.active is None


async def test_notification_fires_at_surface_time_not_accept(
    rt: tuple[MasterRuntime, list[Any]],
    home: Path,
    recording_run: RecordingRun,
) -> None:
    runtime, _ = rt
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
            await send(
                runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
            )
        ).ok
        assert notify_count() == 1  # only the surfaced head is announced
        await runtime.dispatch("e1", "use option B")
        assert notify_count() == 2  # the next head announces when it surfaces
    finally:
        server.close()
        await server.wait_closed()


async def test_startup_resurfaces_the_persisted_head(
    home: Path, recording_run: RecordingRun
) -> None:
    queue_path = home / "escalation-queue.json"
    seeded = EscalationQueue.load(queue_path)
    seeded.accept(EscalationPayload.model_validate(escalation_dict("e1")))
    seeded.accept(
        PermissionEscalationPayload.model_validate(
            permission_escalation_dict("p1")
        )
    )
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
    runtime = MasterRuntime(
        posts.append,
        registry,
        EscalationQueue.load(queue_path),
        cfg,
        anchor_pane="%1",
    )
    task = asyncio.create_task(runtime.serve())
    for _ in range(200):
        if runtime.master_socket_path.exists():
            break
        await asyncio.sleep(0.01)
    else:
        raise TimeoutError("master socket never bound")
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    arrived = [m for m in posts if isinstance(m, EscalationArrived)]
    assert len(arrived) == 1
    assert arrived[0].escalation_id == "e1"
    assert arrived[0].rendered == render_escalation(
        EscalationPayload.model_validate(escalation_dict("e1"))
    )
    # The waiting entry stays unannounced; the fleet view carries it.
    assert [m for m in posts if isinstance(m, PermissionEscalationArrived)] == []
    fleets = [m for m in posts if isinstance(m, FleetUpdated)]
    assert fleets[0].view.queue_depth == 2
    assert fleets[0].view.waiting == ("s1",)


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
        cfg,
        anchor_pane="%1",
    )
    try:
        await runtime.probe_status("s1")
    finally:
        server.close()
    assert registry.get("s1").state is SessionState.COMPLETED
    assert [
        (m.session_id, m.state) for m in posts if isinstance(m, SessionStatusChanged)
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
        cfg,
        anchor_pane="%1",
    )
    try:
        task = asyncio.create_task(runtime.serve())

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
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    finally:
        server.close()
        await server.wait_closed()


async def test_stop_session_retracts_a_queued_permission_escalation(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    assert (
        await send(
            runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
        )
    ).ok
    await runtime.stop_session("s1")
    # The stranded retraction reaches an entry that never surfaced; the broker
    # escalation from the same session is a different raiser and survives.
    assert runtime.queue.depth == 1
    assert runtime.queue.active is not None
    assert runtime.queue.active.escalation_id == "e1"
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("p1" in t and "retracted" in t for t in notices)
    assert [m for m in posts if isinstance(m, PermissionEscalationArrived)] == []


async def test_queue_depth_reflects_every_mutation(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    stub = StubSession()
    server = await serve_unix(home / "s" / "s1.sock", stub.handler)

    def depth() -> tuple[int, tuple[str, ...]]:
        fleets = [m for m in posts if isinstance(m, FleetUpdated)]
        return (fleets[-1].view.queue_depth, fleets[-1].view.waiting)

    try:
        assert depth() == (0, ())  # serve() announces the loaded queue
        assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
        assert depth() == (1, ())
        assert (
            await send(
                runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
            )
        ).ok
        assert depth() == (2, ("s1",))
        await runtime.dispatch("e1", "use option B")
        assert depth() == (1, ())
        assert (
            await send(
                runtime, T_RETRACT, {"escalation_id": "p1", "reason": "answered"}
            )
        ).ok
        assert depth() == (0, ())
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
        assert runtime.queue.active is None
        # Single authority: the master never invents DRIVING at dispatch —
        # the state stays until the broker reports its own transition.
        assert runtime.registry.get("s1").state == "escalated"
        assert (
            await send(runtime, T_LIVE_STATUS, {"state": "driving"})
        ).ok
        assert runtime.registry.get("s1").state == "driving"
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
        assert (
            await send(
                runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
            )
        ).ok
        result = await runtime.clarify_escalation("p1", "q")
        assert "NOT sent" in result
        assert PANE_UNKNOWN in result
        assert stub.envelopes == []
        assert runtime.queue.active is not None
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


async def test_session_ended_removes_from_fleet(
    rt: tuple[MasterRuntime, list[Any]], home: Path
) -> None:
    runtime, posts = rt
    # s1 is seeded driving and visible in the summary the master carries.
    assert "s1" in runtime.render_registry_summary()
    resp = await send(runtime, T_SESSION_ENDED, {})
    assert resp.ok
    # Removed everywhere a router looks: the summary, the fleet, and the
    # registry itself — so it can never be handed a new task or a decision.
    assert "s1" not in runtime.registry.records
    assert runtime.render_registry_summary() == "(no sessions)"
    last_view = [m for m in posts if isinstance(m, FleetUpdated)][-1].view
    assert not any(row.session_id == "s1" for row in last_view.rows)
    # The removal is durable, and a late live-status push cannot resurrect it.
    assert "s1" not in Registry.load(home / "registry.json").records
    assert (await send(runtime, T_LIVE_STATUS, {"state": "driving"})).ok
    assert "s1" not in runtime.registry.records


async def test_session_ended_for_unknown_session_is_acked(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    before = len(posts)
    resp = await send(runtime, T_SESSION_ENDED, {}, session="ghost")
    # A report for a session already gone (or never known) is ACKed and posts
    # nothing — the sender never spins re-reporting.
    assert resp.ok
    assert len(posts) == before


async def test_session_ended_retracts_its_stranded_escalation(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    runtime, posts = rt
    # s1 is the surfaced FIFO head, waiting on a decision the developer never
    # gave before running /exit; a bystander waits behind it and must survive.
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1", "s1"))).ok
    other = EscalationPayload.model_validate(escalation_dict("e9", "s9"))
    runtime.queue.accept(other)
    assert runtime.queue.active is not other
    resp = await send(runtime, T_SESSION_ENDED, {})
    assert resp.ok
    # The broker exits without withdrawing it, so ending must retract it —
    # otherwise the head wedges the queue forever, undispatchable to a gone
    # session, and blocks the bystander behind it.
    assert runtime.queue.depth == 1
    assert runtime.queue.active is other
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("e1" in t and "retracted" in t for t in notices)


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
    assert runtime.queue.active is None


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
            runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1")
        )
    ).ok
    assert (
        await send(
            runtime, T_RETRACT, {"escalation_id": "p1", "reason": "r"}
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
            "retrieved": [
                {"name": "a.py::f", "score": 0.81},
                {"name": "a.py::g", "score": None},
            ],
        },
    )
    assert resp.ok
    arrived = [m for m in posts if isinstance(m, ProposalArrived)]
    assert len(arrived) == 1
    assert "the exact proposed prompt" in arrived[0].rendered
    assert "the exact grounding summary" in arrived[0].rendered
    assert "## Retrieved code\n- a.py::f (seed 0.81)\n- a.py::g" in arrived[0].rendered
    assert runtime.registry.get("s1").state == "awaiting_approval"


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
    # Another session's escalation must survive: attaching s1 says nothing
    # about it.
    other = PermissionEscalationPayload.model_validate(
        permission_escalation_dict("p9", "s9")
    )
    runtime.queue.accept(other)
    runtime.queue.accept(
        EscalationPayload.model_validate(escalation_dict("e1", "s1"))
    )
    runtime.queue.accept(
        PermissionEscalationPayload.model_validate(
            permission_escalation_dict("p1", "s1")
        )
    )
    await runtime.attach_session("s1")
    assert spawn.argvs != []  # the refusals all passed and a broker spawned
    # Both raiser identities retracted: a live same-raiser entry would refuse
    # the resumed broker's first escalation, and a dispatched decision would
    # be discarded to its log.
    assert runtime.queue.depth == 1
    assert runtime.queue.active is other
    notices = [m.text for m in posts if isinstance(m, Notice)]
    assert any("e1" in t and "retracted" in t for t in notices)
    assert any("p1" in t and "retracted" in t for t in notices)


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
    runtime = MasterRuntime(
        lambda _event: None, registry, queue, cfg, anchor_pane="%1"
    )
    view = runtime.build_fleet_view()
    assert view.master_activity is None
    assert view.rows == ()
    assert view.queue_depth == 0


async def test_build_fleet_view_reports_master_activity_and_queue_state(
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
    runtime.note_master_activity("thinking…")
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1"))).ok
    assert (
        await send(
            runtime,
            T_PERMISSION_ESCALATION,
            permission_escalation_dict("p1", "s2"),
            session="s2",
        )
    ).ok
    view = runtime.build_fleet_view()
    assert view.master_activity == "thinking…"
    assert view.queue_depth == 2
    assert view.waiting == ("s2",)


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
    # s1: a queued broker escalation AND a queued permission escalation from
    # the same session — distinct raiser identities, so both badges carry.
    assert (await send(runtime, T_ESCALATION, escalation_dict("e1", "s1"))).ok
    assert (
        await send(
            runtime, T_PERMISSION_ESCALATION, permission_escalation_dict("p1", "s1")
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
    assert rows["s1"].badges == (Attention.ESCALATION, Attention.PERMISSION)
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
    # A push from an unknown session is still ACKed.
    resp = await send(
        runtime, T_LIVE_STATUS, {"state": "driving"}, session="ghost"
    )
    assert resp.ok


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
    changed = [m for m in posts if isinstance(m, SessionStatusChanged)]
    assert len(changed) == 1
    assert len(saves) == 1
    fleet_count = len([m for m in posts if isinstance(m, FleetUpdated)])
    # The same state again: no second announcement, no second save — but the
    # snapshot still publishes for the activity/perm side of the push.
    assert (await send(runtime, T_LIVE_STATUS, {"state": "escalated"})).ok
    changed = [m for m in posts if isinstance(m, SessionStatusChanged)]
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
    assert (await send(runtime, T_COMPLETION, {"summary": "done"})).ok
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
        listing = await runtime.render_sessions_with_permission_prompts()
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
