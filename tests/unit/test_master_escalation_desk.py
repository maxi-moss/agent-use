"""EscalationDesk harness: real stores in a temp dir, a scripted BrokerLink in
place of the broker sockets, driver.subprocess.run monkeypatched, emit = a
recording list."""

import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from broker.config import BrokerConfig
from broker.herdr import driver
from broker.master.broker_link import BrokerLink
from broker.master.escalation_desk import (
    CLARIFY_ESCALATION_TIMEOUT_S,
    EscalationDesk,
)
from broker.master.pane_escalations import PaneEscalations
from broker.master.payload_render import PANE_UNKNOWN
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.master.viewmodel import EscalationArrived, Notice
from broker.paths import BrokerPaths
from broker.protocol.constants import SessionState
from broker.protocol.schemas import (
    PANE_ESCALATION_ADAPTER,
    ClarifyEscalationRequestPayload,
    DecisionDeliveredPayload,
    DecisionUndeliveredPayload,
    DispatchDecisionPayload,
    EscalationPayload,
    EscalationRetractPayload,
    Response,
    WireMessage,
)


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

    def notifications(self) -> int:
        return len([c for c in self.calls if c[1:3] == ["notification", "show"]])


async def _ack(_payload: WireMessage) -> Response:
    return Response(id="r", ok=True)


class ScriptedLink(BrokerLink):
    """BrokerLink whose socket calls are answered by a script, not a broker."""

    def __init__(self, home: Path) -> None:
        paths = BrokerPaths(home)
        super().__init__(
            paths,
            BrokerConfig(model_id="test-model", broker_home=home),
            paths.master_socket,
            home / "claude.json",
            lambda _name, _code: None,
        )
        self.sent: list[WireMessage] = []
        self.timeouts: list[float] = []
        self.answer: Callable[[WireMessage], Awaitable[Response]] = _ack

    async def request(
        self, record: SessionRecord, payload: WireMessage, *, timeout_s: float
    ) -> Response:
        self.sent.append(payload)
        self.timeouts.append(timeout_s)
        return await self.answer(payload)


@dataclass
class Harness:
    desk: EscalationDesk
    link: ScriptedLink
    posts: list[Any]
    run: RecordingRun

    def notices(self) -> list[str]:
        return [m.text for m in self.posts if isinstance(m, Notice)]

    def surfaced(self) -> list[str]:
        return [
            m.escalation_id for m in self.posts if isinstance(m, EscalationArrived)
        ]


@pytest.fixture
def h(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    run = RecordingRun()
    monkeypatch.setattr(driver.subprocess, "run", run)
    registry = Registry.load(tmp_path / "registry.json")
    for name in ("s1", "s2"):
        registry.upsert(
            SessionRecord(
                name=name,
                socket_path=str(tmp_path / f"{name}.sock"),
                cwd="/private/tmp",
                anchor_pane="%1",
                state=SessionState.DRIVING,
            )
        )
    posts: list[Any] = []
    link = ScriptedLink(tmp_path)
    desk = EscalationDesk(
        EscalationQueue.load(tmp_path / "escalation-queue.json"),
        PaneEscalations.load(tmp_path / "pane-escalations.json"),
        registry,
        link,
        posts.append,
    )
    return Harness(desk, link, posts, run)


def escalation(esc_id: str = "e1", session: str = "s1") -> EscalationPayload:
    return EscalationPayload.model_validate(
        {
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
    )


async def accept_permission(h: Harness, esc_id: str = "p1") -> None:
    payload = PANE_ESCALATION_ADAPTER.validate_python(
        {
            "kind": "permission",
            "escalation_id": esc_id,
            "session_id": "s1",
            "tool_name": "tool-name-value",
            "tool_input": {"command": "command-value"},
            "task_intent": "task-intent-value",
            "reason": "reason-value",
            "permission_suggestions": [{"type": "setMode", "mode": "mode-value"}],
        }
    )
    assert await h.desk.accept_pane("s1", payload) is None


async def accept_question(h: Harness, esc_id: str = "q1") -> None:
    payload = PANE_ESCALATION_ADAPTER.validate_python(
        {
            "kind": "question",
            "escalation_id": esc_id,
            "session_id": "s1",
            "task_context": "ctx-task-value",
            "menu": "## Question 1: question-value\n- label-a: description-a",
            "first_question": "question-value",
            "reason": "reason-value",
            "analysis": escalation().disclosure.model_dump(),
        }
    )
    assert await h.desk.accept_pane("s1", payload) is None


async def accept(h: Harness, esc_id: str = "e1", session: str = "s1") -> None:
    assert await h.desk.accept(session, escalation(esc_id, session)) is None


def retract(esc_id: str, reason: str = "answered") -> EscalationRetractPayload:
    return EscalationRetractPayload(escalation_id=esc_id, reason=reason)


def undelivered(esc_id: str, *, still_live: bool) -> DecisionUndeliveredPayload:
    return DecisionUndeliveredPayload(
        escalation_id=esc_id, detail="detail-value", still_live=still_live
    )


async def test_dispatch_and_clarify_refuse_a_question_naming_the_pane(
    h: Harness,
) -> None:
    record = h.desk.registry.get("s1")
    record.pane_id = "w3:p2"
    h.desk.registry.upsert(record)
    await accept_question(h)
    dispatched = await h.desk.dispatch("q1", "label-a")
    clarified = await h.desk.clarify("q1", "why?")
    for result in (dispatched, clarified):
        assert "an AskUserQuestion menu" in result
        assert "pane w3:p2" in result
    assert dispatched.startswith("decision NOT dispatched")
    assert clarified.startswith("question NOT sent")
    assert h.link.sent == []
    assert h.desk.panes.find("q1") is not None


async def test_dispatch_refuses_permission_pane(h: Harness) -> None:
    await accept(h)
    await accept_permission(h)
    result = await h.desk.dispatch("p1", "yes, go ahead")
    # Refused as a permission prompt, not as a stale decision: the developer
    # is told where the answer belongs.
    assert "NOT dispatched" in result
    assert "permission prompt" in result
    # The registry has no pane for s1 here; the refusal still has to say where
    # the answer belongs rather than go silent.
    assert PANE_UNKNOWN in result
    assert h.link.sent == []
    assert h.desk.panes.find("p1") is not None
    active = h.desk.queue.active
    assert active is not None and active.escalation_id == "e1"
    assert any("NOT dispatched" in t for t in h.notices())


async def test_queued_escalation_surfaces_after_resolve(h: Harness) -> None:
    await accept(h, "e1")
    await accept(h, "e2", "s2")
    assert h.surfaced() == ["e1"]
    assert "dispatched" in await h.desk.dispatch("e1", "use option B")
    # Resolution waits for confirmed delivery: e1 is still the head and
    # nothing new surfaces until the broker confirms it reached the pane.
    assert h.surfaced() == ["e1"]
    assert h.run.notifications() == 1  # one per surfacing, none at accept
    await h.desk.delivered("s1", DecisionDeliveredPayload(escalation_id="e1"))
    assert h.surfaced() == ["e1", "e2"]
    active = h.desk.queue.active
    assert active is not None and active.escalation_id == "e2"
    assert h.run.notifications() == 2


async def test_queued_escalation_surfaces_after_retract(h: Harness) -> None:
    await accept(h, "e1")
    await accept(h, "e2", "s2")
    await h.desk.retract("s1", retract("e1"))
    assert h.surfaced() == ["e1", "e2"]
    active = h.desk.queue.active
    assert active is not None and active.escalation_id == "e2"


async def test_retracted_queued_escalation_is_never_surfaced(h: Harness) -> None:
    await accept(h, "e1")
    await accept(h, "e2", "s2")
    await h.desk.retract("s2", retract("e2", "resolved in pane"))
    await h.desk.retract("s1", retract("e1"))
    # The queued escalation was withdrawn before its turn; announcing it would
    # hand the developer a decision nobody is waiting on.
    assert h.surfaced() == ["e1"]
    assert h.desk.queue.active is None
    assert h.run.notifications() == 1


async def test_dispatch_aborts_on_stale_escalation(h: Harness) -> None:
    # No active escalation at all.
    assert "NOT dispatched" in await h.desk.dispatch("ghost", "option B")
    # Mismatched id while another escalation is active.
    await accept(h)
    assert "NOT dispatched" in await h.desk.dispatch("e2", "option B")
    assert h.link.sent == []
    assert any("NOT dispatched" in t for t in h.notices())


async def test_undelivered_still_live_keeps_escalation_for_redecide(
    h: Harness,
) -> None:
    await accept(h)
    assert "dispatched" in await h.desk.dispatch("e1", "use option B")
    await h.desk.undelivered("s1", undelivered("e1", still_live=True))
    # A failed pane write never resolved e1: it is reported loudly and stays
    # the live head so the developer can dispatch again.
    assert any("did NOT reach" in t and "e1" in t for t in h.notices())
    active = h.desk.queue.active
    assert active is not None and active.escalation_id == "e1"
    # The in-flight lock cleared, so a re-decide dispatches rather than
    # bouncing off a decision that is supposedly still being delivered.
    assert "dispatched" in await h.desk.dispatch("e1", "retry")


async def test_undelivered_stale_retracts_orphaned_entry(h: Harness) -> None:
    await accept(h)
    assert "dispatched" in await h.desk.dispatch("e1", "use option B")
    await h.desk.undelivered("s1", undelivered("e1", still_live=False))
    # The broker no longer holds e1 (e.g. answered before a master restart):
    # the orphaned entry is dropped so the queue cannot wedge.
    assert h.desk.queue.active is None
    assert any("cleared" in t and "e1" in t for t in h.notices())
    # The in-flight lock cleared with it: a fresh escalation dispatches.
    await accept(h, "e2")
    assert "dispatched" in await h.desk.dispatch("e2", "go")


async def test_second_dispatch_refused_while_inflight(h: Harness) -> None:
    await accept(h)
    assert "dispatched" in await h.desk.dispatch("e1", "first")
    # A decision is already on its way to the pane; a second would
    # double-submit the same escalation.
    assert "already being delivered" in await h.desk.dispatch("e1", "second")
    assert len(h.link.sent) == 1


async def test_delivery_reply_before_dispatch_ack_leaves_no_inflight_marker(
    h: Harness,
) -> None:
    async def deliver_then_ack(payload: WireMessage) -> Response:
        assert isinstance(payload, DispatchDecisionPayload)
        await h.desk.delivered(
            "s1", DecisionDeliveredPayload(escalation_id=payload.escalation_id)
        )
        return Response(id="r", ok=True)

    h.link.answer = deliver_then_ack
    await accept(h)
    first = await h.desk.dispatch("e1", "go")
    assert first.startswith("decision dispatched"), first
    assert h.desk.queue.active is None
    # A marker set after the ACK would name the resolved e1 and refuse every
    # later dispatch as already being delivered.
    await accept(h, "e2")
    second = await h.desk.dispatch("e2", "go")
    assert second.startswith("decision dispatched"), second


async def test_dispatch_rejected_by_the_session_clears_inflight(
    h: Harness,
) -> None:
    async def nack(_payload: WireMessage) -> Response:
        return Response(id="r", ok=False, payload={"error": "stale"})

    h.link.answer = nack
    await accept(h)
    result = await h.desk.dispatch("e1", "go")
    assert result.startswith("session s1 rejected the dispatched decision")
    assert result in h.notices()
    h.link.answer = _ack
    assert "dispatched" in await h.desk.dispatch("e1", "retry")


async def test_clarify_relays_answer(h: Harness) -> None:
    async def answer(_payload: WireMessage) -> Response:
        return Response(id="r", ok=True, payload={"answer": "it tried A"})

    h.link.answer = answer
    await accept(h)
    result = await h.desk.clarify("e1", "what did it try?")
    assert h.link.sent == [
        ClarifyEscalationRequestPayload(
            escalation_id="e1", question="what did it try?"
        )
    ]
    assert h.link.timeouts == [CLARIFY_ESCALATION_TIMEOUT_S]
    # The answer reaches the developer verbatim; the tool loop gets only an
    # acknowledgement it cannot paraphrase from.
    assert any("it tried A" in t and "e1" in t for t in h.notices())
    assert "it tried A" not in result
    assert "shown to the developer" in result
    active = h.desk.queue.active
    assert active is not None and active.escalation_id == "e1"


async def test_clarify_wrong_id(h: Harness) -> None:
    assert "NOT sent" in await h.desk.clarify("ghost", "q")
    await accept(h)
    assert "NOT sent" in await h.desk.clarify("e2", "q")
    assert h.link.sent == []
    assert any("NOT sent" in t for t in h.notices())


async def test_clarify_permission_refused(h: Harness) -> None:
    await accept(h)
    await accept_permission(h)
    result = await h.desk.clarify("p1", "q")
    assert "NOT sent" in result
    assert "permission prompt" in result
    assert PANE_UNKNOWN in result
    assert h.link.sent == []
    assert h.desk.panes.find("p1") is not None
    assert any("NOT sent" in t for t in h.notices())


async def test_clarify_broker_nack(h: Harness) -> None:
    async def nack(_payload: WireMessage) -> Response:
        return Response(
            id="r", ok=False, payload={"error": "escalation resolved in the pane"}
        )

    h.link.answer = nack
    await accept(h)
    result = await h.desk.clarify("e1", "q")
    assert result.startswith("no clarification from session s1")
    assert result.endswith(": escalation resolved in the pane")
    assert result in h.notices()
    # The master never resolves on the broker's behalf.
    assert h.desk.queue.active is not None
