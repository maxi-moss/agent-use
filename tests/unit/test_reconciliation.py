"""reconcile_registry classification harness: a real registry, queue and
pane store on disk under /private/tmp, driver.agent_running monkeypatched, the
live-broker case served by a real serve_unix socket."""

import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from broker.herdr import driver
from broker.herdr.driver import HerdrError
from broker.master.pane_escalations import PaneEscalations
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.master.runtime import reconcile_registry
from broker.protocol.constants import SessionState
from broker.protocol.schemas import (
    Envelope,
    EscalationPayload,
    PermissionEscalationPayload,
    QuestionEscalationPayload,
    Response,
)
from broker.protocol.server import serve_unix


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        monkeypatch.setenv("BROKER_HOME", td)
        yield Path(td)


def _registry_with_s1(
    home: Path,
    *,
    pane_id: str | None = "w1:p1",
    approved_prompt: str | None = "the first task",
) -> Registry:
    registry = Registry.load(home / "registry.json")
    registry.upsert(
        SessionRecord(
            name="s1",
            socket_path=str(home / "s" / "s1.sock"),
            cwd="/private/tmp",
            anchor_pane="%1",
            state=SessionState.DRIVING,
            pane_id=pane_id,
            claude_session_id="cc-1",
            transcript_path="/private/tmp/cc-1.jsonl",
            approved_prompt=approved_prompt,
        )
    )
    return registry


def _queue(home: Path) -> EscalationQueue:
    return EscalationQueue.load(home / "escalation-queue.json")


def _panes(home: Path) -> PaneEscalations:
    return PaneEscalations.load(home / "pane-escalations.json")


def _broker_escalation(esc_id: str, session: str) -> EscalationPayload:
    return EscalationPayload.model_validate(
        {
            "escalation_id": esc_id,
            "session_id": session,
            "task_context": "ctx",
            "disclosure": {
                "escalation_title": "title",
                "situation": "sit",
                "what_was_asked": "asked",
                "what_is_at_stake": "stake",
                "alternatives": [{"option": "a", "pros": "p", "cons": "c"}],
                "recommendation": "rec",
                "uncertainty": "unc",
                "what_would_change_my_mind": "change",
            },
        }
    )


def _permission_pane(
    esc_id: str, session: str
) -> PermissionEscalationPayload:
    return PermissionEscalationPayload.model_validate(
        {
            "escalation_id": esc_id,
            "session_id": session,
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
            "task_intent": "intent",
            "reason": "reason",
            "raised_at": "2026-08-06T12:00:00+00:00",
        }
    )


def _question_escalation(esc_id: str, session: str) -> QuestionEscalationPayload:
    return QuestionEscalationPayload(
        escalation_id=esc_id,
        session_id=session,
        task_context="ctx",
        menu="",
        first_question="",
        reason="reason",
    )


def _agent_running(monkeypatch: pytest.MonkeyPatch, running: bool) -> None:
    def fake_agent_running(name: str, *, timeout_s: float) -> bool:
        return running

    monkeypatch.setattr(driver, "agent_running", fake_agent_running)


def _agent_probe_fails(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def fake_agent_running(name: str, *, timeout_s: float) -> bool:
        raise exc

    monkeypatch.setattr(driver, "agent_running", fake_agent_running)


def _seed_escalations(home: Path) -> tuple[EscalationQueue, PaneEscalations]:
    queue = _queue(home)
    queue.accept(_broker_escalation("e1", "s1"))
    panes = _panes(home)
    panes.accept(_permission_pane("p1", "s1"))
    panes.accept(_question_escalation("q1", "s1"))
    return queue, panes


def _assert_dropped(home: Path, registry: Registry, warnings: list[str]) -> None:
    # Removed here and on reload, so it can never be handed back to the
    # master as a routing candidate.
    assert "s1" not in registry.records
    assert "s1" not in Registry.load(home / "registry.json").records
    # Every kind retracted from its PERSISTED store, so nothing the runtime
    # re-announces on startup can belong to a dropped session.
    assert _queue(home).depth == 0
    assert _panes(home).entries == ()
    retraction_lines = [w for w in warnings if "retracted" in w]
    assert len(retraction_lines) == 3
    for esc_id in ("e1", "p1", "q1"):
        assert any(esc_id in w for w in retraction_lines)


async def test_live_broker_left_alone(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_s1(home)
    _agent_probe_fails(monkeypatch, AssertionError("a live broker is not probed"))

    async def handler(env: Envelope) -> Response:
        return Response(id=env.id, ok=True)

    server = await serve_unix(home / "s" / "s1.sock", handler)
    try:
        warnings = await reconcile_registry(registry, _queue(home), _panes(home))
    finally:
        server.close()
        await server.wait_closed()
    assert len(warnings) == 1
    assert "still answering" in warnings[0]
    assert Registry.load(home / "registry.json").get("s1").state == "driving"


async def test_live_claude_marked_unmanaged(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_s1(home)
    _agent_running(monkeypatch, True)
    warnings = await reconcile_registry(registry, _queue(home), _panes(home))
    assert len(warnings) == 1
    assert "unmanaged" in warnings[0]
    assert "attach_session" in warnings[0]
    assert Registry.load(home / "registry.json").get("s1").state == "unmanaged"


async def test_exited_claude_removed_and_retracts(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_s1(home)
    queue, panes = _seed_escalations(home)
    _agent_running(monkeypatch, False)
    warnings = await reconcile_registry(registry, queue, panes)
    _assert_dropped(home, registry, warnings)
    assert any("no longer runs" in w for w in warnings)


async def test_inconclusive_probe_never_removes(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_s1(home)
    queue = _queue(home)
    queue.accept(_broker_escalation("e1", "s1"))
    panes = _panes(home)
    panes.accept(_permission_pane("p1", "s1"))
    _agent_probe_fails(monkeypatch, HerdrError("unknown", "herdr hiccup"))
    warnings = await reconcile_registry(registry, queue, panes)
    # A wrong `unmanaged` costs the developer a glance; a wrong removal throws
    # away queued decisions and open prompts.
    assert Registry.load(home / "registry.json").get("s1").state == "unmanaged"
    assert _queue(home).depth == 1
    assert len(_panes(home).entries) == 1
    assert len(warnings) == 1
    assert "herdr hiccup" in warnings[0]


def test_live_claude_without_adoption_fields_removed(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_s1(home, pane_id=None)
    queue, panes = _seed_escalations(home)
    _agent_running(monkeypatch, True)
    # Run through asyncio.run on a fresh loop, exactly as main() does before
    # the TUI builds its own.
    warnings = asyncio.run(reconcile_registry(registry, queue, panes))
    # Neither attach_session nor reassign_session can ever take it over, so
    # keeping it only turns every recovery attempt into the same refusal.
    _assert_dropped(home, registry, warnings)
    assert any("no broker can take it over" in w and "pane_id" in w for w in warnings)
