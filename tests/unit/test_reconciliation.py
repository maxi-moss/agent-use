"""reconcile_registry classification harness: a real registry, queue and
permission store on disk under /private/tmp, driver.pane_read monkeypatched, the live-broker case
served by a real serve_unix socket."""

import asyncio
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from broker.herdr import driver
from broker.herdr.driver import HerdrError
from broker.master.permission_escalations import PermissionEscalations
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.master.runtime import reconcile_registry
from broker.protocol.constants import SessionState
from broker.protocol.schemas import (
    Envelope,
    EscalationPayload,
    PermissionEscalationPayload,
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
            approved_prompt=approved_prompt,
        )
    )
    return registry


def _queue(home: Path) -> EscalationQueue:
    return EscalationQueue.load(home / "escalation-queue.json")


def _permissions(home: Path) -> PermissionEscalations:
    return PermissionEscalations.load(home / "permission-escalations.json")


def _broker_escalation(esc_id: str, session: str) -> EscalationPayload:
    return EscalationPayload.model_validate(
        {
            "escalation_id": esc_id,
            "session_id": session,
            "task_context": "ctx",
            "escalation_title": "title",
            "situation": "sit",
            "what_was_asked": "asked",
            "what_is_at_stake": "stake",
            "alternatives": [{"option": "a", "pros": "p", "cons": "c"}],
            "recommendation": "rec",
            "uncertainty": "unc",
            "what_would_change_my_mind": "change",
        }
    )


def _permission_escalation(
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


def _fail_pane_read(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    def fake_pane_read(pane_id: str, *, timeout_s: float) -> str:
        raise exc

    monkeypatch.setattr(driver, "pane_read", fake_pane_read)


async def test_live_broker_left_alone(home: Path) -> None:
    registry = _registry_with_s1(home)

    async def handler(env: Envelope) -> Response:
        return Response(id=env.id, ok=True)

    server = await serve_unix(home / "s" / "s1.sock", handler)
    try:
        warnings = await reconcile_registry(registry, _queue(home), _permissions(home))
    finally:
        server.close()
        await server.wait_closed()
    assert len(warnings) == 1
    assert "still answering" in warnings[0]
    assert Registry.load(home / "registry.json").get("s1").state == "driving"


async def test_live_pane_marked_unmanaged(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_s1(home)

    def fake_pane_read(pane_id: str, *, timeout_s: float) -> str:
        return "visible pane text"

    monkeypatch.setattr(driver, "pane_read", fake_pane_read)
    warnings = await reconcile_registry(registry, _queue(home), _permissions(home))
    assert len(warnings) == 1
    assert "unmanaged" in warnings[0]
    assert "attach_session" in warnings[0]
    assert Registry.load(home / "registry.json").get("s1").state == "unmanaged"


async def test_dead_pane_removed_and_retracts(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_s1(home)
    queue = _queue(home)
    queue.accept(_broker_escalation("e1", "s1"))
    permissions = _permissions(home)
    permissions.accept(_permission_escalation("p1", "s1"))
    _fail_pane_read(monkeypatch, HerdrError("pane_not_found", "no such pane"))
    warnings = await reconcile_registry(registry, queue, permissions)
    # A session whose pane is gone is finished: removed here and on reload, so
    # it can never be handed back to the master as a routing candidate.
    assert "s1" not in registry.records
    assert "s1" not in Registry.load(home / "registry.json").records
    # Both kinds retracted from their PERSISTED stores, so nothing the runtime
    # re-announces on startup can belong to a dead session.
    assert _queue(home).depth == 0
    assert _permissions(home).entries == ()
    retraction_lines = [w for w in warnings if "retracted" in w]
    assert any("e1" in w for w in retraction_lines)
    assert any("p1" in w for w in retraction_lines)
    assert len(retraction_lines) == 2
    assert any("gone — removed" in w for w in warnings)


async def test_inconclusive_probe_never_marks_dead(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_s1(home)
    queue = _queue(home)
    queue.accept(_broker_escalation("e1", "s1"))
    permissions = _permissions(home)
    permissions.accept(_permission_escalation("p1", "s1"))
    _fail_pane_read(monkeypatch, HerdrError("unknown", "herdr hiccup"))
    warnings = await reconcile_registry(registry, queue, permissions)
    # A wrong `unmanaged` costs the developer a glance; a wrong `dead` throws
    # away queued decisions and open prompts.
    assert Registry.load(home / "registry.json").get("s1").state == "unmanaged"
    assert _queue(home).depth == 1
    assert len(_permissions(home).entries) == 1
    assert len(warnings) == 1
    assert "herdr hiccup" in warnings[0]


def test_missing_pane_id_marked_unmanaged(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry_with_s1(home, pane_id=None)

    def unexpected_pane_read(pane_id: str, *, timeout_s: float) -> str:
        raise AssertionError("no pane probe should be attempted")

    monkeypatch.setattr(driver, "pane_read", unexpected_pane_read)
    # Run through asyncio.run on a fresh loop, exactly as main() does before
    # the TUI builds its own.
    warnings = asyncio.run(
        reconcile_registry(registry, _queue(home), _permissions(home))
    )
    assert len(warnings) == 1
    # The probe-was-attempted failure would read "inconclusive" here, because
    # the inconclusive branch absorbs the AssertionError.
    assert "never learned its pane" in warnings[0]
    assert "unmanaged" in warnings[0]
    assert Registry.load(home / "registry.json").get("s1").state == "unmanaged"
