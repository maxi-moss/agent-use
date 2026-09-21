"""BrokerMasterApp under run_test + Pilot, with a real serve_unix socket on
/private/tmp and a scripted fake LLM."""

import asyncio
import json
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)
from rich.text import Text
from textual import events
from textual.containers import VerticalScroll
from textual.widgets import RichLog, Static

from broker.config import BrokerConfig
from broker.llm import TurnResult
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry
from broker.master.tui.app import BrokerMasterApp
from broker.master.tui.prompt_area import PromptArea
from broker.master.tui.surface import FocusedSurface
from broker.master.viewmodel import (
    Attention,
    EscalationArrived,
    FleetUpdated,
    FleetView,
    PermissionEscalationArrived,
    ProposalArrived,
    SessionRow,
)
from broker.protocol import client
from broker.protocol.constants import SessionState
from broker.protocol.schemas import Envelope


class GatedLLM:
    """Blocks each call until released; returns a fixed text turn."""

    def __init__(self, reply: str = "hi", *, gated: bool = False) -> None:
        self.reply = reply
        self.release = asyncio.Event()
        if not gated:
            self.release.set()
        self.calls = 0
        self.fail = False

    async def __call__(
        self,
        *,
        model: str,
        max_tokens: int,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        tools: list[ToolParam],
        tool_choice: ToolChoiceParam,
    ) -> TurnResult:
        self.calls += 1
        await self.release.wait()
        if self.fail:
            raise RuntimeError("llm exploded")
        return TurnResult(text=self.reply)


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        monkeypatch.setenv("BROKER_HOME", td)
        yield Path(td)


def make_app(home: Path, llm: GatedLLM) -> BrokerMasterApp:
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    queue = EscalationQueue.load(home / "escalation-queue.json")
    return BrokerMasterApp(cfg, registry, queue, llm, anchor_pane="%1")


def chat_texts(app: BrokerMasterApp) -> list[str]:
    texts: list[str] = []
    for widget in app.query(Static):
        content = widget.content
        if isinstance(content, Text):
            texts.append(content.plain)
    return texts


def _fleet_text(app: BrokerMasterApp) -> str:
    content = app.query_one("#fleet-table", Static).content
    assert isinstance(content, Text)
    return content.plain


def _surface_heading(app: BrokerMasterApp) -> str:
    content = app.query_one(".surface-heading", Static).content
    assert isinstance(content, Text)
    return content.plain


def _surface_body(app: BrokerMasterApp) -> str:
    content = app.query_one(".surface-body", Static).content
    assert isinstance(content, Text)
    return content.plain


def _events_text(app: BrokerMasterApp) -> str:
    return "\n".join(strip.text for strip in app.query_one("#events", RichLog).lines)


def _session_row(session_id: str, badges: tuple[Attention, ...]) -> SessionRow:
    if Attention.ESCALATION in badges:
        state = SessionState.ESCALATED
    elif Attention.PROPOSAL in badges:
        state = SessionState.AWAITING_APPROVAL
    else:
        state = SessionState.DRIVING
    return SessionRow(
        session_id=session_id,
        state=state,
        title="",
        task_activity="",
        broker_activity="",
        budget_count=0,
        budget_max=8,
        badges=badges,
        pane_id=None,
    )


def _fleet_view(
    rows: tuple[SessionRow, ...],
    *,
    queue_depth: int = 1,
    head_escalation_id: str | None = None,
) -> FleetView:
    return FleetView(
        master_activity=None,
        rows=rows,
        queue_depth=queue_depth,
        waiting=(),
        head_escalation_id=head_escalation_id,
    )


def _escalation_row_view(
    badges: tuple[Attention, ...], *, queue_depth: int = 1
) -> FleetView:
    return _fleet_view(
        (_session_row("s1", badges),),
        queue_depth=queue_depth,
        head_escalation_id="e1" if queue_depth else None,
    )


async def test_submit_disables_input_and_worker_reenables(home: Path) -> None:
    llm = GatedLLM(reply="routing done", gated=True)
    app = make_app(home, llm)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await pilot.click("#box")
        await pilot.press(*"hello")
        await pilot.press("enter")
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        assert box.text == ""
        assert box.disabled is True  # locked while the LLM worker runs
        llm.release.set()
        await pilot.pause(0.1)
        await pilot.pause()  # drain the posted LLMReply
        assert box.disabled is False  # re-enabled by on_worker_state_changed
        assert llm.calls == 1
        assert any("routing done" in t for t in chat_texts(app))


async def test_paste_preserves_all_lines_and_submits_together(home: Path) -> None:
    llm = GatedLLM(reply="ok")
    app = make_app(home, llm)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        box = app.query_one("#box", PromptArea)
        box.focus()
        app.post_message(events.Paste(text="line one\nline two"))
        await pilot.pause()
        assert box.text == "line one\nline two"  # not truncated to the first line
        await pilot.press("enter")
        await pilot.pause()
        assert llm.calls == 1
        assert any("line one\nline two" in t for t in chat_texts(app))


async def test_ctrl_j_inserts_newline_without_submitting(home: Path) -> None:
    llm = GatedLLM(reply="ok")
    app = make_app(home, llm)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await pilot.click("#box")
        await pilot.press(*"line one")
        await pilot.press("ctrl+j")
        await pilot.press(*"line two")
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        assert box.text == "line one\nline two"
        assert llm.calls == 0  # ctrl+j composed a line, it did not submit


async def test_escalation_arrived_renders_exact_string(home: Path) -> None:
    rendered = "Escalation e1 — session s1\n\n## Situation\nverbatim [text]"
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            EscalationArrived("s1", "e1", rendered)
        )
        await pilot.pause()
        assert rendered in chat_texts(app)  # the exact string, unreflowed


async def test_permission_escalation_arrived_renders_exact_string(
    home: Path,
) -> None:
    rendered = (
        "Permission escalation p1 — session s1\n\nanswer it in pane [w3:p2]"
    )
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            PermissionEscalationArrived("s1", "p1", rendered)
        )
        await pilot.pause()
        assert rendered in chat_texts(app)  # the exact string, unreflowed


async def test_llm_worker_error_reenables_input_and_surfaces(
    home: Path,
) -> None:
    llm = GatedLLM(gated=True)
    llm.fail = True
    app = make_app(home, llm)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await pilot.click("#box")
        await pilot.press(*"boom")
        await pilot.press("enter")
        await pilot.pause()
        llm.release.set()
        await pilot.pause(0.1)
        box = app.query_one("#box", PromptArea)
        assert box.disabled is False  # app survived, input usable (fail loud)
        assert any("master error" in t for t in chat_texts(app))


async def test_fleet_sidebar_renders_badges_and_waiting_count(home: Path) -> None:
    app = make_app(home, GatedLLM())
    view = FleetView(
        master_activity=None,
        rows=(
            SessionRow(
                session_id="s1",
                state=SessionState.ESCALATED,
                title="fix the checkout bug",
                task_activity="",
                broker_activity="",
                budget_count=0,
                budget_max=8,
                badges=(Attention.ESCALATION,),
                pane_id=None,
            ),
            SessionRow(
                session_id="s2",
                state=SessionState.DRIVING,
                title="migrate the users table",
                task_activity="",
                broker_activity="",
                budget_count=1,
                budget_max=8,
                badges=(Attention.PERMISSION,),
                pane_id="w3:p2",
            ),
        ),
        queue_depth=1,
        waiting=(),
        head_escalation_id="e1",
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(FleetUpdated(view))  # pyright: ignore[reportPrivateUsage]
        await pilot.pause()
        text = _fleet_text(app)
        assert "s1" in text
        assert "s2" in text
        assert "needs decision" in text
        assert "permission · w3:p2" in text
        assert "1 request waiting" in text
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(FleetView(None, (), 0, (), None))
        )
        await pilot.pause()
        assert "all clear" in _fleet_text(app)


ESCALATION_RENDERED = "Escalation e1 — session s1\n\n## Situation\nverbatim [text]"
DEFAULT_PLACEHOLDER = "task, decision, or question… (ctrl+j for newline)"


async def test_slash_escalation_focuses_and_slash_main_restores(
    home: Path,
) -> None:
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            EscalationArrived("s1", "e1", ESCALATION_RENDERED)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_escalation_row_view((Attention.ESCALATION,)))
        )
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/escalation"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#chat", VerticalScroll).display is False
        assert app.query_one("#surface", FocusedSurface).display is True
        assert _surface_heading(app) == "Session broker · s1"
        assert _surface_body(app) == ESCALATION_RENDERED
        assert box.placeholder == "answer this escalation via the master…"
        assert box.disabled is False  # no worker ran for a slash command
        box.focus()
        box.text = "/main"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#chat", VerticalScroll).display is True
        assert app.query_one("#surface", FocusedSurface).display is False
        assert box.placeholder == DEFAULT_PLACEHOLDER


async def test_typing_in_the_surface_still_runs_the_llm_worker(home: Path) -> None:
    llm = GatedLLM(reply="dispatched", gated=True)
    app = make_app(home, llm)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            EscalationArrived("s1", "e1", ESCALATION_RENDERED)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_escalation_row_view((Attention.ESCALATION,)))
        )
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/escalation"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#surface", FocusedSurface).display is True
        await pilot.click("#box")
        await pilot.press(*"use option B")
        await pilot.press("enter")
        await pilot.pause()
        assert box.disabled is True  # locked while the LLM worker runs
        llm.release.set()
        await pilot.pause(0.1)
        await pilot.pause()  # drain the posted LLMReply
        assert box.disabled is False
        assert llm.calls == 1  # same routing as a plain chat submission


async def test_fleet_updated_auto_exits_the_surface_when_the_badge_clears(
    home: Path,
) -> None:
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            EscalationArrived("s1", "e1", ESCALATION_RENDERED)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_escalation_row_view((Attention.ESCALATION,)))
        )
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/escalation"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#surface", FocusedSurface).display is True
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_escalation_row_view((), queue_depth=0))
        )
        await pilot.pause()
        assert app.query_one("#chat", VerticalScroll).display is True
        assert app.query_one("#surface", FocusedSurface).display is False


async def test_slash_escalation_opens_the_head_not_the_lowest_badged_row(
    home: Path,
) -> None:
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        # s3 escalated first and is the announced head; s1 is queued behind it
        # and carries the badge without a disclosure of its own.
        app._emit(  # pyright: ignore[reportPrivateUsage]
            EscalationArrived("s3", "e3", ESCALATION_RENDERED)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(
                _fleet_view(
                    (
                        _session_row("s1", (Attention.ESCALATION,)),
                        _session_row("s3", (Attention.ESCALATION,)),
                    ),
                    queue_depth=2,
                    head_escalation_id="e3",
                )
            )
        )
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/escalation"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#surface", FocusedSurface).display is True
        assert _surface_heading(app) == "Session broker · s3"
        assert _surface_body(app) == ESCALATION_RENDERED


async def test_slash_escalation_with_nothing_waiting_stays_on_chat(
    home: Path,
) -> None:
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/escalation"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#surface", FocusedSurface).display is False
        assert app.query_one("#chat", VerticalScroll).display is True
        assert "no escalation is waiting" in _events_text(app)


PROPOSAL_RENDERED = "Prompt proposal p1 — session s1\n\n## Proposed prompt\nverbatim [text]"


async def test_slash_proposal_focuses_the_single_proposal(home: Path) -> None:
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            ProposalArrived("s1", "p1", PROPOSAL_RENDERED)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_fleet_view((_session_row("s1", (Attention.PROPOSAL,)),)))
        )
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/proposal"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#chat", VerticalScroll).display is False
        assert app.query_one("#surface", FocusedSurface).display is True
        assert _surface_heading(app) == "Session broker · s1"
        assert _surface_body(app) == PROPOSAL_RENDERED
        assert box.placeholder == "approve or revise this prompt via the master…"
        assert box.disabled is False  # no worker ran for a slash command


async def test_slash_proposal_disambiguates_between_multiple_proposals(
    home: Path,
) -> None:
    app = make_app(home, GatedLLM())
    s1_rendered = "Prompt proposal p1 — session s1\nverbatim [text one]"
    s2_rendered = "Prompt proposal p2 — session s2\nverbatim [text two]"
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            ProposalArrived("s1", "p1", s1_rendered)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            ProposalArrived("s2", "p2", s2_rendered)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(
                _fleet_view(
                    (
                        _session_row("s1", (Attention.PROPOSAL,)),
                        _session_row("s2", (Attention.PROPOSAL,)),
                    )
                )
            )
        )
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/proposal"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#surface", FocusedSurface).display is False
        assert (
            "no single proposal to focus — use /proposal sN" in _events_text(app)
        )
        box.focus()
        box.text = "/proposal s2"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#surface", FocusedSurface).display is True
        assert _surface_heading(app) == "Session broker · s2"
        assert _surface_body(app) == s2_rendered


async def test_typing_approve_in_the_proposal_surface_runs_the_llm_worker(
    home: Path,
) -> None:
    llm = GatedLLM(reply="approved", gated=True)
    app = make_app(home, llm)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            ProposalArrived("s1", "p1", PROPOSAL_RENDERED)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_fleet_view((_session_row("s1", (Attention.PROPOSAL,)),)))
        )
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/proposal"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#surface", FocusedSurface).display is True
        await pilot.click("#box")
        await pilot.press(*"approve")
        await pilot.press("enter")
        await pilot.pause()
        assert box.disabled is True  # locked while the LLM worker runs
        llm.release.set()
        await pilot.pause(0.1)
        await pilot.pause()  # drain the posted LLMReply
        assert box.disabled is False
        assert llm.calls == 1  # same routing as a plain chat submission


async def test_fleet_updated_auto_exits_the_proposal_surface_when_badge_clears(
    home: Path,
) -> None:
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            ProposalArrived("s1", "p1", PROPOSAL_RENDERED)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_fleet_view((_session_row("s1", (Attention.PROPOSAL,)),)))
        )
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/proposal"
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one("#surface", FocusedSurface).display is True
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_fleet_view((_session_row("s1", ()),), queue_depth=0))
        )
        await pilot.pause()
        assert app.query_one("#chat", VerticalScroll).display is True
        assert app.query_one("#surface", FocusedSurface).display is False


async def test_inject_runs_a_scenario_end_to_end(home: Path) -> None:
    scenarios_dir = home / "scenarios"
    scenarios_dir.mkdir()
    (scenarios_dir / "smoke.json").write_text(
        json.dumps(
            {
                "name": "smoke",
                "steps": [
                    {"op": "seed_session", "name": "s1"},
                    {"op": "start_socket", "session": "s1"},
                    {
                        "op": "escalate",
                        "session": "s1",
                        "escalation_id": "e1",
                        "expect": "ack",
                    },
                    {"op": "assert_surfaced", "escalation_id": "e1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    queue = EscalationQueue.load(home / "escalation-queue.json")
    app = BrokerMasterApp(
        cfg,
        registry,
        queue,
        GatedLLM(),
        anchor_pane="%1",
        scenarios_dir=scenarios_dir,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.1)  # let the master socket bind
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/inject smoke"
        await pilot.press("enter")
        # The scenario drives real escalations over the socket; a single
        # pause is CPU-idle detection, not a drain, so poll for the summary.
        for _ in range(200):
            await pilot.pause(0.05)
            if any(
                "scenario smoke: PASS" in t for t in chat_texts(app)
            ):
                break
        else:
            raise AssertionError(
                f"scenario summary never rendered; chat: {chat_texts(app)}"
            )
        # The escalation itself surfaced into the chat, one block, verbatim.
        assert any("Escalation e1 — session s1" in t for t in chat_texts(app))
        assert app.query_one("#box", PromptArea).disabled is False


async def test_fleet_panel_updates_and_worker_clears_master_activity(
    home: Path,
) -> None:
    llm = GatedLLM(reply="ok", gated=True)
    app = make_app(home, llm)
    async with app.run_test() as pilot:
        # serve() publishes the initial snapshot once the socket is up.
        for _ in range(100):
            await pilot.pause(0.05)
            if "Master — idle" in _fleet_text(app):
                break
        else:
            raise AssertionError(f"no initial fleet render: {_fleet_text(app)!r}")
        await pilot.click("#box")
        await pilot.press(*"hello")
        await pilot.press("enter")
        # The turn is gated inside the LLM call: the header shows thinking.
        for _ in range(100):
            await pilot.pause(0.05)
            if "Master — thinking…" in _fleet_text(app):
                break
        else:
            raise AssertionError(f"no thinking header: {_fleet_text(app)!r}")
        llm.release.set()
        # Worker finish clears the master activity back to idle.
        for _ in range(100):
            await pilot.pause(0.05)
            if "Master — idle" in _fleet_text(app):
                break
        else:
            raise AssertionError(f"header never cleared: {_fleet_text(app)!r}")


async def test_app_mounts_serves_and_unmounts_cleanly(home: Path) -> None:
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        socket_path = app.runtime.master_socket_path
        assert socket_path.exists()
        # The real server answers on the socket while the TUI runs.
        env = Envelope(
            id=uuid.uuid4().hex, type="nonsense", payload={}
        )
        resp = await client.request(socket_path, env, timeout_s=5.0)
        assert resp.ok is False
    # Clean unmount: registry persisted, no exception raised on the way out.
    assert (home / "registry.json").exists()
