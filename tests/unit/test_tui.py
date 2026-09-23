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

from broker import decision_log
from broker.decision_log import DecisionKind
from broker.config import BrokerConfig
from broker.llm import TurnResult
from broker.master.permission_escalations import PermissionEscalations
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.master.tui.app import BrokerMasterApp
from broker.master.tui.fleet import FleetSidebar, SessionRowWidget
from broker.master.tui.notice import AttentionNotice
from broker.master.tui.outcome_modal import OutcomeModal
from broker.master.tui.prompt_area import PromptArea
from broker.master.viewmodel import (
    Attention,
    EscalationArrived,
    FleetUpdated,
    FleetView,
    HeadRequest,
    PermissionEscalationArrived,
    PermissionRequest,
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
    permissions = PermissionEscalations.load(home / "permission-escalations.json")
    return BrokerMasterApp(cfg, registry, queue, permissions, llm, anchor_pane="%1")


def chat_texts(app: BrokerMasterApp) -> list[str]:
    texts: list[str] = []
    for widget in app.query(Static):
        content = widget.content
        if isinstance(content, Text):
            texts.append(content.plain)
    return texts


def _fleet_text(app: BrokerMasterApp) -> str:
    parts: list[str] = []
    for widget in app.query_one(FleetSidebar).query(Static):
        content = widget.content
        if isinstance(content, Text):
            parts.append(content.plain)
    return "\n".join(parts)


def _chat_only_texts(app: BrokerMasterApp) -> list[str]:
    texts: list[str] = []
    for widget in app.query_one("#chat", VerticalScroll).query(Static):
        content = widget.content
        if isinstance(content, Text):
            texts.append(content.plain)
    return texts


def _event_texts(app: BrokerMasterApp) -> list[str]:
    return [strip.text for strip in app.query_one("#events", RichLog).lines]


def _notice_text(app: BrokerMasterApp) -> str:
    content = app.query_one("#notice", AttentionNotice).content
    assert isinstance(content, Text)
    return content.plain


def _session_row(
    session_id: str, badges: tuple[Attention, ...], *, pane_id: str | None = None
) -> SessionRow:
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
        pane_id=pane_id,
    )


def _fleet_view(
    rows: tuple[SessionRow, ...],
    *,
    queue_depth: int = 0,
    head: HeadRequest | None = None,
    permissions: tuple[PermissionRequest, ...] = (),
) -> FleetView:
    return FleetView(
        master_activity=None,
        rows=rows,
        queue_depth=queue_depth,
        waiting=(),
        head=head,
        permissions=permissions,
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


async def test_slash_escalation_pastes_the_head_disclosure_verbatim(
    home: Path,
) -> None:
    rendered = "Escalation e1 — session s1\n\n## Situation\nverbatim [text]"
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            EscalationArrived("s1", "e1", "Queue policy", rendered)
        )
        await pilot.pause()
        assert rendered not in _chat_only_texts(app)  # arrival stays off the chat
        assert "s1 requested a decision: Queue policy" in _event_texts(app)
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/escalation"
        await pilot.press("enter")
        await pilot.pause()
        assert rendered in _chat_only_texts(app)  # the exact string, unreflowed
        assert box.disabled is False  # no LLM turn ran


async def test_slash_escalation_forgets_a_head_the_fleet_no_longer_names(
    home: Path,
) -> None:
    rendered = "Escalation e1 — session s1\n\nverbatim [text]"
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            EscalationArrived("s1", "e1", "Queue policy", rendered)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_fleet_view((_session_row("s1", ()),)))
        )
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/escalation"
        await pilot.press("enter")
        await pilot.pause()
        assert rendered not in _chat_only_texts(app)


async def test_slash_permission_pastes_one_prompt_and_disambiguates(
    home: Path,
) -> None:
    s1_rendered = "Permission escalation p1 — session s1\n\nanswer it in pane [w3:p2]"
    s2_rendered = "Permission escalation p2 — session s2\n\nanswer it in pane [w3:p4]"
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            PermissionEscalationArrived("s1", "p1", s1_rendered)
        )
        await pilot.pause()
        assert s1_rendered not in _chat_only_texts(app)  # arrival stays off the chat
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/escalation"
        await pilot.press("enter")
        await pilot.pause()
        # A permission prompt is not a decision: /escalation never shows it.
        assert s1_rendered not in _chat_only_texts(app)
        box.focus()
        box.text = "/permission"
        await pilot.press("enter")
        await pilot.pause()
        assert s1_rendered in _chat_only_texts(app)  # the only one: no sN needed
        app._emit(  # pyright: ignore[reportPrivateUsage]
            PermissionEscalationArrived("s2", "p2", s2_rendered)
        )
        await pilot.pause()
        box.focus()
        box.text = "/permission"
        await pilot.press("enter")
        await pilot.pause()
        assert s2_rendered not in _chat_only_texts(app)  # two open: ambiguous
        box.focus()
        box.text = "/permission s2"
        await pilot.press("enter")
        await pilot.pause()
        assert s2_rendered in _chat_only_texts(app)


async def test_slash_permission_forgets_a_prompt_the_fleet_no_longer_names(
    home: Path,
) -> None:
    rendered = "Permission escalation p1 — session s1\n\nanswer it in pane [w3:p2]"
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            PermissionEscalationArrived("s1", "p1", rendered)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_fleet_view((_session_row("s1", ()),)))
        )
        await pilot.pause()
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/permission s1"
        await pilot.press("enter")
        await pilot.pause()
        assert rendered not in _chat_only_texts(app)


async def test_slash_proposal_pastes_one_proposal_and_disambiguates(
    home: Path,
) -> None:
    s1_rendered = "Prompt proposal p1 — session s1\nverbatim [text one]"
    s2_rendered = "Prompt proposal p2 — session s2\nverbatim [text two]"
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            ProposalArrived("s1", "p1", s1_rendered)
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            ProposalArrived("s2", "p2", s2_rendered)
        )
        await pilot.pause()
        assert s1_rendered not in _chat_only_texts(app)
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/proposal"
        await pilot.press("enter")
        await pilot.pause()
        assert s1_rendered not in _chat_only_texts(app)  # two waiting: ambiguous
        box.focus()
        box.text = "/proposal s2"
        await pilot.press("enter")
        await pilot.pause()
        assert s2_rendered in _chat_only_texts(app)
        assert s1_rendered not in _chat_only_texts(app)


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
        head=HeadRequest("s1", "e1", "ship it?"),
        permissions=(PermissionRequest("s2", "p2", "Bash"),),
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
            FleetUpdated(FleetView(None, (), 0, (), None, ()))
        )
        await pilot.pause()
        assert "all clear" in _fleet_text(app)


async def test_notice_names_the_head_and_each_pending_proposal(
    home: Path,
) -> None:
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        assert app.query_one("#notice", AttentionNotice).display is False
        # s2 is the announced head; s4 is queued behind it with no disclosure
        # of its own; s3 awaits prompt approval outside the queue.
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(
                _fleet_view(
                    (
                        _session_row("s2", (Attention.ESCALATION,)),
                        _session_row("s3", (Attention.PROPOSAL,)),
                        _session_row("s4", (Attention.ESCALATION,)),
                    ),
                    queue_depth=2,
                    head=HeadRequest("s2", "e2", "retry or fail loud?"),
                )
            )
        )
        await pilot.pause()
        assert app.query_one("#notice", AttentionNotice).display is True
        text = _notice_text(app)
        assert "2 requests waiting · s2 needs a decision: retry or fail loud?" in text
        assert "s3 waiting for opening-prompt approval" in text
        assert "s4" not in text  # only the head is asked about
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_fleet_view((_session_row("s2", ()),)))
        )
        await pilot.pause()
        assert app.query_one("#notice", AttentionNotice).display is False


async def test_notice_lists_every_open_prompt_apart_from_the_count(
    home: Path,
) -> None:
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        rows = (
            _session_row("s1", (Attention.PERMISSION,), pane_id="w3:p2"),
            _session_row("s2", (Attention.ESCALATION,)),
            _session_row("s5", (Attention.PERMISSION,)),
        )
        prompts = (
            PermissionRequest("s1", "p1", "Bash"),
            PermissionRequest("s5", "p5", "Edit"),
        )
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(
                _fleet_view(
                    rows,
                    queue_depth=1,
                    head=HeadRequest("s2", "e2", "retry or fail loud?"),
                    permissions=prompts,
                )
            )
        )
        await pilot.pause()
        text = _notice_text(app)
        # The count is decisions only; each open prompt is its own line.
        assert "1 request waiting · s2 needs a decision: retry or fail loud?" in text
        assert (
            "⚠ s1 permission prompt for Bash — answer it in pane w3:p2; "
            "/permission s1 shows why"
        ) in text
        assert "⚠ s5 permission prompt for Edit — answer it in its pane" in text
        assert "1 request waiting" in _fleet_text(app)
        # With no decision waiting the prompts still show, and nothing counts
        # them as waiting requests.
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_fleet_view(rows, permissions=prompts))
        )
        await pilot.pause()
        text = _notice_text(app)
        assert app.query_one("#notice", AttentionNotice).display is True
        assert "s1 permission prompt for Bash" in text
        assert "waiting ·" not in text
        assert "all clear" in _fleet_text(app)


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
    permissions = PermissionEscalations.load(home / "permission-escalations.json")
    app = BrokerMasterApp(
        cfg,
        registry,
        queue,
        permissions,
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
        # The escalation surfaced into the notice, not the chat.
        assert "s1 needs a decision: what was asked in e1" in _notice_text(app)
        assert not any("Escalation e1" in t for t in _chat_only_texts(app))
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


async def test_slash_outcome_opens_modal_from_the_decision_log(home: Path) -> None:
    app = make_app(home, GatedLLM())
    app.registry.upsert(
        SessionRecord(
            name="s1",
            socket_path=str(home / "s" / "s1.sock"),
            cwd="/private/tmp",
            anchor_pane="%1",
            state=SessionState.COMPLETED,
            title="Attach recovery",
        )
    )
    log = app.runtime.paths.session_decisions("s1")
    decision_log.append(
        log,
        kind=DecisionKind.ANSWERED,
        reasoning="r",
        detail="use uv",
        task_summary="Chose uv",
    )
    decision_log.append(
        log,
        kind=DecisionKind.ESCALATION_RAISED,
        reasoning="r",
        detail="drop the column?",
        task_summary="Asked before dropping a column",
        escalation_id="e1",
        what_was_asked="drop users.legacy?",
    )
    decision_log.append(
        log,
        kind=DecisionKind.DISPATCHED,
        reasoning="d",
        detail="yes, drop it",
        escalation_id="e1",
    )
    decision_log.append(
        log,
        kind=DecisionKind.ANSWERED,
        reasoning="r",
        detail="dropped",
        task_summary="Dropped it",
    )
    decision_log.append(
        log,
        kind=DecisionKind.COMPLETED,
        reasoning="r",
        detail="",
        task_summary="Wrapped up",
        headline="Recovered the session after broker loss",
        supporting="The budget survived the restart",
    )
    completed = SessionRow(
        session_id="s1",
        state=SessionState.COMPLETED,
        title="Attach recovery",
        task_activity="",
        broker_activity="",
        budget_count=2,
        budget_max=8,
        badges=(),
        pane_id=None,
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app._emit(  # pyright: ignore[reportPrivateUsage]
            FleetUpdated(_fleet_view((completed, _session_row("s2", ()))))
        )
        await pilot.pause()
        # The settled row shows the View outcome affordance; the working one does not.
        s1_row = next(
            w for w in app.query(SessionRowWidget) if w.session_id == "s1"
        )
        s2_row = next(
            w for w in app.query(SessionRowWidget) if w.session_id == "s2"
        )
        s1_content = s1_row.content
        assert isinstance(s1_content, Text)
        assert "View outcome" in s1_content.plain
        s2_content = s2_row.content
        assert isinstance(s2_content, Text)
        assert "View outcome" not in s2_content.plain
        # Clicking a working row does nothing; clicking the settled one opens it.
        await pilot.click(s2_row)
        await pilot.pause()
        assert len(app.screen_stack) == 1
        await pilot.click(s1_row)
        await pilot.pause()
        assert isinstance(app.screen, OutcomeModal)
        assert len(app.screen_stack) == 2
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.screen_stack) == 1
        # The /outcome command still opens the same modal.
        box = app.query_one("#box", PromptArea)
        box.focus()
        box.text = "/outcome s1"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, OutcomeModal)
        assert len(app.screen_stack) == 2
        headline = app.screen.query_one("#headline", Static).content
        assert isinstance(headline, Text)
        assert headline.plain == "Recovered the session after broker loss"
        history = "\n".join(
            content.plain
            for widget in app.screen.query(Static)
            if isinstance(content := widget.content, Text)
        )
        assert "Chose uv" in history
        assert "Reason: Asked before dropping a column" in history
        assert "Solution: Dropped it" in history
        assert "yes, drop it" not in history  # raw developer words never shown
        assert "Task completed" in history
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.screen_stack) == 1
        # Gate paths: unknown or still-working sessions get an event line, no modal.
        box.focus()
        box.text = "/outcome s404"
        await pilot.press("enter")
        await pilot.pause()
        box.focus()
        box.text = "/outcome s2"
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.screen_stack) == 1
        lines = _event_texts(app)
        assert any("no such session s404" in line for line in lines)
        assert any("s2 has no outcome yet" in line for line in lines)
