"""BrokerMasterApp — the developer's chat TUI.

Structural rules encoded here:
- The socket server is a plain asyncio task: created in on_mount, cancelled
  and awaited in on_unmount (never action_quit — bypassed by App.exit()).
- The LLM turn runs under a worker with an explicit group ("llm"),
  exclusive=True, exit_on_error=False; the prompt box is re-enabled in
  on_worker_state_changed, never at the worker body's end.
- The runtime emits renderer-neutral view events through ``_emit``; the app
  wraps each in a ViewEventMessage and dispatches it to widgets. Prose fields
  are displayed verbatim (thin-master rule).
"""

import asyncio
import contextlib
from pathlib import Path
from typing import cast

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import RichLog
from textual.worker import Worker, WorkerState

from broker.config import BrokerConfig
from broker.llm import LLMCaller, TurnResult
from broker.master.llm import MasterLLM
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry
from broker.master.runtime import MasterRuntime
from broker.master.testmode import load_scenario, run_scenario
from broker.master.tui.chat_log import ChatMessage, ThinkingIndicator
from broker.master.tui.fleet import FLEET_WIDTH, FleetSidebar
from broker.master.tui.messages import LLMReply, ViewEventMessage
from broker.master.tui.prompt_area import PromptArea
from broker.master.tui.surface import FocusedSurface
from broker.master.viewmodel import (
    Attention,
    CompletionArrived,
    EscalationArrived,
    FleetUpdated,
    FleetView,
    Notice,
    PermissionEscalationArrived,
    ProposalArrived,
    ViewEvent,
)


class BrokerMasterApp(App[None]):
    # The fleet pane holds FLEET_WIDTH content cells plus Textual's default
    # vertical scrollbar (2 cells).
    CSS = f"#fleet {{ width: {FLEET_WIDTH + 2}; }}" + """
    #chat { height: 1fr; }
    #activity {
        height: 8;
        border: round $foreground 30%;
        border-title-color: $text-muted;
        background: transparent;
        padding: 0 1;
    }
    #events { height: 1fr; background: transparent; }
    #box { height: 5; }
    """

    def __init__(
        self,
        cfg: BrokerConfig,
        registry: Registry,
        queue: EscalationQueue,
        llm_call: LLMCaller[TurnResult],
        *,
        anchor_pane: str,
        startup_warnings: list[str] | None = None,
        scenarios_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.registry = registry
        self.startup_warnings = list(startup_warnings or [])
        self.scenarios_dir = scenarios_dir
        # In test mode the runtime's events are teed into a capture the
        # scenario runner reads its assertions from, while still reaching the
        # widgets.
        self._scenario_posts: list[ViewEvent] = []
        self._latest_view: FleetView | None = None
        self._focus_session: str | None = None
        self._focus_kind: Attention | None = None
        self._escalations: dict[str, str] = {}
        self._proposals: dict[str, str] = {}
        self.runtime = MasterRuntime(
            self._emit, registry, queue, cfg, anchor_pane=anchor_pane
        )
        self.master_llm = MasterLLM(llm_call, self.runtime, cfg)
        self._server_task: asyncio.Task[None] | None = None

    @property
    def test_mode(self) -> bool:
        """True when the app was given scenarios to drive instead of an LLM."""
        return self.scenarios_dir is not None

    def _emit(self, event: ViewEvent) -> None:
        """Neutral sink handed to the runtime; pump the event through Textual."""
        if self.test_mode:
            self._scenario_posts.append(event)
        self.post_message(ViewEventMessage(event))

    def compose(self) -> ComposeResult:
        with Horizontal():
            with VerticalScroll(id="fleet"):
                yield FleetSidebar(Text("Master — idle"), id="fleet-table")
            with Vertical():
                yield VerticalScroll(id="chat")
                surface = FocusedSurface(id="surface")
                surface.display = False
                yield surface
                with Vertical(id="activity") as activity:
                    activity.border_title = "activity"
                    yield RichLog(id="events", wrap=True)
                yield PromptArea(
                    placeholder=(
                        "task, decision, or question… (ctrl+j for newline)"
                    ),
                    id="box",
                )

    async def on_mount(self) -> None:
        self._server_task = asyncio.create_task(self.runtime.serve())
        for warning in self.startup_warnings:
            self._event_line(f"warning: {warning}")
        self.query_one("#box", PromptArea).focus()

    async def on_unmount(self) -> None:
        if self._server_task is not None:
            self._server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._server_task
            self._server_task = None
        self.registry.save()

    # ── developer input → LLM worker ─────────────────────────────────────────

    def on_prompt_area_submitted(self, message: PromptArea.Submitted) -> None:
        text = message.text.strip()
        if not text:
            return
        box = self.query_one("#box", PromptArea)
        box.clear()
        if text == "/main":
            self._exit_surface()
            return
        if text == "/escalation":
            self._enter_escalation()
            return
        if text == "/approve" or text.startswith("/approve "):
            self._enter_proposal(text[len("/approve"):].strip() or None)
            return
        box.disabled = True
        self._chat_block(text, role="user", label="you")
        if self.test_mode and text.startswith("/"):
            self._handle_command(text)
            return
        self._show_thinking()
        self.run_worker(
            self._master_turn(text),
            group="llm",
            exclusive=True,
            exit_on_error=False,
        )

    def _handle_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        command = parts[0]
        if command != "/inject":
            self._finish_command(
                f"unknown command {command!r}; available: /inject <name> "
                f"where <name> is one of {', '.join(self._scenario_names())}"
            )
            return
        if len(parts) < 2 or not parts[1].strip():
            self._finish_command("usage: /inject <scenario>")
            return
        name = parts[1].strip()
        self._chat_block(
            f"running scenario: {name} "
            "(some scenarios probe a live socket and take a few seconds)"
        )
        self.run_worker(
            self._run_scenario(name),
            group="scenario",
            exclusive=True,
            exit_on_error=False,
        )

    def _finish_command(self, text: str) -> None:
        """Render a command outcome and free the box: no worker will run."""
        self._chat_block(text)
        self.query_one("#box", PromptArea).disabled = False

    def _scenario_names(self) -> list[str]:
        assert self.scenarios_dir is not None  # reached only in test mode
        if not self.scenarios_dir.is_dir():
            return []
        return sorted(p.stem for p in self.scenarios_dir.glob("*.json"))

    async def _run_scenario(self, name: str) -> None:
        assert self.scenarios_dir is not None  # reached only in test mode
        scenario = load_scenario(self.scenarios_dir / f"{name}.json")
        report = await run_scenario(
            self.runtime,
            self._scenario_posts,
            scenario,
            paths=self.runtime.paths,
        )
        for result in report.results:
            status = "PASS" if result.passed else "FAIL"
            self._chat_block(
                f"{status} [{result.index}] {result.op} — {result.detail}"
            )
        passed = sum(1 for r in report.results if r.passed)
        summary = "PASS" if report.passed else "FAIL"
        self._chat_block(
            f"scenario {report.name}: {summary} "
            f"({passed}/{len(report.results)} steps)"
        )

    async def _master_turn(self, text: str) -> None:
        reply = await self.master_llm.handle_developer_message(
            text, on_activity=self.runtime.note_master_activity
        )
        self.post_message(LLMReply(reply))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        worker = cast(
            "Worker[None]",
            event.worker,  # pyright: ignore[reportUnknownMemberType]
        )
        if worker.group not in ("llm", "scenario"):
            return
        if event.state in (
            WorkerState.SUCCESS,
            WorkerState.ERROR,
            WorkerState.CANCELLED,
        ):
            self.query_one("#box", PromptArea).disabled = False
            self._remove_thinking()
            if worker.group == "llm":
                self.runtime.clear_master_activity()
        if event.state is WorkerState.ERROR:
            # Fail loud; the app (and its socket server) survives.
            self._chat_block(f"[master error] {worker.error!r}")

    # ── view events → UI (structured view + verbatim prose) ──────────────────

    def on_llmreply(self, message: LLMReply) -> None:
        # Textual's handler-name derivation collapses the acronym:
        # LLMReply → "on_llmreply", not "on_llm_reply".
        self._chat_block(message.text, role="master", label="master")

    def on_view_event_message(self, message: ViewEventMessage) -> None:
        event = message.event
        if isinstance(event, FleetUpdated):
            self._latest_view = event.view
            self.query_one("#fleet-table", FleetSidebar).update_view(event.view)
            self._maybe_exit_surface()
        elif isinstance(event, EscalationArrived):
            self._chat_block(event.rendered)
            self._event_line(
                f"escalation {event.escalation_id} from {event.session_id}"
            )
            self._escalations[event.session_id] = event.rendered
        elif isinstance(event, PermissionEscalationArrived):
            self._chat_block(event.rendered)
            self._event_line(
                f"permission escalation {event.escalation_id} from "
                f"{event.session_id}"
            )
        elif isinstance(event, ProposalArrived):
            self._chat_block(event.rendered)
            self._event_line(
                f"proposal {event.proposal_id} from {event.session_id} "
                "awaiting approval"
            )
            self._proposals[event.session_id] = event.rendered
        elif isinstance(event, CompletionArrived):
            self._chat_block(
                f"Session {event.session_id} completed:\n{event.summary}"
            )
            self._event_line(f"{event.session_id} completed")
        elif isinstance(event, Notice):
            self._event_line(event.text)
        else:
            self._event_line(f"{event.session_id} → {event.state}")

    # ── focused surfaces (display mode only; the composer stays the master's) ─

    def _enter_escalation(self) -> None:
        """Focus the head escalation's disclosure, if one is waiting."""
        view = self._latest_view
        row = (
            next((r for r in view.rows if Attention.ESCALATION in r.badges), None)
            if view
            else None
        )
        if row is None:
            self._event_line("no escalation is waiting")
            return
        rendered = self._escalations.get(row.session_id)
        if rendered is None:
            self._event_line("escalation disclosure not yet received")
            return
        self._focus_session, self._focus_kind = row.session_id, Attention.ESCALATION
        self.query_one("#surface", FocusedSurface).show(
            f"Session broker · {row.session_id}", rendered
        )
        self._set_mode(surface=True)

    def _enter_proposal(self, session_id: str | None) -> None:
        """Focus a pending proposal: ``session_id``'s, or the only one."""
        view = self._latest_view
        rows = [
            r for r in (view.rows if view else ()) if Attention.PROPOSAL in r.badges
        ]
        row = (
            next((r for r in rows if r.session_id == session_id), None)
            if session_id
            else (rows[0] if len(rows) == 1 else None)
        )
        if row is None:
            self._event_line(
                "no single proposal to approve — use /approve sN"
                if rows
                else "no proposal is waiting"
            )
            return
        rendered = self._proposals.get(row.session_id)
        if rendered is None:
            self._event_line("proposal not yet received")
            return
        self._focus_session, self._focus_kind = row.session_id, Attention.PROPOSAL
        self.query_one("#surface", FocusedSurface).show(
            f"Session broker · {row.session_id}", rendered
        )
        self._set_mode(surface=True)

    def _exit_surface(self) -> None:
        """Return to the master chat."""
        self._focus_session = self._focus_kind = None
        self._set_mode(surface=False)

    def _set_mode(self, *, surface: bool) -> None:
        """Show either the chat or the focused surface, and match the placeholder."""
        self.query_one("#chat", VerticalScroll).display = not surface
        self.query_one("#surface", FocusedSurface).display = surface
        placeholder = "task, decision, or question… (ctrl+j for newline)"
        if surface and self._focus_kind is Attention.ESCALATION:
            placeholder = "answer this escalation via the master…"
        elif surface and self._focus_kind is Attention.PROPOSAL:
            placeholder = "approve or revise this prompt via the master…"
        self.query_one("#box", PromptArea).placeholder = placeholder

    def _maybe_exit_surface(self) -> None:
        """Leave the surface once its badge is gone from the latest fleet view."""
        if self._focus_session is None or self._latest_view is None:
            return
        row = next(
            (r for r in self._latest_view.rows if r.session_id == self._focus_session),
            None,
        )
        if row is None or self._focus_kind not in row.badges:
            self._exit_surface()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _chat_block(
        self, text: str, *, role: str = "system", label: str | None = None
    ) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        # ChatMessage renders the body as rich.Text — no markup interpretation.
        chat.mount(ChatMessage(role, text, label))
        chat.anchor()

    def _show_thinking(self) -> None:
        self._remove_thinking()
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(ThinkingIndicator())
        chat.anchor()

    def _remove_thinking(self) -> None:
        for widget in self.query(ThinkingIndicator):
            widget.remove()

    def _event_line(self, text: str) -> None:
        self.query_one("#events", RichLog).write(Text(text))
