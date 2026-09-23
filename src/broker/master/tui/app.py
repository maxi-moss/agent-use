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
from broker.master.permission_escalations import PermissionEscalations
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry
from broker.master.runtime import MasterRuntime
from broker.master.testmode import load_scenario, run_scenario
from broker.master.tui.chat_log import ChatMessage, ThinkingIndicator
from broker.master.tui.fleet import FLEET_WIDTH, FleetSidebar, SessionRowWidget
from broker.master.tui.messages import LLMReply, ViewEventMessage
from broker.master.tui.notice import AttentionNotice
from broker.master.tui.outcome_modal import OutcomeModal
from broker.master.tui.prompt_area import PromptArea
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
    """The master's Textual app: fleet sidebar, attention notice, chat and composer over one MasterRuntime."""

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
        permissions: PermissionEscalations,
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
        # Disclosures are held here, off the chat, until /escalation,
        # /permission or /proposal pastes one in; FleetUpdated prunes what is
        # no longer live.
        self._head: EscalationArrived | None = None
        self._permissions: dict[str, str] = {}
        self._proposals: dict[str, str] = {}
        self._last_view: FleetView | None = None
        self.runtime = MasterRuntime(
            self._emit, registry, queue, permissions, cfg, anchor_pane=anchor_pane
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
            yield FleetSidebar(id="fleet")
            with Vertical():
                notice = AttentionNotice(Text(""), id="notice")
                notice.display = False
                yield notice
                yield VerticalScroll(id="chat")
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
        """Route a submitted line: paste command, scenario command, or master turn."""
        text = message.text.strip()
        if not text:
            return
        box = self.query_one("#box", PromptArea)
        box.clear()
        if text == "/escalation":
            self._show_escalation()
            return
        if text == "/permission" or text.startswith("/permission "):
            self._show_permission(text[len("/permission"):].strip() or None)
            return
        if text == "/proposal" or text.startswith("/proposal "):
            self._show_proposal(text[len("/proposal"):].strip() or None)
            return
        if text == "/outcome" or text.startswith("/outcome "):
            self._show_outcome(text[len("/outcome"):].strip() or None)
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
        """Free the composer when a worker ends and show a failed turn's error."""
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
            self._last_view = event.view
            self._prune_disclosures(event.view)
            self.query_one(FleetSidebar).update_view(event.view)
            self.query_one("#notice", AttentionNotice).update_view(event.view)
        elif isinstance(event, EscalationArrived):
            self._head = event
            self._event_line(
                f"{event.session_id} requested a decision: "
                f"{event.escalation_title}"
            )
        elif isinstance(event, PermissionEscalationArrived):
            self._permissions[event.session_id] = event.rendered
            self._event_line(
                f"permission escalation {event.escalation_id} from "
                f"{event.session_id}"
            )
        elif isinstance(event, ProposalArrived):
            self._proposals[event.session_id] = event.rendered
            self._event_line(
                f"proposal {event.proposal_id} from {event.session_id} "
                "awaiting approval"
            )
        elif isinstance(event, CompletionArrived):
            self._chat_block(
                f"Session {event.session_id} completed:\n"
                f"{event.headline}\n{event.supporting}"
            )
            self._event_line(f"{event.session_id} completed")
        elif isinstance(event, Notice):
            self._event_line(event.text)
        else:
            self._event_line(f"{event.session_id} → {event.state}")

    # ── disclosures on demand (deterministic; no LLM turn) ───────────────────

    def _prune_disclosures(self, view: FleetView) -> None:
        """Drop held disclosures that ``view`` no longer lists as live."""
        head = view.head
        if self._head is not None and (
            head is None or head.escalation_id != self._head.escalation_id
        ):
            self._head = None
        prompting = {p.session_id for p in view.permissions}
        self._permissions = {
            sid: text for sid, text in self._permissions.items() if sid in prompting
        }
        proposing = {r.session_id for r in view.rows if Attention.PROPOSAL in r.badges}
        self._proposals = {
            sid: text for sid, text in self._proposals.items() if sid in proposing
        }

    def _show_escalation(self) -> None:
        """Paste the head escalation's disclosure into the chat, verbatim."""
        if self._head is None:
            self._event_line("no escalation is waiting")
            return
        self._chat_block(self._head.rendered)

    def _show_permission(self, session_id: str | None) -> None:
        """Paste an open permission prompt's disclosure into the chat, verbatim."""
        self._paste_held(
            self._permissions,
            session_id,
            noun="permission prompt",
            command="/permission",
        )

    def _show_proposal(self, session_id: str | None) -> None:
        """Paste a pending proposal into the chat, verbatim."""
        self._paste_held(
            self._proposals, session_id, noun="proposal", command="/proposal"
        )

    def _paste_held(
        self, held: dict[str, str], session_id: str | None, *, noun: str, command: str
    ) -> None:
        """Paste ``session_id``'s held disclosure, or the only one, into the chat.

        Args:
            held: Rendered disclosures by session id.
            session_id: Session named on the command line, if any.
            noun: What the disclosures are, for the activity line on a miss.
            command: The command that names a session, for that same line.
        """
        if session_id is None and len(held) == 1:
            session_id = next(iter(held))
        rendered = held.get(session_id) if session_id else None
        if rendered is None:
            self._event_line(
                f"no single {noun} — use {command} sN"
                if held
                else f"no {noun} is waiting"
            )
            return
        self._chat_block(rendered)

    def _show_outcome(self, session_id: str | None) -> None:
        """Open the read-only outcome modal for a completed/failed session."""
        if session_id is None:
            self._event_line("usage: /outcome sN")
            return
        rows = self._last_view.rows if self._last_view is not None else ()
        row = next((r for r in rows if r.session_id == session_id), None)
        if row is None:
            self._event_line(f"no such session {session_id}")
            return
        if not row.is_settled:
            self._event_line(
                f"{session_id} has no outcome yet (state: {row.state})"
            )
            return
        self.push_screen(OutcomeModal(self.runtime.build_session_outcome(session_id)))

    def on_session_row_widget_selected(
        self, message: SessionRowWidget.Selected
    ) -> None:
        """A settled sidebar row was clicked: open its outcome."""
        self._show_outcome(message.session_id)

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
