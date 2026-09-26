"""BrokerMasterApp — the developer's chat TUI.

Structural rules encoded here:
- The runtime is started in on_mount and closed in on_unmount (never
  action_quit — bypassed by App.exit()).
- The LLM turn runs under a worker with an explicit group ("llm"),
  exclusive=True, exit_on_error=False; the prompt box is re-enabled in
  on_worker_state_changed, never at the worker body's end.
- The runtime emits renderer-neutral view events through the relay; the app
  wraps each in a ViewEventMessage and dispatches it to widgets. Prose fields
  are displayed verbatim (thin-master rule).
"""

from typing import cast

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import RichLog
from textual.worker import Worker, WorkerState

from broker.master.llm import MasterLLM
from broker.master.runtime import MasterRuntime
from broker.master.testmode import InjectCommand
from broker.master.tui.chat_log import ChatMessage, ThinkingIndicator
from broker.master.tui.fleet import FLEET_WIDTH, FleetSidebar, SessionRowWidget
from broker.master.tui.messages import LLMReply, ViewEventMessage
from broker.master.tui.notice import AttentionNotice
from broker.master.tui.outcome_modal import OutcomeModal
from broker.master.tui.prompt_area import PromptArea
from broker.master.viewmodel import (
    CompletionArrived,
    EscalationArrived,
    FleetUpdated,
    Notice,
    PaneEscalationArrived,
    ProposalArrived,
    ViewEvent,
    ViewEventRelay,
)
from broker.protocol.constants import PaneKind


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
        runtime: MasterRuntime,
        master_llm: MasterLLM,
        relay: ViewEventRelay,
        *,
        startup_warnings: list[str],
        inject: InjectCommand | None,
    ) -> None:
        """Attach the app to a built runtime and its master LLM.

        Args:
            runtime: The master runtime, not yet started.
            master_llm: Runs the developer's turns.
            relay: The runtime's event sink; the app connects its widgets to it.
            startup_warnings: Lines shown in the activity log on mount.
            inject: Test-mode command handler for ``/`` lines, or None.
        """
        super().__init__()
        self.runtime = runtime
        self.master_llm = master_llm
        self.startup_warnings = startup_warnings
        self.inject = inject
        relay.connect(self._post_view_event)

    def _post_view_event(self, event: ViewEvent) -> None:
        """Pump one runtime view event through Textual."""
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
        self.runtime.start()
        for warning in self.startup_warnings:
            self._event_line(f"warning: {warning}")
        self.query_one("#box", PromptArea).focus()

    async def on_unmount(self) -> None:
        await self.runtime.aclose()

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
        matched, session_id = self._parse_command(text, "/permission")
        if matched:
            self._show_pane(
                PaneKind.PERMISSION,
                session_id,
                noun="permission prompt",
                command="/permission",
            )
            return
        matched, session_id = self._parse_command(text, "/question")
        if matched:
            self._show_pane(
                PaneKind.QUESTION,
                session_id,
                noun="question",
                command="/question",
            )
            return
        matched, session_id = self._parse_command(text, "/proposal")
        if matched:
            self._show_proposal(session_id)
            return
        matched, session_id = self._parse_command(text, "/outcome")
        if matched:
            self._show_outcome(session_id)
            return
        box.disabled = True
        self._chat_block(text, role="user", label="you")
        self._show_thinking()
        if self.inject is not None and self.inject.handles(text):
            self.run_worker(
                self._run_inject(self.inject, text),
                group="scenario",
                exclusive=True,
                exit_on_error=False,
            )
            return
        self.run_worker(
            self._master_turn(text),
            group="llm",
            exclusive=True,
            exit_on_error=False,
        )

    async def _run_inject(self, inject: InjectCommand, text: str) -> None:
        for line in await inject.run(text):
            self._chat_block(line)

    async def _master_turn(self, text: str) -> None:
        reply = await self.master_llm.handle_developer_message(
            text, on_activity=self.runtime.board.note_master_activity
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
                self.runtime.board.clear_master_activity()
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
            self.query_one(FleetSidebar).update_view(event.view)
            self.query_one("#notice", AttentionNotice).update_view(event.view)
        elif isinstance(event, EscalationArrived):
            self._event_line(
                f"{event.session_id} requested a decision: "
                f"{event.escalation_title}"
            )
        elif isinstance(event, PaneEscalationArrived):
            self._event_line(
                f"{event.kind} escalation {event.escalation_id} from "
                f"{event.session_id}"
            )
        elif isinstance(event, ProposalArrived):
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

    def _parse_command(self, text: str, name: str) -> tuple[bool, str | None]:
        """Match a slash command, with or without a trailing session id.

        Args:
            text: The developer's submitted line, already stripped.
            name: The command's spelling, including its leading slash.

        Returns:
            ``(True, argument)`` when ``text`` is ``name`` or ``name`` plus an
            argument (``None`` when there was none); ``(False, None)`` when
            ``text`` does not name this command at all.
        """
        if text == name:
            return True, None
        if text.startswith(name + " "):
            return True, text[len(name):].strip() or None
        return False, None

    def _show_escalation(self) -> None:
        """Paste the head escalation's disclosure into the chat, verbatim."""
        rendered = self.runtime.desk.rendered_head()
        if rendered is None:
            self._event_line("no escalation is waiting")
            return
        self._chat_block(rendered)

    def _show_pane(
        self, kind: PaneKind, session_id: str | None, *, noun: str, command: str
    ) -> None:
        """Paste an open pane escalation's disclosure into the chat, verbatim."""
        self._paste_held(
            self.runtime.desk.rendered_panes(kind),
            session_id,
            noun=noun,
            command=command,
        )

    def _show_proposal(self, session_id: str | None) -> None:
        """Paste a pending proposal into the chat, verbatim."""
        self._paste_held(
            self.runtime.board.rendered_proposals(),
            session_id,
            noun="proposal",
            command="/proposal",
        )

    def _paste_held(
        self, held: dict[str, str], session_id: str | None, *, noun: str, command: str
    ) -> None:
        """Paste ``session_id``'s live disclosure, or the only one, into the chat.

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
        rows = self.runtime.board.build_view().rows
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
