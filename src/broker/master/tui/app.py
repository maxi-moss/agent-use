"""BrokerMasterApp — the developer's chat TUI.

Structural rules encoded here:
- The socket server is a plain asyncio task: created in on_mount, cancelled
  and awaited in on_unmount (never action_quit — bypassed by App.exit()).
- The LLM turn runs under a worker with an explicit group ("llm"),
  exclusive=True, exit_on_error=False; the prompt box is re-enabled in
  on_worker_state_changed, never at the worker body's end.
- Widgets receive pre-rendered STRINGS from the runtime via post_message —
  never a payload they could re-render (thin-master rule).
"""

import asyncio
import contextlib
from pathlib import Path
from typing import Any, cast

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import RichLog, Static
from textual.worker import Worker, WorkerState

from broker.config import BrokerConfig
from broker.llm import LLMCaller, TurnResult
from broker.master.llm import MasterLLM
from broker.master.messages import (
    CompletionArrived,
    EscalationArrived,
    LLMReply,
    Notice,
    PermissionEscalationArrived,
    ProposalArrived,
    QueueDepthChanged,
    SessionStatusChanged,
)
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry
from broker.master.runtime import MasterRuntime
from broker.master.testmode import load_scenario, run_scenario
from broker.master.tui.prompt_widget import PromptArea


class BrokerMasterApp(App[None]):
    CSS = """
    #chat { height: 1fr; }
    #queue-depth { height: 1; }
    #events { height: 10; }
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
        # In test mode the runtime's posts are teed into a capture the scenario
        # runner reads its assertions from, while still reaching the widgets.
        self._scenario_posts: list[Any] = []
        post = self._tee_post if self.test_mode else self.post_message
        self.runtime = MasterRuntime(
            post, registry, queue, cfg, anchor_pane=anchor_pane
        )
        self.master_llm = MasterLLM(llm_call, self.runtime, cfg)
        self._server_task: asyncio.Task[None] | None = None

    @property
    def test_mode(self) -> bool:
        """True when the app was given scenarios to drive instead of an LLM."""
        return self.scenarios_dir is not None

    def _tee_post(self, message: Message) -> object:
        """Capture a runtime message for the scenario runner, then post it."""
        self._scenario_posts.append(message)
        return self.post_message(message)

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat")
        yield Static(Text("escalation queue: empty"), id="queue-depth")
        yield RichLog(id="events", wrap=True)
        yield PromptArea(
            placeholder="task, decision, or question… (ctrl+j for newline)",
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
        box.disabled = True
        self._chat_block(f"you: {text}")
        if self.test_mode and text.startswith("/"):
            self._handle_command(text)
            return
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
        reply = await self.master_llm.handle_developer_message(text)
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
        if event.state is WorkerState.ERROR:
            # Fail loud; the app (and its socket server) survives.
            self._chat_block(f"[master error] {worker.error!r}")

    # ── runtime → UI (pre-rendered strings, displayed verbatim) ──────────────

    def on_llmreply(self, message: LLMReply) -> None:
        # Textual's handler-name derivation collapses the acronym:
        # LLMReply → "on_llmreply", not "on_llm_reply".
        self._chat_block(f"master: {message.text}")

    def on_escalation_arrived(self, message: EscalationArrived) -> None:
        self._chat_block(message.rendered)
        self._event_line(
            f"escalation {message.escalation_id} from {message.session_id}"
        )

    def on_permission_escalation_arrived(
        self, message: PermissionEscalationArrived
    ) -> None:
        self._chat_block(message.rendered)
        self._event_line(
            f"permission escalation {message.escalation_id} from "
            f"{message.session_id}"
        )

    def on_proposal_arrived(self, message: ProposalArrived) -> None:
        self._chat_block(message.rendered)
        self._event_line(
            f"proposal {message.proposal_id} from {message.session_id} "
            "awaiting approval"
        )

    def on_completion_arrived(self, message: CompletionArrived) -> None:
        # Wrap with context; the summary itself passes through untouched.
        self._chat_block(
            f"Session {message.session_id} completed:\n{message.summary}"
        )
        self._event_line(f"{message.session_id} completed")

    def on_notice(self, message: Notice) -> None:
        self._event_line(message.text)

    def on_session_status_changed(self, message: SessionStatusChanged) -> None:
        self._event_line(f"{message.session_id} → {message.state}")

    def on_queue_depth_changed(self, message: QueueDepthChanged) -> None:
        # Ids and a count only — payloads live in the chat, when surfaced.
        if message.depth == 0:
            text = "escalation queue: empty"
        elif message.waiting:
            text = (
                f"escalation queue: {message.depth} "
                f"(waiting: {', '.join(message.waiting)})"
            )
        else:
            text = f"escalation queue: {message.depth}"
        self.query_one("#queue-depth", Static).update(Text(text))

    # ── helpers ──────────────────────────────────────────────────────────────

    def _chat_block(self, text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        # rich.Text: no markup interpretation — the string renders verbatim.
        chat.mount(Static(Text(text)))
        chat.anchor()

    def _event_line(self, text: str) -> None:
        self.query_one("#events", RichLog).write(Text(text))
