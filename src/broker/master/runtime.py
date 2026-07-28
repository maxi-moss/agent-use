"""Master runtime layer: socket server, escalation slot, session
spawn/stop, dispatch with liveness-at-dispatch, and the ONLY
renderers of broker payloads.

The runtime/LLM split is load-bearing: everything the developer reads is
rendered HERE, verbatim, and handed to the TUI (and to the LLM layer as an
opaque block). Re-summarising happens nowhere — structurally.

Every broker → master message is ACKED with Response(ok=True/False): session
brokers deliver upward messages via client.request and fail loud when nothing
answers.
"""

import asyncio
import contextlib
import json
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from textual.message import Message

from broker.claude.trust import seed_trust
from broker.config import BrokerConfig
from broker.master import notifier
from broker.master.messages import (
    CompletionArrived,
    EscalationArrived,
    Notice,
    ProposalArrived,
    SessionStatusChanged,
)
from broker.master.registry import Registry, SessionRecord
from broker.protocol import client
from broker.protocol.constants import (
    T_APPROVE_PROMPT,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_FATAL_ERROR,
    T_GET_DECISION_LOG,
    T_PROMPT_PROPOSAL,
    T_RETRACT,
    T_SEND_PROMPT,
    T_SHUTDOWN,
    T_STATUS,
)
from broker.protocol.schemas import (
    ApprovePromptPayload,
    BudgetUpdatePayload,
    CompletionPayload,
    DispatchDecisionPayload,
    Envelope,
    EscalationPayload,
    FatalErrorPayload,
    PromptProposalPayload,
    Response,
    RetractPayload,
    SendPromptPayload,
    StatusPayload,
)
from broker.protocol.server import serve_unix

# object, not None: App.post_message returns bool, and the bound method is
# passed here directly.
AppPost = Callable[[Message], object]

REQUEST_TIMEOUT_S = 10.0
STOP_WAIT_S = 10.0


class ProtocolViolation(Exception):
    """A broker broke the one-outstanding-escalation invariant."""


class EscalationSlot:
    """Single active escalation (accept / retract / resolve / active)."""

    def __init__(self) -> None:
        """Start with no escalation active."""
        self._active: EscalationPayload | None = None

    @property
    def active(self) -> EscalationPayload | None:
        """Return the escalation awaiting the developer, or ``None``."""
        return self._active

    def accept(self, payload: EscalationPayload) -> None:
        """Make ``payload`` the active escalation.

        Args:
            payload: The escalation to surface to the developer.

        Raises:
            ProtocolViolation: An escalation is already active.
        """
        if self._active is not None:
            raise ProtocolViolation(
                f"escalation {payload.escalation_id} arrived while "
                f"{self._active.escalation_id} is active"
            )
        self._active = payload

    def retract(self, escalation_id: str) -> EscalationPayload | None:
        """Clear an escalation its session has withdrawn.

        Args:
            escalation_id: The escalation being withdrawn.

        Returns:
            The cleared payload, or ``None`` when it was not the active one.
        """
        return self._clear(escalation_id)

    def resolve(self, escalation_id: str) -> EscalationPayload | None:
        """Clear an escalation the developer has decided.

        Args:
            escalation_id: The escalation that was answered.

        Returns:
            The cleared payload, or ``None`` when it was not the active one.
        """
        return self._clear(escalation_id)

    def _clear(self, escalation_id: str) -> EscalationPayload | None:
        """Clear the active escalation when it matches ``escalation_id``.

        Args:
            escalation_id: The escalation expected to be active.

        Returns:
            The cleared payload, or ``None`` when a different escalation — or
            none at all — was active.
        """
        if (
            self._active is not None
            and self._active.escalation_id == escalation_id
        ):
            cleared = self._active
            self._active = None
            return cleared
        return None


def render_escalation(p: EscalationPayload) -> str:
    """Render the escalation block, deterministic and verbatim.

    Args:
        p: Validated escalation payload from a session broker.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    lines = [
        f"Escalation {p.escalation_id} — session {p.session_id}",
        "",
        "## Task context",
        p.task_context,
        "",
        "## Situation",
        p.situation,
        "",
        "## What was asked",
        p.what_was_asked,
        "",
        "## What is at stake",
        p.what_is_at_stake,
        "",
        "## Alternatives",
    ]
    for alt in p.alternatives:
        lines += [
            f"- {alt.option}",
            f"  pros: {alt.pros}",
            f"  cons: {alt.cons}",
        ]
    lines += [
        "",
        "## Recommendation",
        p.recommendation,
        "",
        "## Uncertainty",
        p.uncertainty,
        "",
        "## What would change my mind",
        p.what_would_change_my_mind,
    ]
    return "\n".join(lines)


def render_proposal(p: PromptProposalPayload) -> str:
    """Render the proposal block, prompt and grounding verbatim.

    Args:
        p: Validated prompt-proposal payload from a session broker.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    return "\n".join(
        [
            f"Prompt proposal {p.proposal_id}",
            "",
            "## Proposed prompt",
            p.proposed_prompt,
            "",
            "## Grounding summary",
            p.grounding_summary,
        ]
    )


class MasterRuntime:
    def __init__(
        self,
        app_post: AppPost,
        registry: Registry,
        cfg: BrokerConfig,
        *,
        anchor_pane: str,
    ) -> None:
        """Wire the runtime to the TUI, the registry and the broker config.

        Args:
            app_post: Posts a Textual message to the app.
            registry: Loaded session registry.
            cfg: Broker configuration.
            anchor_pane: Herdr pane every spawned session is anchored to.
        """
        self.app_post = app_post
        self.registry = registry
        self.cfg = cfg
        self.anchor_pane = anchor_pane
        self.slot = EscalationSlot()
        self.master_socket_path = cfg.broker_home / "master.sock"
        self._proposals: dict[str, str] = {}  # proposal_id -> session name
        self._procs: dict[str, asyncio.subprocess.Process] = {}

    # ── socket server ────────────────────────────────────────────────────────

    async def serve(self) -> None:
        """Bind the master socket and serve until cancelled."""
        server = await serve_unix(self.master_socket_path, self.handle)
        async with server:
            await server.serve_forever()

    async def handle(self, env: Envelope) -> Response | None:
        """Handle one inbound envelope, turning any failure into a NACK.

        Args:
            env: Envelope received on the master socket.

        Returns:
            The ACK or NACK for ``env``.
        """
        try:
            return await self._handle(env)
        except Exception as exc:  # fail loud to the developer, never crash serve
            self.app_post(
                Notice(f"master handler error on {env.type!r}: {exc!r}")
            )
            return self._ack(env, ok=False)

    async def _handle(self, env: Envelope) -> Response | None:
        """Route an envelope to the handling for its message type.

        Args:
            env: Envelope received on the master socket.

        Returns:
            ``Response(ok=True)`` once handled, ``ok=False`` if unknown.
        """
        name = env.session_id or ""
        if env.type == T_ESCALATION:
            return await self._on_escalation(env, name)
        if env.type == T_COMPLETION:
            p = CompletionPayload.model_validate(env.payload)
            self._set_state(name, "completed")
            self.app_post(CompletionArrived(name, p.summary))
            await self._notify(
                notifier.notify_done, f"Session {name} complete", p.summary
            )
            return self._ack(env, ok=True)
        if env.type == T_FATAL_ERROR:
            p = FatalErrorPayload.model_validate(env.payload)
            self._set_state(name, "error")
            self.app_post(
                Notice(f"session {name} FATAL [{p.error_class}]: {p.detail}")
            )
            await self._notify(
                notifier.notify_request,
                f"Session {name} failed",
                f"{p.error_class}: {p.detail}",
            )
            return self._ack(env, ok=True)
        if env.type == T_RETRACT:
            p = RetractPayload.model_validate(env.payload)
            self.slot.retract(p.escalation_id)
            # If it was surfaced, the developer must learn it is no longer
            # live.
            self.app_post(
                Notice(
                    f"escalation {p.escalation_id} from session {name} "
                    f"retracted: {p.reason}"
                )
            )
            return self._ack(env, ok=True)
        if env.type == T_PROMPT_PROPOSAL:
            p = PromptProposalPayload.model_validate(env.payload)
            self._proposals[p.proposal_id] = name
            self._set_state(name, "awaiting_approval")
            self.app_post(
                ProposalArrived(name, p.proposal_id, render_proposal(p))
            )
            return self._ack(env, ok=True)
        if env.type == T_BUDGET_UPDATE:
            p = BudgetUpdatePayload.model_validate(env.payload)
            record = self.registry.get(name)
            record.budget_count = p.count
            self.registry.upsert(record)
            return self._ack(env, ok=True)
        self.app_post(Notice(f"unknown message type {env.type!r} from {name!r}"))
        return self._ack(env, ok=False)

    async def _on_escalation(self, env: Envelope, name: str) -> Response:
        """Validate one escalation, make it active, and surface it.

        Args:
            env: Envelope carrying the escalation payload.
            name: Session name from the envelope, used for rejection notices.

        Returns:
            ``Response(ok=True)`` once live and announced, ``ok=False`` if
            rejected.
        """
        try:
            p = EscalationPayload.model_validate(env.payload)
        except ValidationError as exc:
            # Never render a thin escalation as if complete.
            self.app_post(
                Notice(
                    f"MALFORMED escalation from session {name!r} — NOT "
                    f"surfaced.\nvalidation: {exc}\nraw payload: {env.payload!r}"
                )
            )
            return self._ack(env, ok=False)
        try:
            self.slot.accept(p)
        except ProtocolViolation as exc:
            self.app_post(Notice(f"PROTOCOL VIOLATION: {exc}"))
            return self._ack(env, ok=False)
        self._set_state(p.session_id, "escalated")
        self.app_post(
            EscalationArrived(
                p.session_id, p.escalation_id, render_escalation(p)
            )
        )
        await self._notify(
            notifier.notify_request,
            f"Escalation from session {p.session_id}",
            p.what_was_asked,
        )
        return self._ack(env, ok=True)

    # ── session control (LLM-layer tool implementations) ─────────────────────

    async def spawn_session(self, intent: str, cwd: str) -> str:
        """Spawn a session broker for ``cwd`` and register it.

        Args:
            intent: Raw developer intent for the session, held until an
                approved prompt supersedes it.
            cwd: Working directory for the session; must already exist.

        Returns:
            A confirmation line naming the session, its pid and its cwd.

        Raises:
            ValueError: ``cwd`` is not an existing directory.
        """
        cwd_path = Path(cwd).resolve()
        if not cwd_path.is_dir():
            raise ValueError(f"cwd does not exist: {cwd}")
        name = self.registry.allocate_name()
        socket_path = self.cfg.broker_home / "s" / f"{name}.sock"
        record = SessionRecord(
            name=name,
            socket_path=str(socket_path),
            cwd=str(cwd_path),
            anchor_pane=self.anchor_pane,
            intent=intent,
        )
        seed_trust(cwd_path)  # BEFORE spawn — the dialog eats input
        config_json = json.dumps(
            {
                "name": name,
                "socket_path": str(socket_path),
                "master_socket_path": str(self.master_socket_path),
                "cwd": str(cwd_path),
                "anchor_pane": self.anchor_pane,
                "intent": intent,
                "budget_count": record.budget_count,
                "model_id": self.cfg.model_id,
                "max_tokens": self.cfg.max_tokens,
                "watchdog_seconds": self.cfg.watchdog_seconds,
                "budget_max": self.cfg.budget_max,
            }
        )
        # By module string, never by import — keeps the module boundary
        # structural.
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "broker.session", "--config-json", config_json
        )
        record.pid = proc.pid
        self._procs[name] = proc
        self.registry.upsert(record)
        self.app_post(SessionStatusChanged(name, record.state))
        return f"spawned session {name} (pid {proc.pid}) in {cwd_path}"

    async def approve_prompt(self, proposal_id: str, prompt: str) -> str:
        """Approve a pending prompt proposal and send it to its session.

        Args:
            proposal_id: Identifier of the proposal being answered.
            prompt: Prompt text to send — the developer's edit of the
                proposal, or the proposal verbatim.

        Returns:
            An outcome line: approved, unknown proposal, or rejected as stale.
        """
        name = self._proposals.get(proposal_id)
        if name is None:
            return f"unknown proposal {proposal_id!r} — nothing approved"
        record = self.registry.get(name)
        env = self._env(
            T_APPROVE_PROMPT,
            ApprovePromptPayload(
                proposal_id=proposal_id, prompt=prompt
            ).model_dump(),
        )
        resp = await client.request(
            Path(record.socket_path), env, timeout_s=REQUEST_TIMEOUT_S
        )
        if not resp.ok:
            return (
                f"session {name} rejected approval for proposal "
                f"{proposal_id} (stale)"
            )
        del self._proposals[proposal_id]
        record.approved_prompt = prompt  # the AUTHORITATIVE intent
        self.registry.upsert(record)
        return f"prompt approved for session {name}"

    async def dispatch(self, escalation_id: str, decision: str) -> str:
        """Dispatch a decision to the session whose escalation it answers.

        Args:
            escalation_id: The escalation the decision answers.
            decision: The developer's decision, sent verbatim.

        Returns:
            An outcome line: dispatched to the named session, or a refusal
            naming the escalation that is no longer live.
        """
        # Liveness is checked THE INSTANT before the write, not at
        # surface time. A stale dispatch is the worst failure this system
        # can produce.
        active = self.slot.active
        if active is None or active.escalation_id != escalation_id:
            msg = (
                f"decision NOT dispatched — escalation {escalation_id} is "
                "no longer live"
            )
            self.app_post(Notice(msg))
            return msg
        record = self.registry.get(active.session_id)
        env = self._env(
            T_DISPATCH_DECISION,
            DispatchDecisionPayload(
                escalation_id=escalation_id, response=decision
            ).model_dump(),
        )
        await client.request(
            Path(record.socket_path), env, timeout_s=REQUEST_TIMEOUT_S
        )
        self.slot.resolve(escalation_id)
        self._set_state(record.name, "driving")
        return f"decision dispatched to session {record.name}"

    async def send_prompt(self, session_id: str, text: str) -> str:
        """Send a developer prompt straight to a session.

        Args:
            session_id: Registry name of the target session.
            text: Prompt text, sent verbatim.

        Returns:
            A confirmation line naming the session.
        """
        record = self.registry.get(session_id)
        env = self._env(T_SEND_PROMPT, SendPromptPayload(text=text).model_dump())
        await client.request(
            Path(record.socket_path), env, timeout_s=REQUEST_TIMEOUT_S
        )
        record.budget_count = 0  # developer prompt resets the budget
        self.registry.upsert(record)
        return f"prompt sent to session {session_id}"

    async def probe_status(self, session_id: str) -> StatusPayload:
        """Ask a session for its status and fold the reply into the registry.

        Args:
            session_id: Registry name of the session to probe.

        Returns:
            The status as reported by the session broker.
        """
        record = self.registry.get(session_id)
        env = self._env(T_STATUS, {})
        resp = await client.request(
            Path(record.socket_path), env, timeout_s=REQUEST_TIMEOUT_S
        )
        status = StatusPayload.model_validate(resp.payload)
        record.state = status.state
        record.pane_id = status.pane_id or record.pane_id
        record.claude_session_id = (
            status.claude_session_id or record.claude_session_id
        )
        record.transcript_path = status.transcript_path or record.transcript_path
        self.registry.upsert(record)
        return status

    async def get_decision_log(self, session_id: str) -> str:
        """Fetch a session's decision log as text.

        Args:
            session_id: Registry name of the session to query.

        Returns:
            The log text verbatim, or the empty string when the session sent
            no text or sent something that was not text.
        """
        record = self.registry.get(session_id)
        env = self._env(T_GET_DECISION_LOG, {})
        resp = await client.request(
            Path(record.socket_path), env, timeout_s=REQUEST_TIMEOUT_S
        )
        text = resp.payload.get("text", "")
        return text if isinstance(text, str) else ""

    async def stop_session(self, session_id: str) -> str:
        """Shut a session broker down and mark it stopped.

        Args:
            session_id: Registry name of the session to stop.

        Returns:
            A confirmation line naming the session.
        """
        record = self.registry.get(session_id)
        with contextlib.suppress(ConnectionError, TimeoutError, OSError):
            await client.request(
                Path(record.socket_path),
                self._env(T_SHUTDOWN, {}),
                timeout_s=REQUEST_TIMEOUT_S,
            )
        proc = self._procs.pop(session_id, None)
        if proc is not None and proc.returncode is None:
            try:
                async with asyncio.timeout(STOP_WAIT_S):
                    await proc.wait()
            except TimeoutError:
                proc.terminate()
                await proc.wait()
        self._set_state(session_id, "stopped")
        return f"session {session_id} stopped"

    def render_registry_summary(self) -> str:
        """Render the registry summary for the LLM context and list_sessions.

        Returns:
            One line per session, or ``"(no sessions)"``.
        """
        if not self.registry.records:
            return "(no sessions)"
        lines: list[str] = []
        for name in sorted(self.registry.records):
            r = self.registry.records[name]
            intent = r.approved_prompt or r.intent
            lines.append(
                f"- {name}: state={r.state} "
                f"budget={r.budget_count}/{self.cfg.budget_max} "
                f"cwd={r.cwd} intent={intent}"
            )
        return "\n".join(lines)

    # ── internals ────────────────────────────────────────────────────────────

    def _set_state(self, name: str, state: str) -> None:
        """Record a session's new state and tell the TUI.

        Args:
            name: Registry name of the session.
            state: New state string to persist.
        """
        try:
            record = self.registry.get(name)
        except KeyError:
            self.app_post(Notice(f"message from unknown session {name!r}"))
            return
        record.state = state
        self.registry.upsert(record)
        self.app_post(SessionStatusChanged(name, state))

    async def _notify(
        self,
        fn: Callable[[str, str], Awaitable[None]],
        title: str,
        body: str,
    ) -> None:
        """Send one notification, downgrading a notifier failure to a notice.

        Args:
            fn: Notifier coroutine from ``broker.master.notifier``.
            title: Notification title.
            body: Notification body, passed through verbatim.
        """
        try:
            await fn(title, body)
        except Exception as exc:
            # A dead notifier must not lose the escalation it announces.
            self.app_post(Notice(f"notification failed: {exc}"))

    def _env(self, msg_type: str, payload: dict[str, Any]) -> Envelope:
        """Wrap a payload in an envelope with a fresh message id."""
        return Envelope(id=uuid.uuid4().hex, type=msg_type, payload=payload)

    def _ack(self, env: Envelope, *, ok: bool) -> Response:
        """Build the ACK or NACK answering an envelope."""
        return Response(id=env.id, ok=ok)
