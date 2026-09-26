"""Master runtime layer: socket server, message routing, session spawn,
stop and control, and the fleet view published once per routed message or
runtime tool call. Every escalation waiting on the developer belongs to
``broker.master.escalation_desk``.

Every broker → master message is ACKED with Response(ok=True/False): session
brokers deliver upward messages via client.request and fail loud when nothing
answers. Every session-control tool returns a ``ControlResult``: an expected
refusal, an unknown session included, is ``ok=False``, never an exception.
"""

import asyncio
import contextlib
import functools
import logging
from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any, Concatenate

from pydantic import ValidationError

from broker import decision_log
from broker.config import AdoptedSession, BrokerConfig, ResumedTask
from broker.paths import BrokerPaths
from broker.master import notifier
from broker.master.broker_link import (
    LINK_FAILURES,
    REQUEST_TIMEOUT_S,
    BrokerLink,
    adoption_fields,
    broker_is_listening,
)
from broker.master.control_result import ControlResult, refusal
from broker.master.escalation_desk import EscalationDesk
from broker.master.fleet_board import FleetBoard
from broker.master.outcome import SessionOutcome, build_outcome
from broker.master.viewmodel import (
    CompletionArrived,
    EventSink,
    Notice,
    SessionStateChanged,
)
from broker.master.pane_escalations import PaneEscalations
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.protocol.constants import (
    ABSORBING_STATES,
    SETTLED_STATES,
    NackCode,
    SessionState,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DECISION_DELIVERED,
    T_DECISION_UNDELIVERED,
    T_ESCALATION,
    T_ESCALATION_RETRACT,
    T_FATAL_ERROR,
    T_LIVE_STATUS,
    T_PANE_ESCALATION,
    T_PANE_RETRACT,
    T_PROMPT_PROPOSAL,
    T_PROMPT_UNDELIVERED,
    T_SESSION_ENDED,
)
from broker.protocol.schemas import (
    MASTER_SOCKET_PAYLOADS,
    ApprovePromptPayload,
    BudgetUpdatePayload,
    CompletionPayload,
    DecisionLogPayload,
    DecisionLogRequestPayload,
    Envelope,
    EscalationPayload,
    FatalErrorPayload,
    LiveStatusPayload,
    NackPayload,
    PermissionEscalationPayload,
    PermissionLogPayload,
    PermissionLogRequestPayload,
    PromptProposalPayload,
    PromptUndeliveredPayload,
    QuestionEscalationPayload,
    ReactivatePayload,
    Response,
    SendPromptPayload,
    SessionEndedPayload,
    StatusPayload,
    StatusRequestPayload,
    WireMessage,
    nack_response,
)
from broker.protocol.server import serve_unix

logger = logging.getLogger(__name__)

_Handler = Callable[[str, Any], Awaitable[NackPayload | None]]

# Failures the on-demand status probe absorbs into a warning line: a session
# that cannot be reached must not fail the whole listing.
PROBE_FAILURES = (*LINK_FAILURES, ValidationError)


def _publishes[**P, R](
    method: Callable[Concatenate["MasterRuntime", P], Coroutine[Any, Any, R]],
) -> Callable[Concatenate["MasterRuntime", P], Coroutine[Any, Any, R]]:
    """Publish the fleet view and surface the head once ``method`` finishes."""

    @functools.wraps(method)
    async def wrapper(self: "MasterRuntime", *args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await method(self, *args, **kwargs)
        finally:
            self.board.publish()
            await self.desk.surface_head()

    return wrapper


class MasterRuntime:
    """Routes broker messages and runs the developer's session-control tools.

    Every connection handler runs concurrently with the LLM tool loop, so
    state read before an ``await`` is re-read after it, and markers are set
    before the send they guard.
    """

    def __init__(
        self,
        emit: EventSink,
        registry: Registry,
        queue: EscalationQueue,
        panes: PaneEscalations,
        cfg: BrokerConfig,
        *,
        anchor_pane: str,
        claude_json: Path,
    ) -> None:
        """Wire the runtime to its frontend sink, its persisted stores and the config.

        Args:
            emit: Receives every renderer-neutral view event the runtime produces.
            registry: Loaded session registry.
            queue: Loaded decision-escalation queue.
            panes: Loaded open pane escalations.
            cfg: Broker configuration.
            anchor_pane: Herdr pane every spawned session is anchored to.
            claude_json: Claude Code's ``~/.claude.json`` state file, resolved
                once by the composition root.
        """
        self.emit = emit
        self.registry = registry
        self.cfg = cfg
        self.anchor_pane = anchor_pane
        self.paths = BrokerPaths(cfg.broker_home)
        self.master_socket_path = self.paths.master_socket
        self.link = BrokerLink(
            self.paths,
            cfg,
            self.master_socket_path,
            claude_json,
            self._on_broker_exit,
        )
        self.board = FleetBoard(registry, queue, panes, cfg.budget_max, emit)
        self.desk = EscalationDesk(queue, panes, registry, self.link, emit)
        self._serve_task: asyncio.Task[None] | None = None
        self._handlers: dict[str, _Handler] = {
            T_ESCALATION: self.desk.accept,
            T_PANE_ESCALATION: self.desk.accept_pane,
            T_COMPLETION: self._on_completion,
            T_SESSION_ENDED: self._on_session_ended,
            T_FATAL_ERROR: self._on_fatal_error,
            T_ESCALATION_RETRACT: self.desk.retract,
            T_PANE_RETRACT: self.desk.retract_pane,
            T_PROMPT_UNDELIVERED: self._on_prompt_undelivered,
            T_PROMPT_PROPOSAL: self._on_prompt_proposal,
            T_BUDGET_UPDATE: self._on_budget_update,
            T_DECISION_DELIVERED: self.desk.delivered,
            T_DECISION_UNDELIVERED: self.desk.undelivered,
            T_LIVE_STATUS: self._on_live_status,
        }

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start serving the master socket in a background task.

        Raises:
            RuntimeError: The runtime is already serving.
        """
        if self._serve_task is not None:
            raise RuntimeError("MasterRuntime.start called while already serving")
        self._serve_task = asyncio.create_task(self._serve())

    async def aclose(self) -> None:
        """Stop serving and watching brokers, and persist the registry."""
        task, self._serve_task = self._serve_task, None
        try:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await self.link.aclose()
        finally:
            self.registry.save()

    # ── socket server ────────────────────────────────────────────────────────

    async def _serve(self) -> None:
        """Bind the master socket and serve until cancelled."""
        # A head or open pane escalation loaded from disk has never been
        # announced in this process, so each is announced here, exactly once.
        self.board.publish()
        await self.desk.surface_head()
        await self.desk.announce_open_panes()
        server = await serve_unix(self.master_socket_path, self.handle)
        await self._repopulate_from_brokers()
        async with server:
            await server.serve_forever()

    @_publishes
    async def _repopulate_from_brokers(self) -> None:
        """Refresh state, task-activity and any pending proposal from each surviving broker."""

        for name in self.registry.names_in_order():
            if self.registry.records[name].state in ABSORBING_STATES:
                continue
            try:
                status = await self.probe_status(name)
            except PROBE_FAILURES:
                continue
            if status.pending_proposal is not None:
                self.board.register_proposal(name, status.pending_proposal)

    async def handle(self, env: Envelope) -> Response:
        """Handle one inbound envelope, turning any failure into a NACK.

        Args:
            env: Envelope received on the master socket.

        Returns:
            The ACK or NACK for ``env``.
        """
        try:
            return await self._handle(env)
        except Exception as exc:  # fail loud to the developer, never crash serve
            logger.exception("master handler error on %r", env.type)
            self.emit(
                Notice(f"master handler error on {env.type!r}: {exc!r}")
            )
            return nack_response(env, f"master handler error: {exc!r}", None)

    async def _handle(self, env: Envelope) -> Response:
        """Validate an envelope, check its sender, and run its message's handler.

        Args:
            env: Envelope received on the master socket.

        Returns:
            The ACK once handled, otherwise the NACK saying why not.
        """
        session_id = env.session_id or ""
        validate = MASTER_SOCKET_PAYLOADS.get(env.type)
        if validate is None:
            msg = f"unknown message type {env.type!r} from session {session_id!r}"
            self.emit(Notice(msg))
            return nack_response(env, msg, NackCode.MALFORMED)
        try:
            payload = validate(env.payload)
        except ValidationError as exc:
            # Never act on a thin message as if it were complete.
            self.emit(
                Notice(
                    f"MALFORMED {env.type} from session {session_id!r} — NOT "
                    f"handled.\nvalidation: {exc}\nraw payload: {env.payload!r}"
                )
            )
            return nack_response(
                env, f"malformed {env.type}: {exc}", NackCode.MALFORMED
            )
        if session_id not in self.registry.records:
            msg = f"{env.type} from unknown session {session_id!r} — refused"
            self.emit(Notice(msg))
            return nack_response(env, msg, NackCode.UNKNOWN_SESSION)
        if (
            isinstance(
                payload,
                EscalationPayload
                | PermissionEscalationPayload
                | QuestionEscalationPayload,
            )
            and payload.session_id != session_id
        ):
            msg = (
                f"{env.type} sent by session {session_id!r} names session "
                f"{payload.session_id!r} — refused"
            )
            self.emit(Notice(msg))
            return nack_response(env, msg, NackCode.UNKNOWN_SESSION)
        return await self._route(env, session_id, payload)

    @_publishes
    async def _route(
        self, env: Envelope, session_id: str, payload: WireMessage
    ) -> Response:
        """Run an accepted message's handler.

        Args:
            env: Envelope the message arrived in.
            session_id: Registered session that sent it.
            payload: The validated message.

        Returns:
            The ACK once handled, otherwise the NACK carrying the refusal.
        """
        refusal = await self._handlers[env.type](session_id, payload)
        if refusal is not None:
            return nack_response(env, refusal.error, refusal.reason_code)
        return Response(id=env.id, ok=True)

    async def _on_completion(
        self, session_id: str, p: CompletionPayload
    ) -> NackPayload | None:
        """Tell the developer a session finished its task.

        Args:
            session_id: Session that completed.
            p: The session's completion report.

        Returns:
            ``None``; a completion is never refused.
        """
        self.emit(CompletionArrived(session_id, p.headline, p.supporting))
        await notifier.notify_or_notice(
            self.emit,
            notifier.notify_done,
            f"Session {session_id} complete",
            p.headline,
        )
        return None

    async def _on_fatal_error(
        self, session_id: str, p: FatalErrorPayload
    ) -> NackPayload | None:
        """Report a session's failure and retract the escalations it can no longer answer.

        Args:
            session_id: Session that failed.
            p: The failure the broker reported.

        Returns:
            ``None``; a fatal error is never refused.
        """
        self.emit(
            Notice(f"session {session_id} FATAL [{p.error_class}]: {p.detail}")
        )
        # An errored session can no longer answer; its escalations would
        # otherwise wedge the queue, undispatchable to a dead session.
        await self.desk.retract_stranded(session_id)
        self.desk.retract_stranded_panes(session_id)
        await notifier.notify_or_notice(
            self.emit,
            notifier.notify_request,
            f"Session {session_id} failed",
            f"{p.error_class}: {p.detail}",
        )
        return None

    async def _on_prompt_undelivered(
        self, session_id: str, p: PromptUndeliveredPayload
    ) -> NackPayload | None:
        """Tell the developer an accepted prompt never reached its pane.

        Args:
            session_id: Session whose pane refused the prompt.
            p: Why the prompt did not land.

        Returns:
            ``None``; the report is never refused.
        """
        self.emit(
            Notice(
                f"prompt for session {session_id} did NOT reach its pane: "
                f"{p.detail}"
            )
        )
        return None

    async def _on_prompt_proposal(
        self, session_id: str, p: PromptProposalPayload
    ) -> NackPayload | None:
        """Hold a session's prompt proposal for the developer's approval.

        Args:
            session_id: Session that proposed the prompt.
            p: The proposal.

        Returns:
            ``None``; a proposal is never refused.
        """
        self.board.register_proposal(session_id, p)
        return None

    async def _on_budget_update(
        self, session_id: str, p: BudgetUpdatePayload
    ) -> NackPayload | None:
        """Persist a session's autonomous-answer count.

        Args:
            session_id: Session whose count changed.
            p: The new count.

        Returns:
            ``None``; the update is never refused.
        """
        record = self.registry.get(session_id)
        record.budget_count = p.count
        self.registry.upsert(record)
        return None

    async def _on_live_status(
        self, session_id: str, p: LiveStatusPayload
    ) -> NackPayload | None:
        """Fold a session's pushed live status into the fleet.

        Args:
            session_id: Session that pushed its status.
            p: The broker's current live status.

        Returns:
            ``None``; an absorbed session's push is ACKed and ignored.
        """
        if self.board.apply_live_status(session_id, p):
            self._set_state(session_id, p.state)
        return None

    # ── session control (LLM-layer tool implementations) ─────────────────────

    @_publishes
    async def spawn_session(self, intent: str, cwd: str) -> ControlResult:
        """Spawn a session broker for ``cwd`` and register it.

        Args:
            intent: Raw developer intent for the session, held until an
                approved prompt supersedes it.
            cwd: Working directory for the session; must already exist.

        Returns:
            A confirmation naming the session, its pid and its cwd, or the
            refusal when ``cwd`` is not an existing directory.
        """
        cwd_path = Path(cwd).resolve()
        if not cwd_path.is_dir():
            return refusal(self.emit, f"cwd does not exist: {cwd}")
        name = self.registry.allocate_name()
        record = SessionRecord(
            name=name,
            socket_path=str(self.paths.session_socket(name)),
            cwd=str(cwd_path),
            anchor_pane=self.anchor_pane,
            intent=intent,
        )
        pid = await self.link.spawn(record, adopt=None, resume=None)
        self.registry.upsert(record)
        self.emit(SessionStateChanged(name, record.state))
        return ControlResult(True, f"spawned session {name} (pid {pid}) in {cwd_path}")

    @_publishes
    async def reassign_session(
        self, session_id: str, intent: str
    ) -> ControlResult:
        """Hand a live session to a freshly spawned broker with a new task.

        The Claude session, its pane and its transcript survive; only the
        broker driving them is replaced. The new broker reuses the session's
        socket path, because ``BROKER_SOCKET`` was baked into the pane's
        environment when it was split and cannot be changed afterwards.

        Args:
            session_id: Registry name of the session to hand over.
            intent: Raw developer intent for the new task.

        Returns:
            A confirmation naming the session and the new broker's pid, or the
            refusal: the session is unknown or ended while this call was in
            flight, the registry does not know its pane, Claude session id or
            transcript path, or a broker still answers on its socket.
        """
        record = self._lookup(session_id)
        if isinstance(record, ControlResult):
            return record
        # BEFORE anything is torn down: an unreassignable session must not be
        # left with its old broker killed and no replacement.
        adopt = self._adoption(record)
        if isinstance(adopt, ControlResult):
            return adopt
        stopped = await self._stop(record)
        if not stopped.ok:
            return stopped
        record = self._lookup(session_id)
        if isinstance(record, ControlResult):
            return record
        busy = await self.link.require_socket_free(record)
        if busy is not None:
            return refusal(self.emit, busy)
        record = self._lookup(session_id)
        if isinstance(record, ControlResult):
            return record
        record.intent = intent
        record.approved_prompt = None  # superseded; set again on approval
        record.title = ""
        record.budget_count = 0
        pid = await self.link.spawn(record, adopt=adopt, resume=None)
        record = self._lookup(session_id)
        if isinstance(record, ControlResult):
            return record
        self.registry.upsert(record)
        self._set_state(session_id, SessionState.SPAWNING)
        return ControlResult(
            True, f"session {session_id} reassigned to a new broker (pid {pid})"
        )

    @_publishes
    async def attach_session(self, session_id: str) -> ControlResult:
        """Bind a fresh broker to a session whose own broker is gone.

        A pure resume: the persisted approved prompt and budget count carry
        over untouched, no new intent is taken, no grounding runs and no
        proposal comes back. The new broker reuses the session's socket path,
        because ``BROKER_SOCKET`` was baked into the pane's environment when
        it was split and cannot be changed afterwards.

        Args:
            session_id: Registry name of the session to reattach.

        Returns:
            A confirmation naming the session and the new broker's pid, or the
            refusal: no such session (a session whose pane was found gone is
            removed from the registry, so there is nothing left to attach), the
            registry only partly knows it, no approved prompt was ever
            persisted for it, or a broker still answers on its socket.
        """
        record = self._lookup(session_id)
        if isinstance(record, ControlResult):
            return record
        # Every refusal fires before any side effect.
        adopt = self._adoption(record)
        if isinstance(adopt, ControlResult):
            return adopt
        if record.approved_prompt is None:
            return refusal(
                self.emit,
                f"session {session_id} has no persisted approved prompt to "
                "resume — its broker died before a prompt was approved. Use "
                "reassign_session with a new task instead.",
            )
        # One probe, never a poll: nothing was stopped, so waiting cannot
        # free the socket. Anything alive or ambiguous refuses.
        if await broker_is_listening(Path(record.socket_path)):
            return refusal(
                self.emit,
                f"session {session_id}: a broker is still answering on "
                f"{record.socket_path} — refusing to attach",
            )
        # ``record`` stays bound to the same registry object throughout (a
        # concurrent upsert mutates it in place); each lookup below is only
        # a liveness check, so the ``approved_prompt`` narrowing above holds.
        if isinstance(gone := self._lookup(session_id), ControlResult):
            return gone
        # The dead broker's stranded escalations, of both kinds: a decision
        # dispatched to one would be discarded, and a live entry would refuse
        # the resumed broker's first raise. It re-raises if the situation
        # still holds.
        await self.desk.retract_stranded(session_id)
        if isinstance(gone := self._lookup(session_id), ControlResult):
            return gone
        self.desk.retract_stranded_panes(session_id)
        resume = ResumedTask(
            approved_prompt=record.approved_prompt,
            completed=record.state == SessionState.COMPLETED,
        )
        pid = await self.link.spawn(record, adopt=adopt, resume=resume)
        if isinstance(gone := self._lookup(session_id), ControlResult):
            return gone
        self.registry.upsert(record)
        self._set_state(session_id, SessionState.SPAWNING)
        return ControlResult(
            True, f"session {session_id} reattached to a new broker (pid {pid})"
        )

    @_publishes
    async def reactivate_session(
        self, session_id: str, intent: str
    ) -> ControlResult:
        """Give a completed session a new task without replacing its broker.

        Args:
            session_id: Registry name of the completed session.
            intent: Raw developer intent for the new task.

        Returns:
            A confirmation, or the refusal: the session is unknown, never
            replied, or is not completed and so has a task it is still driving.
        """
        record = self._lookup(session_id)
        if isinstance(record, ControlResult):
            return record
        rejected = await self.link.deliver(
            record,
            ReactivatePayload(intent=intent),
            rejection=f"session {session_id} refused reactivation",
        )
        if rejected is not None:
            return refusal(self.emit, rejected)
        record.intent = intent
        record.approved_prompt = None  # superseded; set again on approval
        record.title = ""
        self.registry.upsert(record)
        return ControlResult(
            True, f"session {session_id} reactivated — grounding the new task"
        )

    @_publishes
    async def approve_prompt(
        self, proposal_id: str, prompt: str, title: str
    ) -> ControlResult:
        """Approve a pending prompt proposal and send it to its session.

        Args:
            proposal_id: Identifier of the proposal being answered.
            prompt: Prompt text to send — the developer's edit of the
                proposal, or the proposal verbatim.
            title: Short task label shown next to the session in the fleet.

        Returns:
            Approved, or the refusal: an unknown proposal or session, no reply,
            or rejected as stale.
        """
        pending = self.board.proposals.get(proposal_id)
        if pending is None:
            return refusal(
                self.emit, f"unknown proposal {proposal_id!r} — nothing approved"
            )
        name = pending.session_id
        record = self._lookup(name)
        if isinstance(record, ControlResult):
            return record
        rejected = await self.link.deliver(
            record,
            ApprovePromptPayload(proposal_id=proposal_id, prompt=prompt),
            rejection=(
                f"session {name} rejected approval for proposal "
                f"{proposal_id} (stale)"
            ),
        )
        if rejected is not None:
            return refusal(self.emit, rejected)
        del self.board.proposals[proposal_id]
        record.approved_prompt = prompt  # the AUTHORITATIVE intent
        record.title = title
        self.registry.upsert(record)
        return ControlResult(True, f"prompt approved for session {name}")

    @_publishes
    async def send_prompt(self, session_id: str, text: str) -> ControlResult:
        """Send a developer prompt straight to a session.

        Args:
            session_id: Registry name of the target session.
            text: Prompt text, sent verbatim.

        Returns:
            A confirmation that the session accepted the prompt, or the
            refusal: an unknown session, no reply, or the session's reason if
            it NACKed.
        """
        record = self._lookup(session_id)
        if isinstance(record, ControlResult):
            return record
        rejected = await self.link.deliver(
            record,
            SendPromptPayload(text=text),
            rejection=f"session {session_id} rejected the prompt",
        )
        if rejected is not None:
            return refusal(self.emit, rejected)
        return ControlResult(True, f"prompt accepted by session {session_id}")

    async def probe_status(self, session_id: str) -> StatusPayload:
        """Ask a session for its status and record the state it reports.

        Args:
            session_id: Registry name of the session to probe.

        Returns:
            The status as reported by the session broker.
        """
        pushes = self.board.push_count(session_id)
        resp = await self.link.request(
            self.registry.get(session_id),
            StatusRequestPayload(),
            timeout_s=REQUEST_TIMEOUT_S,
        )
        status = StatusPayload.model_validate(resp.payload)
        # A push applied during the await is newer than this reply.
        if self.board.push_count(session_id) != pushes:
            return status
        if self.board.apply_probed_status(session_id, status):
            self._set_state(session_id, status.state)
        return status

    def build_session_outcome(self, session_id: str) -> SessionOutcome:
        """Assemble a settled session's read-only outcome from its decision log.

        Args:
            session_id: Registry name of the session.

        Returns:
            The structured outcome the modal renders.

        Raises:
            KeyError: No such session.
        """
        record = self.registry.get(session_id)
        # Off disk, not over the socket: an error/stopped session's broker
        # process is already gone, so the file is the only source left.
        rows = decision_log.read_rows(self.paths.session_decisions(session_id))
        return build_outcome(
            session_id=session_id,
            title=record.title,
            state=record.state,
            rows=rows,
        )

    @_publishes
    async def get_decision_log(self, session_id: str) -> ControlResult:
        """Fetch a session's decision log as text.

        Args:
            session_id: Registry name of the session to query.

        Returns:
            The log text verbatim, or the refusal: an unknown session, no
            reply, or a NACK.

        Raises:
            ValidationError: The session's reply was not a decision log.
        """
        record = self._lookup(session_id)
        if isinstance(record, ControlResult):
            return record
        reply = await self.link.query(
            record,
            DecisionLogRequestPayload(),
            timeout_s=REQUEST_TIMEOUT_S,
            rejection=f"session {session_id} refused its decision log",
        )
        if isinstance(reply, str):
            return refusal(self.emit, reply)
        return ControlResult(
            True, DecisionLogPayload.model_validate(reply.payload).text
        )

    @_publishes
    async def get_permission_log(self, session_id: str) -> ControlResult:
        """Fetch a session's permission log as text.

        Args:
            session_id: Registry name of the session to query.

        Returns:
            The log text verbatim, or the refusal: an unknown session, no
            reply, or a NACK.

        Raises:
            ValidationError: The session's reply was not a permission log.
        """
        record = self._lookup(session_id)
        if isinstance(record, ControlResult):
            return record
        reply = await self.link.query(
            record,
            PermissionLogRequestPayload(),
            timeout_s=REQUEST_TIMEOUT_S,
            rejection=f"session {session_id} refused its permission log",
        )
        if isinstance(reply, str):
            return refusal(self.emit, reply)
        return ControlResult(
            True, PermissionLogPayload.model_validate(reply.payload).text
        )

    @_publishes
    async def stop_session(self, session_id: str) -> ControlResult:
        """Shut a session broker down and mark it stopped.

        Args:
            session_id: Registry name of the session to stop.

        Returns:
            A confirmation naming the session, or the refusal: an unknown
            session, or a broker process that outlived its terminate.
        """
        record = self._lookup(session_id)
        if isinstance(record, ControlResult):
            return record
        return await self._stop(record)

    def registry_summary(self) -> str:
        """Render the registry summary for the LLM context and ``list_sessions``.

        Returns:
            One line per session, or ``"(no sessions)"``.
        """
        if not self.registry.records:
            return "(no sessions)"
        lines: list[str] = []
        for name in self.registry.names_in_order():
            r = self.registry.records[name]
            intent = r.approved_prompt or r.intent
            lines.append(
                f"- {name}: state={r.state} "
                f"budget={r.budget_count}/{self.cfg.budget_max} "
                f"cwd={r.cwd} intent={intent}"
            )
        return "\n".join(lines)

    @_publishes
    async def probe_sessions(self) -> dict[str, Exception | None]:
        """Probe every session for its live status.

        Returns:
            Each session's probe failure, or ``None`` for a session that
            answered, in registry order.
        """
        failures: dict[str, Exception | None] = {}
        for name in self.registry.names_in_order():
            try:
                await self.probe_status(name)
            except PROBE_FAILURES as exc:
                failures[name] = exc
            else:
                failures[name] = None
        return failures

    async def list_sessions(self) -> ControlResult:
        """Render the registry summary, probing each session for a live prompt.

        The prompt flag is refreshed by the probe and never enters the
        summary the master carries into every turn.

        Returns:
            The registry summary, followed by a line for each session found
            waiting on a native permission prompt and for each session that
            could not be reached.
        """
        lines = [self.registry_summary()]
        for name, failure in (await self.probe_sessions()).items():
            if failure is not None:
                # An unreachable session costs one line of the listing, never
                # the whole listing.
                lines.append(
                    f"- {name}: unreachable ({failure!r}) — could not read "
                    "whether it is sitting on a permission prompt"
                )
                continue
            if self.board.permission_prompt(name):
                lines.append(
                    f"- {name}: sitting on a permission prompt, answered in "
                    f"pane {self.desk.pane_of(name)}"
                )
        return ControlResult(True, "\n".join(lines))

    # ── internals ────────────────────────────────────────────────────────────

    def _lookup(self, session_id: str) -> SessionRecord | ControlResult:
        """Return a session's record, or the refusal naming it unknown."""
        record = self.registry.records.get(session_id)
        if record is None:
            return refusal(
                self.emit, f"session {session_id!r} is not in the registry"
            )
        return record

    def _adoption(self, record: SessionRecord) -> AdoptedSession | ControlResult:
        """Return what a replacement broker adopts, or the refusal naming the gaps."""
        try:
            return adoption_fields(record)
        except ValueError as exc:
            return refusal(self.emit, str(exc))

    async def _stop(self, record: SessionRecord) -> ControlResult:
        """Shut a session's broker down and mark the session stopped.

        Args:
            record: Registry record of the session to stop.

        Returns:
            A confirmation naming the session, or the refusal: a broker process
            that outlived its terminate.
        """
        still_running = await self.link.stop(record)
        if still_running is not None:
            return refusal(self.emit, still_running)
        self.desk.session_stopped(record.name)
        self._set_state(record.name, SessionState.STOPPED)
        return ControlResult(True, f"session {record.name} stopped")

    async def _on_session_ended(
        self, session_id: str, p: SessionEndedPayload
    ) -> NackPayload | None:
        """Retire a session whose broker reported its ``SessionEnd``.

        Args:
            session_id: Session that ended.
            p: The terminal report.

        Returns:
            ``None``; the report is never refused.
        """
        self.board.forget(session_id)
        self.link.forget(session_id)
        self.registry.remove(session_id)
        # The broker exits without withdrawing its live escalations, so
        # retract them all — a stranded head would wedge the FIFO queue,
        # undispatchable to a gone session.
        await self.desk.retract_stranded(session_id)
        self.desk.retract_stranded_panes(session_id)
        self.emit(
            Notice(f"session {session_id} ended (/exit) — removed from the fleet")
        )
        return None

    def _on_broker_exit(self, name: str, returncode: int) -> None:
        """Mark a session unmanaged when its broker exits on its own.

        Args:
            name: Registry name of the session whose broker exited.
            returncode: The broker process's exit code.
        """
        record = self.registry.records.get(name)
        if record is None or record.state in SETTLED_STATES:
            return
        self.emit(
            Notice(
                f"session {name}: its broker exited unexpectedly (exit code "
                f"{returncode}) — see {self.paths.session_stderr(name)}"
            )
        )
        self._set_state(name, SessionState.UNMANAGED)
        self.board.publish()

    def _set_state(self, name: str, state: SessionState) -> None:
        """Record a session's new state and tell the TUI, once per change.

        Args:
            name: Registry name of the session.
            state: New state to persist.
        """
        # No ``await`` may sit between this read and the upsert below, or two
        # interleaving handlers would read-modify-write clobber each other
        # under N concurrent brokers.
        try:
            record = self.registry.get(name)
        except KeyError:
            self.emit(Notice(f"message from unknown session {name!r}"))
            return
        if record.state == state:
            return
        logger.info("session %s: %s -> %s", name, record.state, state)
        record.state = state
        self.registry.upsert(record)  # sync; no await before this point
        if state in SETTLED_STATES | {SessionState.UNMANAGED}:
            self.board.settle(name)
        self.emit(SessionStateChanged(name, state))
