"""Master runtime layer: socket server, escalation queue, session
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
import logging
import re
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from broker.claude.settings import write_session_permissions
from broker.claude.trust import seed_trust
from broker.config import (
    AdoptedSession,
    BrokerConfig,
    ResumedTask,
    SessionBrokerConfig,
)
from broker.herdr import driver
from broker.herdr.driver import HerdrError
from broker.paths import BrokerPaths
from broker.master import notifier
from broker.master.viewmodel import (
    Attention,
    CompletionArrived,
    EscalationArrived,
    EventSink,
    FleetUpdated,
    FleetView,
    HeadRequest,
    Notice,
    PermissionEscalationArrived,
    ProposalArrived,
    SessionRow,
    SessionStatusChanged,
)
from broker.master.queue import (
    EscalationQueue,
    ProtocolViolation,
    QueuePayload,
)
from broker.master.registry import Registry, SessionRecord
from broker.protocol import client
from broker.protocol.constants import (
    NACK_MALFORMED,
    NACK_PROTOCOL_VIOLATION,
    NACK_UNKNOWN_SESSION,
    SessionState,
    T_APPROVE_PROMPT,
    T_CLARIFY_ESCALATION,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DECISION_UNDELIVERED,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_FATAL_ERROR,
    T_GET_DECISION_LOG,
    T_GET_PERMISSION_LOG,
    T_LIVE_STATUS,
    T_PERMISSION_ESCALATION,
    T_PROMPT_PROPOSAL,
    T_REACTIVATE,
    T_RETRACT,
    T_SEND_PROMPT,
    T_SESSION_ENDED,
    T_SHUTDOWN,
    T_STATUS,
)
from broker.protocol.schemas import (
    ApprovePromptPayload,
    ClarifyEscalationReplyPayload,
    ClarifyEscalationRequestPayload,
    BudgetUpdatePayload,
    CompletionPayload,
    DecisionLogPayload,
    DecisionUndeliveredPayload,
    DispatchDecisionPayload,
    Envelope,
    EscalationPayload,
    FatalErrorPayload,
    LiveStatusPayload,
    PermissionEscalationPayload,
    PermissionLogPayload,
    PermissionSuggestion,
    PromptProposalPayload,
    RaiserIdentity,
    ReactivatePayload,
    Response,
    RetractPayload,
    RetrievedSymbol,
    SendPromptPayload,
    StatusPayload,
)
from broker.protocol.server import serve_unix

logger = logging.getLogger(__name__)

_SESSION_NUM = re.compile(r"s(\d+)\Z")


def session_sort_key(name: str) -> tuple[int, str]:
    """Order sessions by numeric id (s2 before s10); any non-'sN' name last."""
    m = _SESSION_NUM.match(name)
    return (int(m.group(1)), "") if m else (10**9, name)


REQUEST_TIMEOUT_S = 10.0
# The broker runs an LLM call before it can reply, so this is far longer than
# REQUEST_TIMEOUT_S and must exceed the broker's own CLARIFY_TIMEOUT_S.
CLARIFY_ESCALATION_TIMEOUT_S = 60.0
STOP_WAIT_S = 10.0
SOCKET_POLL_S = 0.1
SOCKET_PROBE_TIMEOUT_S = 2.0
PANE_PROBE_TIMEOUT_S = 5.0

# Stands in for a pane the registry cannot name. A permission escalation is
# still worth surfacing without it: the developer knows the session.
PANE_UNKNOWN = "(pane unknown)"

# Failures the on-demand status probe absorbs into a warning line: a session
# that cannot be reached must not fail the whole listing.
PROBE_FAILURES = (OSError, ConnectionError, TimeoutError, ValidationError)

# A session in one of these states is settled: it has no live operating
# status, so a late-arriving push must not resurrect it. Left only by a
# master-initiated boundary write (spawn/reassign/attach/reactivate). A
# session that is gone is not settled here — it is removed from the registry
# entirely, so no state stands in for "finished".
_ABSORBING = frozenset(
    {
        SessionState.COMPLETED,
        SessionState.ERROR,
        SessionState.STOPPED,
        SessionState.UNMANAGED,
    }
)


async def _broker_is_listening(path: Path) -> bool:
    """Report whether anything still accepts connections on a session socket.

    Args:
        path: Session socket to probe.

    Returns:
        ``True`` when the connection is accepted, and also when the probe
        itself is inconclusive — an ambiguous result must never read as free.
    """
    try:
        async with asyncio.timeout(SOCKET_PROBE_TIMEOUT_S):
            _, writer = await asyncio.open_unix_connection(str(path))
    except TimeoutError:
        return True  # BEFORE OSError, which TimeoutError subclasses
    except OSError:
        return False  # nothing bound, or a stale file refusing connections
    writer.close()
    with contextlib.suppress(OSError, ConnectionError):
        await writer.wait_closed()
    return True


def _adoption_fields(record: SessionRecord) -> AdoptedSession:
    """Build the block a replacement broker needs to adopt a live session.

    Args:
        record: Registry record of the session a replacement broker takes over.

    Returns:
        The pane id, Claude session id and transcript path, all present.

    Raises:
        ValueError: Any of them is unknown. A broker must never adopt a
            session it only partly knows.
    """
    pane_id = record.pane_id
    claude_session_id = record.claude_session_id
    transcript_path = record.transcript_path
    if not (pane_id and claude_session_id and transcript_path):
        missing = sorted(
            field
            for field, value in (
                ("pane_id", pane_id),
                ("claude_session_id", claude_session_id),
                ("transcript_path", transcript_path),
            )
            if not value
        )
        raise ValueError(
            f"session {record.name} cannot be adopted: the registry has no "
            + ", ".join(missing)
        )
    return AdoptedSession(
        pane_id=pane_id,
        claude_session_id=claude_session_id,
        transcript_path=transcript_path,
    )


async def reconcile_registry(
    registry: Registry, queue: EscalationQueue
) -> list[str]:
    """Classify every registry session at startup, retracting for dead ones.

    Probes are client-side connects and pane reads only — nothing is spawned
    and nothing binds a socket. A session whose pane is definitively gone is
    removed and its queued escalations are retracted; every other outcome
    leaves the session recoverable. An inconclusive probe never removes a
    session — a wrong ``unmanaged`` costs the developer a glance, a wrong
    removal throws away queued decisions.

    Args:
        registry: Loaded session registry; a gone session is removed, every
            other classification updates its state in place and is saved once.
        queue: Persisted escalation queue; a dead session's queued
            escalations are retracted from it before the TUI re-announces
            the head.

    Returns:
        One classification line per session, plus one line per retraction.
    """
    warnings: list[str] = []
    for name in sorted(registry.records, key=session_sort_key):
        record = registry.records[name]
        if await _broker_is_listening(Path(record.socket_path)):
            warnings.append(
                f"session {name}: broker still answering — left as-is"
            )
            continue
        if not record.pane_id:
            record.state = SessionState.UNMANAGED
            warnings.append(
                f"session {name}: nothing answers its socket and the "
                "registry never learned its pane — marked unmanaged"
            )
            continue
        try:
            await asyncio.to_thread(
                driver.pane_read, record.pane_id, timeout_s=PANE_PROBE_TIMEOUT_S
            )
        except HerdrError as exc:
            if exc.code == "pane_not_found":
                warnings.append(
                    f"session {name}: pane {record.pane_id} gone — removed"
                )
                for component in ("broker", "permission"):
                    cleared = queue.retract_for_raiser(
                        RaiserIdentity(component=component, session_id=name)
                    )
                    if cleared is not None:
                        warnings.append(
                            f"session {name}: queued escalation "
                            f"{cleared.escalation_id} retracted — the session "
                            "is dead and no decision can reach it"
                        )
                # A session whose pane is gone is finished: drop it so it can
                # never be a routing candidate. remove() persists on its own.
                registry.remove(name)
                continue
            record.state = SessionState.UNMANAGED
            warnings.append(
                f"session {name}: pane probe inconclusive ({exc}) — "
                "marked unmanaged"
            )
            continue
        except Exception as exc:
            record.state = SessionState.UNMANAGED
            warnings.append(
                f"session {name}: pane probe inconclusive ({exc!r}) — "
                "marked unmanaged"
            )
            continue
        record.state = SessionState.UNMANAGED
        line = (
            f"session {name}: pane alive with nothing driving it — marked "
            "unmanaged; recover it with attach_session"
        )
        if record.approved_prompt is None:
            line += (
                " (no approved prompt persisted — reassign_session with a "
                "new task is the route instead)"
            )
        warnings.append(line)
    if registry.records:
        registry.save()
    return warnings


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


def _render_suggestion(suggestion: PermissionSuggestion) -> str:
    """Render one of Claude Code's permission suggestions as its raw object."""
    data = suggestion if isinstance(suggestion, dict) else suggestion.model_dump()
    return json.dumps(data, sort_keys=True)


def render_permission_escalation(
    p: PermissionEscalationPayload, pane_id: str
) -> str:
    """Render the permission-escalation block, deterministic and verbatim.

    Args:
        p: Validated permission-escalation payload from a session broker.
        pane_id: Pane holding the native prompt, or ``PANE_UNKNOWN``.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    lines = [
        f"Permission escalation {p.escalation_id} — session {p.session_id}",
        "",
        f"The developer answers this in pane {pane_id}, on the native "
        "permission prompt already waiting there. It cannot be answered "
        "here, and no decision sent from here reaches it.",
        "",
        "## Tool",
        p.tool_name,
        "",
        "## Tool input",
        json.dumps(p.tool_input, indent=2, sort_keys=True),
        "",
        "## Why it was escalated",
        p.reason,
        "",
        "## Task intent it was judged against",
        p.task_intent,
        "",
        "## Permission suggestions",
    ]
    if p.permission_suggestions:
        lines += [
            f"- {_render_suggestion(s)}" for s in p.permission_suggestions
        ]
    else:
        lines.append("(none)")
    return "\n".join(lines)


def render_proposal(p: PromptProposalPayload) -> str:
    """Render the proposal block: prompt, grounding, and retrieved code, verbatim.

    Args:
        p: Validated prompt-proposal payload from a session broker.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    lines = [
        f"Prompt proposal {p.proposal_id}",
        "",
        "## Proposed prompt",
        p.proposed_prompt,
        "",
        "## Grounding summary",
        p.grounding_summary,
        "",
        "## Retrieved code",
    ]
    if p.retrieved:
        lines += [_render_retrieved(s) for s in p.retrieved]
    else:
        lines.append("(none)")
    return "\n".join(lines)


def _render_retrieved(s: RetrievedSymbol) -> str:
    """Render one retrieved symbol as a bullet line."""
    if s.score is None:
        return f"- {s.name}"
    return f"- {s.name} (seed {s.score:.2f})"


@dataclass(frozen=True, slots=True)
class PendingProposal:
    """A prompt proposal awaiting the developer's approval."""

    session_name: str
    payload: PromptProposalPayload


class MasterRuntime:
    def __init__(
        self,
        emit: EventSink,
        registry: Registry,
        queue: EscalationQueue,
        cfg: BrokerConfig,
        *,
        anchor_pane: str,
    ) -> None:
        """Wire the runtime to its frontend sink, the registry, the queue and the config.

        Args:
            emit: Receives every renderer-neutral view event the runtime produces.
            registry: Loaded session registry.
            queue: Loaded escalation queue.
            cfg: Broker configuration.
            anchor_pane: Herdr pane every spawned session is anchored to.
        """
        self.emit = emit
        self.registry = registry
        self.cfg = cfg
        self.anchor_pane = anchor_pane
        self.queue = queue
        self._surfaced_id: str | None = None
        self.paths = BrokerPaths(cfg.broker_home)
        self.master_socket_path = self.paths.master_socket
        self.proposals: dict[str, PendingProposal] = {}
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        # Dashboard-only, never persisted: pushed live status per session and
        # the master's own current activity.
        self._activity: dict[str, str] = {}
        self._task_activity: dict[str, str] = {}
        self._permission_prompt_pending: set[str] = set()
        self._master_activity: str | None = None

    # ── socket server ────────────────────────────────────────────────────────

    async def serve(self) -> None:
        """Bind the master socket and serve until cancelled."""
        # A head loaded from disk has never been announced in this process, so
        # it surfaces here, exactly once.
        self._publish_fleet()
        await self._surface_head()
        server = await serve_unix(self.master_socket_path, self.handle)
        await self._repopulate_from_brokers()
        async with server:
            await server.serve_forever()

    async def _repopulate_from_brokers(self) -> None:
        """Refresh state, task-activity and any pending proposal from each surviving broker."""
        for name in sorted(self.registry.records, key=session_sort_key):
            if self.registry.records[name].state in _ABSORBING:
                continue
            try:
                status = await self.probe_status(name)
            except PROBE_FAILURES:
                continue
            if status.pending_proposal is not None:
                self._register_proposal(name, status.pending_proposal)
        self._publish_fleet()

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
            self.emit(
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
        if env.type == T_PERMISSION_ESCALATION:
            return await self._on_permission_escalation(env, name)
        if env.type == T_COMPLETION:
            p = CompletionPayload.model_validate(env.payload)
            self._set_state(name, SessionState.COMPLETED)
            self.emit(CompletionArrived(name, p.summary))
            await self._notify(
                notifier.notify_done, f"Session {name} complete", p.summary
            )
            return self._ack(env, ok=True)
        if env.type == T_SESSION_ENDED:
            return await self._on_session_ended(env, name)
        if env.type == T_FATAL_ERROR:
            p = FatalErrorPayload.model_validate(env.payload)
            self._set_state(name, SessionState.ERROR)
            self.emit(
                Notice(f"session {name} FATAL [{p.error_class}]: {p.detail}")
            )
            # An errored session can no longer answer; its escalations would
            # otherwise wedge the queue, undispatchable to a dead session.
            await self._retract_stranded_escalation(name, "broker")
            await self._retract_stranded_escalation(name, "permission")
            await self._notify(
                notifier.notify_request,
                f"Session {name} failed",
                f"{p.error_class}: {p.detail}",
            )
            return self._ack(env, ok=True)
        if env.type == T_RETRACT:
            p = RetractPayload.model_validate(env.payload)
            cleared = self.queue.retract(p.escalation_id)
            if isinstance(cleared, EscalationPayload):
                # Raising it set ESCALATED here; withdrawing it must undo that
                # or the registry outlives the escalation it describes.
                self._set_state(name, SessionState.DRIVING)
            if (
                cleared is not None
                and cleared.escalation_id == self._surfaced_id
            ):
                # It was surfaced, so the developer must learn it is no
                # longer live; a waiting entry they never saw retracts
                # silently.
                self._surfaced_id = None
                self.emit(
                    Notice(
                        f"escalation {p.escalation_id} from session {name} "
                        f"retracted: {p.reason}"
                    )
                )
            self._publish_fleet()
            await self._surface_head()
            return self._ack(env, ok=True)
        if env.type == T_PROMPT_PROPOSAL:
            p = PromptProposalPayload.model_validate(env.payload)
            self._set_state(name, SessionState.AWAITING_APPROVAL)
            self._register_proposal(name, p)
            return self._ack(env, ok=True)
        if env.type == T_BUDGET_UPDATE:
            p = BudgetUpdatePayload.model_validate(env.payload)
            record = self.registry.get(name)
            record.budget_count = p.count
            self.registry.upsert(record)
            self._publish_fleet()
            return self._ack(env, ok=True)
        if env.type == T_DECISION_UNDELIVERED:
            p = DecisionUndeliveredPayload.model_validate(env.payload)
            return await self._on_decision_undelivered(env, name, p)
        if env.type == T_LIVE_STATUS:
            p = LiveStatusPayload.model_validate(env.payload)
            rec = self.registry.records.get(name)
            # A settled/gone session ignores late pushes: absorbing states are
            # left only by a master-initiated boundary write, never by a stale
            # in-flight push arriving after the fact (cross-connection sends
            # reorder even though each is individually ACKed).
            if rec is not None and rec.state not in _ABSORBING:
                if p.activity:
                    self._activity[name] = p.activity
                else:
                    self._activity.pop(name, None)
                self._note_task_activity(name, p.task_activity)
                if p.permission_prompt:
                    self._permission_prompt_pending.add(name)
                else:
                    self._permission_prompt_pending.discard(name)
                state_changed = self._set_state(name, p.state)
                if not state_changed:
                    self._publish_fleet()  # activity/perm-only change
            # ACK every push, absorbing/unknown included, so the sender never
            # spins re-sending a snapshot the master refuses to apply.
            return self._ack(env, ok=True)
        self.emit(Notice(f"unknown message type {env.type!r} from {name!r}"))
        return self._ack(env, ok=False)

    async def _on_escalation(self, env: Envelope, name: str) -> Response:
        """Validate one escalation, queue it, and surface it if it is next.

        Args:
            env: Envelope carrying the escalation payload.
            name: Session name from the envelope, used for rejection notices.

        Returns:
            ``Response(ok=True)`` once live, ``ok=False`` with a reason code
            if rejected.
        """
        try:
            p = EscalationPayload.model_validate(env.payload)
        except ValidationError as exc:
            # Never render a thin escalation as if complete.
            self.emit(
                Notice(
                    f"MALFORMED escalation from session {name!r} — NOT "
                    f"surfaced.\nvalidation: {exc}\nraw payload: {env.payload!r}"
                )
            )
            return self._nack(env, f"malformed escalation: {exc}", NACK_MALFORMED)
        unknown = self._reject_unknown_session(env, p.session_id, "escalation")
        if unknown is not None:
            return unknown
        refused = self._accept_into_queue(env, p)
        if refused is not None:
            return refused
        self._set_state(p.session_id, SessionState.ESCALATED)
        self._publish_fleet()
        await self._surface_head()
        return self._ack(env, ok=True)

    async def _on_permission_escalation(
        self, env: Envelope, name: str
    ) -> Response:
        """Validate one permission escalation, queue it, and surface it if it
        is next.

        Args:
            env: Envelope carrying the permission-escalation payload.
            name: Session name from the envelope, used for rejection notices.

        Returns:
            ``Response(ok=True)`` once live, ``ok=False`` with a reason code
            if rejected.
        """
        try:
            p = PermissionEscalationPayload.model_validate(env.payload)
        except ValidationError as exc:
            # A permission escalation missing the tool, the reason or the
            # session is not something the developer could act on.
            self.emit(
                Notice(
                    f"MALFORMED permission escalation from session {name!r} — "
                    f"NOT surfaced.\nvalidation: {exc}\n"
                    f"raw payload: {env.payload!r}"
                )
            )
            return self._nack(
                env, f"malformed permission escalation: {exc}", NACK_MALFORMED
            )
        unknown = self._reject_unknown_session(
            env, p.session_id, "permission escalation"
        )
        if unknown is not None:
            return unknown
        refused = self._accept_into_queue(env, p)
        if refused is not None:
            return refused
        self._publish_fleet()
        await self._surface_head()
        return self._ack(env, ok=True)

    def _reject_unknown_session(
        self, env: Envelope, session_id: str, kind: str
    ) -> Response | None:
        """Refuse an escalation from a session the registry does not know.

        Args:
            env: Envelope being answered.
            session_id: Session the payload claims to come from.
            kind: Word naming the escalation kind, used in the notice.

        Returns:
            ``None`` when the session is known, otherwise the NACK.
        """
        if session_id in self.registry.records:
            return None
        msg = f"{kind} from unknown session {session_id!r} — NOT surfaced"
        self.emit(Notice(msg))
        return self._nack(env, msg, NACK_UNKNOWN_SESSION)

    def _accept_into_queue(
        self, env: Envelope, payload: QueuePayload
    ) -> Response | None:
        """Queue ``payload``, or build the NACK refusing it.

        Args:
            env: Envelope being answered.
            payload: The escalation offered to the queue.

        Returns:
            ``None`` once accepted, otherwise the NACK.
        """
        try:
            self.queue.accept(payload)
        except ProtocolViolation as exc:
            self.emit(Notice(f"PROTOCOL VIOLATION: {exc}"))
            return self._nack(env, str(exc), NACK_PROTOCOL_VIOLATION)
        return None

    async def _surface_head(self) -> None:
        """Render, announce and notify the queue's head, exactly once."""
        head = self.queue.active
        if head is None or head.escalation_id == self._surfaced_id:
            return
        self._surfaced_id = head.escalation_id
        if isinstance(head, PermissionEscalationPayload):
            pane_id = self.pane_of(head.session_id)
            self.emit(
                PermissionEscalationArrived(
                    head.session_id,
                    head.escalation_id,
                    render_permission_escalation(head, pane_id),
                )
            )
            await self._notify(
                notifier.notify_request,
                f"Permission prompt in session {head.session_id}",
                f"{head.tool_name} — answer it in pane {pane_id}",
            )
            return
        self.emit(
            EscalationArrived(
                head.session_id, head.escalation_id, render_escalation(head)
            )
        )
        await self._notify(
            notifier.notify_request,
            f"Escalation from session {head.session_id}",
            head.what_was_asked,
        )

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
        record = SessionRecord(
            name=name,
            socket_path=str(self.paths.session_socket(name)),
            cwd=str(cwd_path),
            anchor_pane=self.anchor_pane,
            intent=intent,
        )
        seed_trust(cwd_path)  # BEFORE spawn — the dialog eats input
        proc = await self._spawn_broker(record, adopt=None)
        record.pid = proc.pid
        self._procs[name] = proc
        self.registry.upsert(record)
        self.emit(SessionStatusChanged(name, record.state))
        self._publish_fleet()
        return f"spawned session {name} (pid {proc.pid}) in {cwd_path}"

    async def reassign_session(self, session_id: str, intent: str) -> str:
        """Hand a live session to a freshly spawned broker with a new task.

        The Claude session, its pane and its transcript survive; only the
        broker driving them is replaced. The new broker reuses the session's
        socket path, because ``BROKER_SOCKET`` was baked into the pane's
        environment when it was split and cannot be changed afterwards.

        Args:
            session_id: Registry name of the session to hand over.
            intent: Raw developer intent for the new task.

        Returns:
            A confirmation line naming the session and the new broker's pid.

        Raises:
            ValueError: The registry does not know the session's pane, Claude
                session id or transcript path.
            RuntimeError: A broker is still answering on the session socket.
        """
        record = self.registry.get(session_id)
        # BEFORE anything is torn down: an unreassignable session must not be
        # left with its old broker killed and no replacement.
        adopt = _adoption_fields(record)
        await self.stop_session(session_id)
        await self._require_socket_free(record)
        record.intent = intent
        record.approved_prompt = None  # superseded; set again on approval
        record.title = ""
        record.budget_count = 0
        proc = await self._spawn_broker(record, adopt=adopt)
        record.pid = proc.pid
        self._procs[session_id] = proc
        self.registry.upsert(record)
        self._set_state(session_id, SessionState.SPAWNING)
        return (
            f"session {session_id} reassigned to a new broker (pid {proc.pid})"
        )

    async def attach_session(self, session_id: str) -> str:
        """Bind a fresh broker to a session whose own broker is gone.

        A pure resume: the persisted approved prompt and budget count carry
        over untouched, no new intent is taken, no grounding runs and no
        proposal comes back. The new broker reuses the session's socket path,
        because ``BROKER_SOCKET`` was baked into the pane's environment when
        it was split and cannot be changed afterwards.

        Args:
            session_id: Registry name of the session to reattach.

        Returns:
            A confirmation line naming the session and the new broker's pid.

        Raises:
            KeyError: No such session — a session whose pane was found gone is
                removed from the registry, so there is nothing left to attach.
            ValueError: The registry only partly knows the session, or no
                approved prompt was ever persisted for it.
            RuntimeError: A broker is still answering on the session socket.
        """
        record = self.registry.get(session_id)  # KeyError if gone or unknown
        # Every refusal fires before any side effect.
        adopt = _adoption_fields(record)
        if record.approved_prompt is None:
            raise ValueError(
                f"session {session_id} has no persisted approved prompt to "
                "resume — its broker died before a prompt was approved. Use "
                "reassign_session with a new task instead."
            )
        # One probe, never a poll: nothing was stopped, so waiting cannot
        # free the socket. Anything alive or ambiguous refuses.
        if await _broker_is_listening(Path(record.socket_path)):
            raise RuntimeError(
                f"session {session_id}: a broker is still answering on "
                f"{record.socket_path} — refusing to attach"
            )
        # The dead broker's stranded escalations, both raiser identities: a
        # decision dispatched to one would be discarded, and a live
        # same-raiser entry would refuse the resumed broker's first
        # escalation. It re-raises if the situation still holds.
        await self._retract_stranded_escalation(session_id, "broker")
        await self._retract_stranded_escalation(session_id, "permission")
        resume = ResumedTask(
            approved_prompt=record.approved_prompt,
            completed=record.state == SessionState.COMPLETED,
        )
        proc = await self._spawn_broker(record, adopt=adopt, resume=resume)
        record.pid = proc.pid
        self._procs[session_id] = proc
        self.registry.upsert(record)
        self._set_state(session_id, SessionState.SPAWNING)
        return (
            f"session {session_id} reattached to a new broker (pid {proc.pid})"
        )

    async def reactivate_session(self, session_id: str, intent: str) -> str:
        """Give a completed session a new task without replacing its broker.

        Args:
            session_id: Registry name of the completed session.
            intent: Raw developer intent for the new task.

        Returns:
            A confirmation line, or a rejection when the session is not
            completed and so has a task it is still driving.
        """
        record = self.registry.get(session_id)
        env = self._env(
            T_REACTIVATE, ReactivatePayload(intent=intent).model_dump()
        )
        rejected = await self._deliver(
            record.socket_path,
            env,
            rejection=f"session {session_id} refused reactivation",
        )
        if rejected is not None:
            self.emit(Notice(rejected))
            return rejected
        record.intent = intent
        record.approved_prompt = None  # superseded; set again on approval
        record.title = ""
        record.budget_count = 0
        self.registry.upsert(record)
        self._set_state(session_id, SessionState.GROUNDING)
        return f"session {session_id} reactivated — grounding the new task"

    async def approve_prompt(
        self, proposal_id: str, prompt: str, title: str
    ) -> str:
        """Approve a pending prompt proposal and send it to its session.

        Args:
            proposal_id: Identifier of the proposal being answered.
            prompt: Prompt text to send — the developer's edit of the
                proposal, or the proposal verbatim.
            title: Short task label shown next to the session in the fleet.

        Returns:
            An outcome line: approved, unknown proposal, or rejected as stale.
        """
        pending = self.proposals.get(proposal_id)
        if pending is None:
            msg = f"unknown proposal {proposal_id!r} — nothing approved"
            self.emit(Notice(msg))
            return msg
        name = pending.session_name
        record = self.registry.get(name)
        env = self._env(
            T_APPROVE_PROMPT,
            ApprovePromptPayload(
                proposal_id=proposal_id, prompt=prompt
            ).model_dump(),
        )
        rejected = await self._deliver(
            record.socket_path,
            env,
            rejection=(
                f"session {name} rejected approval for proposal "
                f"{proposal_id} (stale)"
            ),
        )
        if rejected is not None:
            self.emit(Notice(rejected))
            return rejected
        del self.proposals[proposal_id]
        record.approved_prompt = prompt  # the AUTHORITATIVE intent
        record.title = title
        self.registry.upsert(record)
        self._publish_fleet()
        return f"prompt approved for session {name}"

    async def dispatch(self, escalation_id: str, decision: str) -> str:
        """Dispatch a decision to the session whose escalation it answers.

        Args:
            escalation_id: The escalation the decision answers.
            decision: The developer's decision, sent verbatim.

        Returns:
            An outcome line: dispatched to the named session, a refusal
            naming the escalation that is no longer live, a refusal naming the
            pane when the escalation is a permission prompt, or a rejection if
            the session NACKed delivery.
        """
        # Liveness is checked THE INSTANT before the write, not at
        # surface time. A stale dispatch is the worst failure this system
        # can produce.
        active = self.queue.active
        if active is None or active.escalation_id != escalation_id:
            msg = (
                f"decision NOT dispatched — escalation {escalation_id} is "
                "no longer live"
            )
            self.emit(Notice(msg))
            return msg
        if isinstance(active, PermissionEscalationPayload):
            # The native prompt is the only thing that can answer it, and it is
            # on the session's own screen.
            pane_id = self.pane_of(active.session_id)
            msg = (
                f"decision NOT dispatched — escalation {escalation_id} is a "
                f"permission prompt in session {active.session_id}. The "
                f"developer answers it in pane {pane_id}."
            )
            self.emit(Notice(msg))
            return msg
        record = self.registry.get(active.session_id)
        env = self._env(
            T_DISPATCH_DECISION,
            DispatchDecisionPayload(
                escalation_id=escalation_id, response=decision
            ).model_dump(),
        )
        rejected = await self._deliver(
            record.socket_path,
            env,
            rejection=(
                f"session {record.name} rejected the dispatched decision "
                f"for escalation {escalation_id} (stale)"
            ),
        )
        if rejected is not None:
            self.emit(Notice(rejected))
            return rejected
        # Resolve on the accept-ACK, not on pane delivery: a missed delivery
        # comes back via T_DECISION_UNDELIVERED.
        self.queue.resolve(escalation_id)
        self._surfaced_id = None
        # No state write: DRIVING is the broker's transition to report, and
        # its push carries it — the master never invents an operating state.
        self._publish_fleet()
        await self._surface_head()
        return f"decision dispatched to session {record.name}"

    async def _on_decision_undelivered(
        self, env: Envelope, name: str, p: DecisionUndeliveredPayload
    ) -> Response:
        """Surface a dispatched decision that failed to reach the pane.

        Args:
            env: The undelivered-decision envelope.
            name: Session that reported the miss.
            p: The escalation the decision answered and why it did not land.

        Returns:
            The ACK.
        """
        self.emit(
            Notice(
                f"decision for escalation {p.escalation_id} did NOT reach "
                f"session {name}: {p.detail}"
            )
        )
        return self._ack(env, ok=True)

    async def clarify_escalation(self, escalation_id: str, question: str) -> str:
        """Relay a read-only question about the live escalation to its broker.

        The escalation stays pending. The broker's answer is shown to the
        developer verbatim (a Notice); this returns only an acknowledgement to
        the tool loop, so the master never rewrites developer-facing text.

        Args:
            escalation_id: The escalation the question is about.
            question: The developer's question, sent verbatim.

        Returns:
            An acknowledgement line, or a refusal naming why no answer was
            obtained (not the head, a permission prompt, or the broker declined).
        """
        active = self.queue.active
        if active is None or active.escalation_id != escalation_id:
            msg = f"question NOT sent — escalation {escalation_id} is no longer live"
            self.emit(Notice(msg))
            return msg
        if isinstance(active, PermissionEscalationPayload):
            pane_id = self.pane_of(active.session_id)
            msg = (
                f"question NOT sent — escalation {escalation_id} is a permission "
                f"prompt in session {active.session_id}. The developer answers it "
                f"in pane {pane_id}."
            )
            self.emit(Notice(msg))
            return msg
        record = self.registry.get(active.session_id)
        env = self._env(
            T_CLARIFY_ESCALATION,
            ClarifyEscalationRequestPayload(
                escalation_id=escalation_id, question=question
            ).model_dump(),
        )
        resp = await client.request(
            Path(record.socket_path), env, timeout_s=CLARIFY_ESCALATION_TIMEOUT_S
        )
        if not resp.ok:
            reason = resp.payload.get("error")
            msg = (
                f"no clarification from session {record.name} for escalation "
                f"{escalation_id}"
            )
            if isinstance(reason, str) and reason:
                msg = f"{msg}: {reason}"
            self.emit(Notice(msg))
            return msg
        answer = ClarifyEscalationReplyPayload.model_validate(resp.payload).answer
        self.emit(
            Notice(f"Session {record.name} on escalation {escalation_id}:\n\n{answer}")
        )
        return f"clarification from session {record.name} shown to the developer"

    async def send_prompt(self, session_id: str, text: str) -> str:
        """Send a developer prompt straight to a session.

        Args:
            session_id: Registry name of the target session.
            text: Prompt text, sent verbatim.

        Returns:
            A confirmation line naming the session, or a rejection if the
            session NACKed delivery.
        """
        record = self.registry.get(session_id)
        env = self._env(T_SEND_PROMPT, SendPromptPayload(text=text).model_dump())
        rejected = await self._deliver(
            record.socket_path,
            env,
            rejection=f"session {session_id} rejected the prompt (stale)",
        )
        if rejected is not None:
            self.emit(Notice(rejected))
            return rejected
        record.budget_count = 0  # developer prompt resets the budget
        self.registry.upsert(record)
        self._publish_fleet()
        return f"prompt sent to session {session_id}"

    async def probe_status(self, session_id: str) -> StatusPayload:
        """Ask a session for its status and fold the reply into the registry.

        Args:
            session_id: Registry name of the session to probe.

        Returns:
            The status as reported by the session broker.
        """
        socket_path = Path(self.registry.get(session_id).socket_path)
        env = self._env(T_STATUS, {})
        resp = await client.request(socket_path, env, timeout_s=REQUEST_TIMEOUT_S)
        status = StatusPayload.model_validate(resp.payload)
        record = self.registry.get(session_id)
        record.pane_id = status.pane_id or record.pane_id
        record.claude_session_id = (
            status.claude_session_id or record.claude_session_id
        )
        record.transcript_path = status.transcript_path or record.transcript_path
        self.registry.upsert(record)
        self._set_state(session_id, status.state)
        self._note_task_activity(session_id, status.task_activity)
        return status

    async def get_decision_log(self, session_id: str) -> str:
        """Fetch a session's decision log as text.

        Args:
            session_id: Registry name of the session to query.

        Returns:
            The log text verbatim.

        Raises:
            ValidationError: The session's reply was not a decision log.
        """
        record = self.registry.get(session_id)
        env = self._env(T_GET_DECISION_LOG, {})
        resp = await client.request(
            Path(record.socket_path), env, timeout_s=REQUEST_TIMEOUT_S
        )
        return DecisionLogPayload.model_validate(resp.payload).text

    async def get_permission_log(self, session_id: str) -> str:
        """Fetch a session's permission log as text.

        Args:
            session_id: Registry name of the session to query.

        Returns:
            The log text verbatim.

        Raises:
            ValidationError: The session's reply was not a permission log.
        """
        record = self.registry.get(session_id)
        env = self._env(T_GET_PERMISSION_LOG, {})
        resp = await client.request(
            Path(record.socket_path), env, timeout_s=REQUEST_TIMEOUT_S
        )
        return PermissionLogPayload.model_validate(resp.payload).text

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
        await self._retract_stranded_escalation(session_id, "permission")
        self._set_state(session_id, SessionState.STOPPED)
        return f"session {session_id} stopped"

    def pane_of(self, session_id: str) -> str:
        """Return the pane holding a session, or ``PANE_UNKNOWN``.

        Args:
            session_id: Registry name of the session.

        Returns:
            The pane id, or ``PANE_UNKNOWN`` when the registry has none.
        """
        try:
            return self.registry.get(session_id).pane_id or PANE_UNKNOWN
        except KeyError:
            return PANE_UNKNOWN

    def pending_proposals(self) -> list[PendingProposal]:
        """Return every proposal awaiting approval, oldest first."""
        return list(self.proposals.values())

    def render_registry_summary(self) -> str:
        """Render the registry summary for the LLM context and list_sessions.

        Returns:
            One line per session, or ``"(no sessions)"``.
        """
        if not self.registry.records:
            return "(no sessions)"
        lines: list[str] = []
        for name in sorted(self.registry.records, key=session_sort_key):
            r = self.registry.records[name]
            intent = r.approved_prompt or r.intent
            lines.append(
                f"- {name}: state={r.state} "
                f"budget={r.budget_count}/{self.cfg.budget_max} "
                f"cwd={r.cwd} intent={intent}"
            )
        return "\n".join(lines)

    async def render_sessions_with_permission_prompts(self) -> str:
        """Render the registry summary, probing each session for a live prompt.

        The prompt flag lives in broker memory and is read on demand, so it
        never enters the summary the master carries into every turn.

        Returns:
            The registry summary, followed by a line for each session found
            waiting on a native permission prompt and for each session that
            could not be reached.
        """
        lines = [self.render_registry_summary()]
        for name in sorted(self.registry.records, key=session_sort_key):
            try:
                status = await self.probe_status(name)
            except PROBE_FAILURES as exc:
                # An unreachable session costs one line of the listing, never
                # the whole listing.
                lines.append(
                    f"- {name}: unreachable ({exc!r}) — could not read "
                    "whether it is sitting on a permission prompt"
                )
                continue
            if status.permission_prompt:
                lines.append(
                    f"- {name}: sitting on a permission prompt, answered in "
                    f"pane {status.pane_id or PANE_UNKNOWN}"
                )
        return "\n".join(lines)

    # ── internals ────────────────────────────────────────────────────────────

    async def _spawn_broker(
        self,
        record: SessionRecord,
        *,
        adopt: AdoptedSession | None,
        resume: ResumedTask | None = None,
    ) -> asyncio.subprocess.Process:
        """Start a session-broker subprocess for ``record``.

        Args:
            record: Supplies the identity, socket, cwd, intent and budget the
                broker starts from.
            adopt: Pane, Claude session and transcript of a running session the
                broker takes over; ``None`` starts a fresh one.
            resume: Approved prompt and completed-ness the broker resumes
                instead of grounding; ``None`` grounds a new task.

        Returns:
            The spawned process.
        """
        settings_path = self.paths.session_claude_settings(record.name)
        # The rules have to be on disk before the session reads them: the
        # master is the only writer of anything outside the repo.
        write_session_permissions(
            settings_path, self.cfg.permission_rules.model_dump()
        )
        config = SessionBrokerConfig(
            name=record.name,
            socket_path=record.socket_path,
            master_socket_path=str(self.master_socket_path),
            broker_home=self.paths.home,
            cwd=record.cwd,
            anchor_pane=record.anchor_pane,
            intent=record.intent,
            budget_count=record.budget_count,
            model_id=self.cfg.model_id,
            max_tokens=self.cfg.max_tokens,
            classifier=self.cfg.classifier,
            embedding=self.cfg.embedding,
            watchdog_seconds=self.cfg.watchdog_seconds,
            budget_max=self.cfg.budget_max,
            claude_settings_path=str(settings_path),
            adopt=adopt,
            resume=resume,
        )
        # By module string, never by import — keeps the module boundary
        # structural.
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "broker.session",
            "--config-json",
            config.model_dump_json(),
        )

    async def _require_socket_free(self, record: SessionRecord) -> None:
        """Block until nothing answers on a session's socket.

        The replacement broker binds the same path, and a unix socket rebind
        over a live listener succeeds silently — two brokers would then split
        the pane's hook traffic between them.

        Args:
            record: Registry record of the session being reassigned.

        Raises:
            RuntimeError: A broker was still accepting connections after
                ``STOP_WAIT_S``.
        """
        path = Path(record.socket_path)
        deadline = asyncio.get_running_loop().time() + STOP_WAIT_S
        while await _broker_is_listening(path):
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"session {record.name}: a broker is still serving "
                    f"{path} after {STOP_WAIT_S:.0f} s — refusing to reassign"
                )
            await asyncio.sleep(SOCKET_POLL_S)

    async def _retract_stranded_escalation(
        self, session_id: str, component: Literal["broker", "permission"]
    ) -> None:
        """Clear a queued escalation whose raiser died with the broker.

        An escalation is normally withdrawn by the component that raised it.
        With the broker gone nothing is left to withdraw it, and it would
        wait in the queue indefinitely.

        Args:
            session_id: Session whose broker is gone.
            component: Raiser component whose live entry is cleared.
        """
        cleared = self.queue.retract_for_raiser(
            RaiserIdentity(component=component, session_id=session_id)
        )
        if cleared is None:
            return
        if cleared.escalation_id == self._surfaced_id:
            self._surfaced_id = None
        self.emit(
            Notice(
                f"escalation {cleared.escalation_id} from session "
                f"{session_id} retracted: its broker is gone"
            )
        )
        self._publish_fleet()
        await self._surface_head()

    async def _on_session_ended(self, env: Envelope, name: str) -> Response:
        """Retire a session whose broker reported its ``SessionEnd``.

        Args:
            env: Envelope carrying the terminal report.
            name: Session that ended.

        Returns:
            The ACK, sent even when the session is already gone.
        """
        if name in self.registry.records:
            logger.info("session %s: ended (/exit) — removed from the fleet", name)
            self._activity.pop(name, None)
            self._task_activity.pop(name, None)
            self._permission_prompt_pending.discard(name)
            self._discard_proposals(name)
            self._procs.pop(name, None)
            self.registry.remove(name)
            # The broker exits without withdrawing a live escalation of either
            # kind, so retract both — a stranded head would wedge the FIFO
            # queue, undispatchable to a gone session.
            await self._retract_stranded_escalation(name, "broker")
            await self._retract_stranded_escalation(name, "permission")
            self.emit(
                Notice(f"session {name} ended (/exit) — removed from the fleet")
            )
            self._publish_fleet()
        return self._ack(env, ok=True)

    def _set_state(self, name: str, state: SessionState) -> bool:
        """Record a session's new state and tell the TUI, once per change.

        Args:
            name: Registry name of the session.
            state: New state to persist.

        Returns:
            ``True`` when the state changed, ``False`` when it was already
            ``state`` or the session is unknown.
        """
        # No ``await`` may sit between this read and the upsert below, or two
        # interleaving handlers would read-modify-write clobber each other
        # under N concurrent brokers.
        try:
            record = self.registry.get(name)
        except KeyError:
            self.emit(Notice(f"message from unknown session {name!r}"))
            return False
        if record.state == state:
            return False
        logger.info("session %s: %s -> %s", name, record.state, state)
        record.state = state
        self.registry.upsert(record)  # sync; no await before this point
        if state in _ABSORBING:
            # A settled session shows no live status, and the absorbing
            # guard blocks the pushes that would otherwise clear these.
            self._activity.pop(name, None)
            self._task_activity.pop(name, None)
            self._permission_prompt_pending.discard(name)
            self._discard_proposals(name)
        self.emit(SessionStatusChanged(name, state))
        self._publish_fleet()
        return True

    def _note_task_activity(self, name: str, text: str) -> None:
        """Show ``text`` as a live session's task activity; a settled session shows none."""
        if not text or self.registry.get(name).state in _ABSORBING:
            self._task_activity.pop(name, None)
        else:
            self._task_activity[name] = text

    def build_fleet_view(self) -> FleetView:
        """Assemble the structured sidebar view from current runtime state."""
        badges = self._badges_by_session()
        rows: list[SessionRow] = []
        for name in sorted(self.registry.records, key=session_sort_key):
            r = self.registry.records[name]
            rows.append(
                SessionRow(
                    session_id=name,
                    state=r.state,
                    title=r.title,
                    task_activity=self._task_activity.get(name, ""),
                    broker_activity=self._activity.get(name, ""),
                    budget_count=r.budget_count,
                    budget_max=self.cfg.budget_max,
                    badges=badges.get(name, ()),
                    pane_id=r.pane_id,
                )
            )
        return FleetView(
            master_activity=self._master_activity,
            rows=tuple(rows),
            queue_depth=self.queue.depth,
            waiting=self.queue.waiting,
            head=self._head_request(),
        )

    def _head_request(self) -> HeadRequest | None:
        head = self.queue.active
        if head is None:
            return None
        if isinstance(head, PermissionEscalationPayload):
            return HeadRequest(
                head.session_id,
                head.escalation_id,
                Attention.PERMISSION,
                head.tool_name,
            )
        return HeadRequest(
            head.session_id,
            head.escalation_id,
            Attention.ESCALATION,
            head.what_was_asked,
        )

    def _badges_by_session(self) -> dict[str, tuple[Attention, ...]]:
        """Distinct attention badges per session, from the three live stores."""
        acc: dict[str, set[Attention]] = {}
        for entry in self.queue.entries:
            kind = (
                Attention.PERMISSION
                if entry.raiser.component == "permission"
                else Attention.ESCALATION
            )
            acc.setdefault(entry.session_id, set()).add(kind)
        for pending in self.proposals.values():
            acc.setdefault(pending.session_name, set()).add(Attention.PROPOSAL)
        for name in self._permission_prompt_pending:
            acc.setdefault(name, set()).add(Attention.PERMISSION)
        return {
            name: tuple(sorted(kinds, key=lambda a: a.value))
            for name, kinds in acc.items()
        }

    def _register_proposal(self, name: str, payload: PromptProposalPayload) -> None:
        """Store a pending proposal and surface it to the developer."""
        self.proposals[payload.proposal_id] = PendingProposal(name, payload)
        self.emit(
            ProposalArrived(name, payload.proposal_id, render_proposal(payload))
        )

    def _discard_proposals(self, name: str) -> None:
        """Drop any pending proposal a session left behind on settling or exit."""
        for pid in [
            pid for pid, p in self.proposals.items() if p.session_name == name
        ]:
            del self.proposals[pid]

    def _publish_fleet(self) -> None:
        """Emit the current structured sidebar snapshot."""
        self.emit(FleetUpdated(self.build_fleet_view()))

    def note_master_activity(self, text: str) -> None:
        """Show what the master itself is doing on the dashboard header."""
        self._master_activity = text
        self._publish_fleet()

    def clear_master_activity(self) -> None:
        """Return the dashboard header to idle."""
        self._master_activity = None
        self._publish_fleet()

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
            self.emit(Notice(f"notification failed: {exc}"))

    async def _deliver(
        self, socket_path: str, env: Envelope, *, rejection: str
    ) -> str | None:
        """Send ``env`` to a session socket, reporting a NACK.

        Args:
            socket_path: Session socket to write to.
            env: Envelope to send.
            rejection: Message logged and returned when the session NACKs.

        Returns:
            ``None`` when the session ACKed, otherwise ``rejection``, with the
            broker's own reason appended when it sent one.
        """
        resp = await client.request(
            Path(socket_path), env, timeout_s=REQUEST_TIMEOUT_S
        )
        if resp.ok:
            return None
        reason = resp.payload.get("error")
        if isinstance(reason, str) and reason:
            rejection = f"{rejection}: {reason}"
        logger.warning("%s", rejection)
        return rejection

    def _env(self, msg_type: str, payload: dict[str, Any]) -> Envelope:
        """Wrap a payload in an envelope with a fresh message id."""
        return Envelope(id=uuid.uuid4().hex, type=msg_type, payload=payload)

    def _ack(self, env: Envelope, *, ok: bool) -> Response:
        """Build the ACK or NACK answering an envelope."""
        return Response(id=env.id, ok=ok)

    def _nack(self, env: Envelope, error: str, reason_code: str) -> Response:
        """Build a refusal a sender can act on without parsing the message.

        Args:
            env: Envelope being answered.
            error: Human-readable reason, carried for the developer.
            reason_code: Machine-readable reason from the closed NACK set.

        Returns:
            The refusal.
        """
        return Response(
            id=env.id,
            ok=False,
            payload={"error": error, "reason_code": reason_code},
        )
