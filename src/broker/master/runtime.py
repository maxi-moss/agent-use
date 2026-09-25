"""Master runtime layer: socket server, decision-escalation queue, open
pane escalations, session spawn/stop, dispatch with
liveness-at-dispatch, and the ONLY renderers of broker payloads.

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
from typing import Any

from pydantic import ValidationError

from broker import decision_log
from broker.claude.settings import write_session_permissions
from broker.claude.trust import seed_trust
from broker.config import (
    AdoptedSession,
    BrokerConfig,
    ResumedTask,
    SessionBrokerConfig,
)
from broker.herdr import driver
from broker.paths import BrokerPaths
from broker.master import notifier
from broker.master.outcome import SessionOutcome, build_outcome
from broker.master.viewmodel import (
    Attention,
    CompletionArrived,
    EscalationArrived,
    EventSink,
    FleetUpdated,
    FleetView,
    HeadRequest,
    Notice,
    PaneEscalationArrived,
    PaneRequest,
    ProposalArrived,
    SessionRow,
    SessionStateChanged,
)
from broker.master.pane_escalations import PaneEscalations, PaneProtocolViolation
from broker.master.queue import EscalationProtocolViolation, EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.protocol import client
from broker.protocol.constants import (
    NACK_MALFORMED,
    NACK_PROTOCOL_VIOLATION,
    NACK_UNKNOWN_SESSION,
    PaneKind,
    SessionState,
    T_APPROVE_PROMPT,
    T_CLARIFY_ESCALATION,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DECISION_DELIVERED,
    T_DECISION_UNDELIVERED,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_ESCALATION_RETRACT,
    T_FATAL_ERROR,
    T_GET_DECISION_LOG,
    T_GET_PERMISSION_LOG,
    T_LIVE_STATUS,
    T_PANE_ESCALATION,
    T_PANE_RETRACT,
    T_PROMPT_PROPOSAL,
    T_PROMPT_UNDELIVERED,
    T_REACTIVATE,
    T_SEND_PROMPT,
    T_SESSION_ENDED,
    T_SHUTDOWN,
    T_STATUS,
)
from broker.protocol.schemas import (
    PANE_ESCALATION_ADAPTER,
    ApprovePromptPayload,
    ClarifyEscalationReplyPayload,
    ClarifyEscalationRequestPayload,
    BudgetUpdatePayload,
    CompletionPayload,
    DecisionDeliveredPayload,
    DecisionLogPayload,
    DecisionUndeliveredPayload,
    DispatchDecisionPayload,
    Envelope,
    EscalationDisclosure,
    EscalationPayload,
    EscalationRetractPayload,
    FatalErrorPayload,
    LiveStatusPayload,
    PaneEscalationPayload,
    PaneRetractPayload,
    PermissionEscalationPayload,
    PermissionLogPayload,
    PermissionSuggestion,
    PromptProposalPayload,
    PromptUndeliveredPayload,
    QuestionEscalationPayload,
    ReactivatePayload,
    Response,
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
AGENT_PROBE_TIMEOUT_S = 5.0

# Stands in for a pane the registry cannot name. A pane escalation is still
# worth surfacing without it: the developer knows the session.
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


def _drop_from_fleet(
    registry: Registry,
    queue: EscalationQueue,
    panes: PaneEscalations,
    name: str,
) -> list[str]:
    """Remove a session no broker can drive again, retracting its escalations.

    Args:
        registry: Session registry; the removal persists on its own.
        queue: Persisted escalation queue.
        panes: Persisted pane escalations.
        name: Session to drop.

    Returns:
        One line per retracted escalation.
    """
    lines: list[str] = []
    queued = queue.retract_for_session(name)
    if queued is not None:
        lines.append(
            f"session {name}: queued escalation {queued.escalation_id} "
            "retracted — no decision can reach it"
        )
    for prompt in panes.retract_for_session(name):
        lines.append(
            f"session {name}: {prompt.kind} escalation "
            f"{prompt.escalation_id} retracted — no broker will see it closed"
        )
    registry.remove(name)
    return lines


async def reconcile_registry(
    registry: Registry,
    queue: EscalationQueue,
    panes: PaneEscalations,
) -> list[str]:
    """Classify every registry session at startup, dropping finished ones.

    Probes are client-side connects and herdr reads only — nothing is spawned
    and nothing binds a socket. A session with no answering broker is dropped
    once its Claude Code no longer runs, or when the registry lacks what a
    replacement broker needs to adopt it. An inconclusive probe never drops a
    session — a wrong ``unmanaged`` costs the developer a glance, a wrong
    removal throws away queued decisions.

    Args:
        registry: Loaded session registry; a dropped session is removed, every
            other classification updates its state in place and is saved once.
        queue: Persisted escalation queue; a dropped session's queued
            escalation is retracted from it before the TUI re-announces
            the head.
        panes: Persisted pane escalations; a dropped session's open ones are
            retracted before the TUI re-announces what remains.

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
        try:
            running = await asyncio.to_thread(
                driver.agent_running, name, timeout_s=AGENT_PROBE_TIMEOUT_S
            )
        except Exception as exc:
            record.state = SessionState.UNMANAGED
            warnings.append(
                f"session {name}: agent probe inconclusive ({exc!r}) — "
                "marked unmanaged"
            )
            continue
        if not running:
            warnings.append(
                f"session {name}: Claude Code no longer runs — it exited or "
                "its pane closed; removed"
            )
            warnings.extend(_drop_from_fleet(registry, queue, panes, name))
            continue
        try:
            _adoption_fields(record)
        except ValueError as exc:
            warnings.append(
                f"session {name}: Claude Code still runs but no broker can "
                f"take it over ({exc}) — removed; its pane is left untouched"
            )
            warnings.extend(_drop_from_fleet(registry, queue, panes, name))
            continue
        record.state = SessionState.UNMANAGED
        line = (
            f"session {name}: Claude Code alive with nothing driving it — "
            "marked unmanaged; recover it with attach_session"
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


def _render_disclosure_sections(d: EscalationDisclosure) -> list[str]:
    """Render a broker's disclosure as block lines, verbatim."""
    lines = [
        "## Title",
        d.escalation_title,
        "",
        "## Situation",
        d.situation,
        "",
        "## What was asked",
        d.what_was_asked,
        "",
        "## What is at stake",
        d.what_is_at_stake,
        "",
        "## Alternatives",
    ]
    for alt in d.alternatives:
        lines += [
            f"- {alt.option}",
            f"  pros: {alt.pros}",
            f"  cons: {alt.cons}",
        ]
    lines += [
        "",
        "## Recommendation",
        d.recommendation,
        "",
        "## Uncertainty",
        d.uncertainty,
        "",
        "## What would change my mind",
        d.what_would_change_my_mind,
    ]
    return lines


def render_escalation(p: EscalationPayload) -> str:
    """Render the decision-escalation block, deterministic and verbatim.

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
        *_render_disclosure_sections(p.disclosure),
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


def render_question_escalation(p: QuestionEscalationPayload, pane_id: str) -> str:
    """Render the question-escalation block, deterministic and verbatim.

    Args:
        p: Validated question-escalation payload from a session broker.
        pane_id: Pane holding the AskUserQuestion menu, or ``PANE_UNKNOWN``.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    lines = [
        f"Question escalation {p.escalation_id} — session {p.session_id}",
        "",
        f"The developer answers this in pane {pane_id}, on the AskUserQuestion "
        "menu already waiting there. It cannot be answered here, and no "
        "decision sent from here reaches it.",
        "",
        "## Task context",
        p.task_context,
        "",
        "## Menu",
    ]
    lines += [
        p.menu or "(the menu could not be read — see the pane)",
        "",
        "## Why the broker did not answer",
        p.reason,
    ]
    if p.analysis is not None:
        lines += ["", *_render_disclosure_sections(p.analysis)]
    return "\n".join(lines)


def render_pane_escalation(p: PaneEscalationPayload, pane_id: str) -> str:
    """Render a pane-escalation block of either kind, deterministic and verbatim.

    Args:
        p: Validated pane-escalation payload.
        pane_id: Pane holding the native prompt, or ``PANE_UNKNOWN``.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    if isinstance(p, PermissionEscalationPayload):
        return render_permission_escalation(p, pane_id)
    return render_question_escalation(p, pane_id)


def pane_label(p: PaneEscalationPayload) -> str:
    """Name a pane escalation in one line: the tool, or the menu's first question."""
    if isinstance(p, PermissionEscalationPayload):
        return p.tool_name
    return p.first_question or "(unreadable menu)"


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

    session_id: str
    payload: PromptProposalPayload


class MasterRuntime:
    def __init__(
        self,
        emit: EventSink,
        registry: Registry,
        queue: EscalationQueue,
        panes: PaneEscalations,
        cfg: BrokerConfig,
        *,
        anchor_pane: str,
    ) -> None:
        """Wire the runtime to its frontend sink, its persisted stores and the config.

        Args:
            emit: Receives every renderer-neutral view event the runtime produces.
            registry: Loaded session registry.
            queue: Loaded decision-escalation queue.
            panes: Loaded open pane escalations.
            cfg: Broker configuration.
            anchor_pane: Herdr pane every spawned session is anchored to.
        """
        self.emit = emit
        self.registry = registry
        self.cfg = cfg
        self.anchor_pane = anchor_pane
        self.queue = queue
        self.panes = panes
        self._surfaced_id: str | None = None
        # Escalation whose decision has been dispatched but not yet confirmed
        # delivered to the pane. Memory-only: an unconfirmed dispatch simply
        # re-surfaces on restart, and the developer re-decides.
        self._inflight: str | None = None
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
        # A head or open pane escalation loaded from disk has never been
        # announced in this process, so each is announced here, exactly once.
        self._publish_fleet()
        await self._surface_head()
        for prompt in self.open_pane_escalations():
            await self._announce_pane_escalation(prompt)
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
        if env.type == T_PANE_ESCALATION:
            return await self._on_pane_escalation(env, name)
        if env.type == T_COMPLETION:
            p = CompletionPayload.model_validate(env.payload)
            self._set_state(name, SessionState.COMPLETED)
            self.emit(CompletionArrived(name, p.headline, p.supporting))
            await self._notify(
                notifier.notify_done, f"Session {name} complete", p.headline
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
            await self._retract_stranded_escalation(name)
            self._retract_stranded_pane_escalations(name)
            await self._notify(
                notifier.notify_request,
                f"Session {name} failed",
                f"{p.error_class}: {p.detail}",
            )
            return self._ack(env, ok=True)
        if env.type == T_ESCALATION_RETRACT:
            rp = EscalationRetractPayload.model_validate(env.payload)
            return await self._on_escalation_retract(env, name, rp)
        if env.type == T_PANE_RETRACT:
            pp = PaneRetractPayload.model_validate(env.payload)
            return self._on_pane_retract(env, name, pp)
        if env.type == T_PROMPT_UNDELIVERED:
            up = PromptUndeliveredPayload.model_validate(env.payload)
            self.emit(
                Notice(f"prompt for session {name} did NOT reach its pane: {up.detail}")
            )
            return self._ack(env, ok=True)
        if env.type == T_PROMPT_PROPOSAL:
            p = PromptProposalPayload.model_validate(env.payload)
            self._set_state(name, SessionState.AWAITING_APPROVAL)
            self._register_proposal(name, p)
            self._publish_fleet()
            return self._ack(env, ok=True)
        if env.type == T_BUDGET_UPDATE:
            p = BudgetUpdatePayload.model_validate(env.payload)
            record = self.registry.get(name)
            record.budget_count = p.count
            self.registry.upsert(record)
            self._publish_fleet()
            return self._ack(env, ok=True)
        if env.type == T_DECISION_DELIVERED:
            dp = DecisionDeliveredPayload.model_validate(env.payload)
            return await self._on_decision_delivered(env, name, dp)
        if env.type == T_DECISION_UNDELIVERED:
            p = DecisionUndeliveredPayload.model_validate(env.payload)
            return await self._on_decision_undelivered(env, name, p)
        if env.type == T_LIVE_STATUS:
            p = LiveStatusPayload.model_validate(env.payload)
            rec = self.registry.records.get(name)
            if rec is not None:
                # Outside the absorbing guard: /clear in a settled session
                # binds a new Claude session id.
                self._note_identity(rec, p)
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
        try:
            self.queue.accept(p)
        except EscalationProtocolViolation as exc:
            self.emit(Notice(f"PROTOCOL VIOLATION: {exc}"))
            return self._nack(env, str(exc), NACK_PROTOCOL_VIOLATION)
        self._set_state(p.session_id, SessionState.ESCALATED)
        self._publish_fleet()
        await self._surface_head()
        return self._ack(env, ok=True)

    async def _on_pane_escalation(self, env: Envelope, name: str) -> Response:
        """Validate one pane escalation, hold it, and announce it at once.

        Args:
            env: Envelope carrying the pane-escalation payload.
            name: Session name from the envelope, used for rejection notices.

        Returns:
            ``Response(ok=True)`` once live, ``ok=False`` with a reason code
            if rejected.
        """
        try:
            p = PANE_ESCALATION_ADAPTER.validate_python(env.payload)
        except ValidationError as exc:
            # A pane escalation missing its session or content is not
            # something the developer could act on.
            self.emit(
                Notice(
                    f"MALFORMED pane escalation from session {name!r} — "
                    f"NOT surfaced.\nvalidation: {exc}\n"
                    f"raw payload: {env.payload!r}"
                )
            )
            return self._nack(
                env, f"malformed pane escalation: {exc}", NACK_MALFORMED
            )
        unknown = self._reject_unknown_session(
            env, p.session_id, f"{p.kind} escalation"
        )
        if unknown is not None:
            return unknown
        try:
            self.panes.accept(p)
        except PaneProtocolViolation as exc:
            self.emit(Notice(f"PROTOCOL VIOLATION: {exc}"))
            return self._nack(env, str(exc), NACK_PROTOCOL_VIOLATION)
        self._publish_fleet()
        await self._announce_pane_escalation(p)
        return self._ack(env, ok=True)

    async def _on_escalation_retract(
        self, env: Envelope, name: str, p: EscalationRetractPayload
    ) -> Response:
        """Clear a decision escalation its session resolved out of band.

        Args:
            env: The retract envelope.
            name: Session that withdrew the escalation.
            p: The escalation withdrawn and why.

        Returns:
            The ACK.
        """
        self._clear_inflight(p.escalation_id)
        cleared = self.queue.retract(p.escalation_id)
        if cleared is not None:
            # Raising it set ESCALATED here; withdrawing it must undo that
            # or the registry outlives the escalation it describes.
            self._set_state(name, SessionState.DRIVING)
        if cleared is not None and cleared.escalation_id == self._surfaced_id:
            # It was surfaced, so the developer must learn it is no longer
            # live; a waiting entry they never saw retracts silently.
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

    def _on_pane_retract(
        self, env: Envelope, name: str, p: PaneRetractPayload
    ) -> Response:
        """Clear a pane escalation whose native prompt is no longer open.

        Args:
            env: The retract envelope.
            name: Session whose prompt closed.
            p: The pane escalation withdrawn and why.

        Returns:
            The ACK.
        """
        cleared = self.panes.retract(p.escalation_id)
        if cleared is not None:
            self.emit(
                Notice(
                    f"{cleared.kind} escalation {p.escalation_id} from session "
                    f"{name} retracted: {p.reason}"
                )
            )
            self._publish_fleet()
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

    async def _surface_head(self) -> None:
        """Render, announce and notify the queue's head, exactly once."""
        head = self.queue.active
        if head is None or head.escalation_id == self._surfaced_id:
            return
        self._surfaced_id = head.escalation_id
        self.emit(
            EscalationArrived(
                head.session_id,
                head.escalation_id,
                head.disclosure.escalation_title,
                render_escalation(head),
            )
        )
        await self._notify(
            notifier.notify_request,
            f"Escalation from session {head.session_id}",
            head.disclosure.what_was_asked,
        )

    async def _announce_pane_escalation(self, p: PaneEscalationPayload) -> None:
        """Render, announce and notify one open pane escalation."""
        pane_id = self.pane_of(p.session_id)
        self.emit(
            PaneEscalationArrived(
                p.kind,
                p.session_id,
                p.escalation_id,
                render_pane_escalation(p, pane_id),
            )
        )
        title = (
            f"Permission prompt in session {p.session_id}"
            if p.kind == PaneKind.PERMISSION
            else f"Question in session {p.session_id}"
        )
        await self._notify(
            notifier.notify_request,
            title,
            f"{pane_label(p)} — answer it in pane {pane_id}",
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
        self.emit(SessionStateChanged(name, record.state))
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
            KeyError: The session ended while this call was in flight.
            ValueError: The registry does not know the session's pane, Claude
                session id or transcript path.
            RuntimeError: A broker is still answering on the session socket.
        """
        record = self.registry.get(session_id)
        # BEFORE anything is torn down: an unreassignable session must not be
        # left with its old broker killed and no replacement.
        adopt = _adoption_fields(record)
        await self.stop_session(session_id)
        record = self.registry.get(session_id)  # KeyError if ended meanwhile
        await self._require_socket_free(record)
        record = self.registry.get(session_id)  # KeyError if ended meanwhile
        record.intent = intent
        record.approved_prompt = None  # superseded; set again on approval
        record.title = ""
        record.budget_count = 0
        proc = await self._spawn_broker(record, adopt=adopt)
        record = self.registry.get(session_id)  # KeyError if ended meanwhile
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
        # ``record`` stays bound to the same registry object throughout (a
        # concurrent upsert mutates it in place); each re-`get` below is only
        # a liveness check, so the ``approved_prompt`` narrowing above holds.
        self.registry.get(session_id)  # KeyError if ended meanwhile
        # The dead broker's stranded escalations, of both kinds: a decision
        # dispatched to one would be discarded, and a live entry would refuse
        # the resumed broker's first raise. It re-raises if the situation
        # still holds.
        await self._retract_stranded_escalation(session_id)
        self.registry.get(session_id)  # KeyError if ended meanwhile
        self._retract_stranded_pane_escalations(session_id)
        resume = ResumedTask(
            approved_prompt=record.approved_prompt,
            completed=record.state == SessionState.COMPLETED,
        )
        proc = await self._spawn_broker(record, adopt=adopt, resume=resume)
        self.registry.get(session_id)  # KeyError if ended meanwhile
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
        name = pending.session_id
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
            pane when the escalation is a pane escalation, a refusal when a
            decision is already in flight for it, or a rejection if the session
            NACKed delivery.
        """
        refused = self._pane_escalation_refusal(
            escalation_id, "decision NOT dispatched"
        )
        if refused is not None:
            return refused
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
        if self._inflight is not None:
            # A decision is already on its way to the pane; a second would
            # double-submit the same escalation.
            msg = (
                f"decision NOT dispatched — a decision for escalation "
                f"{self._inflight} is already being delivered"
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
        # The ACK only confirms the broker accepted the decision for
        # processing. Resolution waits for T_DECISION_DELIVERED confirming it
        # reached the pane; until then the escalation stays surfaced and no
        # second decision may be dispatched for it.
        self._inflight = escalation_id
        self._publish_fleet()
        return f"decision dispatched to session {record.name}"

    def _pane_escalation_refusal(self, escalation_id: str, lead: str) -> str | None:
        """Refuse a master action aimed at a pane escalation, naming the pane.

        Args:
            escalation_id: The escalation the action targets.
            lead: Opening words of the refusal, naming what was not sent.

        Returns:
            The refusal, already shown to the developer, or ``None`` when
            ``escalation_id`` is not an open pane escalation.
        """
        prompt = self.panes.find(escalation_id)
        if prompt is None:
            return None
        what = (
            "a permission prompt"
            if prompt.kind == PaneKind.PERMISSION
            else "an AskUserQuestion menu"
        )
        msg = (
            f"{lead} — escalation {escalation_id} is {what} in session "
            f"{prompt.session_id}. The developer answers it in pane "
            f"{self.pane_of(prompt.session_id)}."
        )
        self.emit(Notice(msg))
        return msg

    def _clear_inflight(self, escalation_id: str) -> None:
        """Drop the in-flight dispatch marker if it names ``escalation_id``."""
        if self._inflight == escalation_id:
            self._inflight = None

    async def _on_decision_delivered(
        self, env: Envelope, name: str, p: DecisionDeliveredPayload
    ) -> Response:
        """Resolve an escalation now that its decision reached the pane.

        Args:
            env: The delivered-decision envelope.
            name: Session that confirmed delivery.
            p: The escalation whose decision landed.

        Returns:
            The ACK.
        """
        self._clear_inflight(p.escalation_id)
        if self.queue.resolve(p.escalation_id) is not None:
            # It was the live head: surface whatever is next. DRIVING is the
            # broker's transition to report; the master never invents it.
            self._surfaced_id = None
            self._publish_fleet()
            await self._surface_head()
        return self._ack(env, ok=True)

    async def _on_decision_undelivered(
        self, env: Envelope, name: str, p: DecisionUndeliveredPayload
    ) -> Response:
        """Handle a dispatched decision that did not reach the pane.

        Resolution waits for confirmed delivery, so the escalation was never
        resolved: a still-live miss (a failed pane write) stays surfaced for a
        re-decide, while a stale dispatch (the broker moved past it) drops the
        now orphaned queue entry.

        Args:
            env: The undelivered-decision envelope.
            name: Session that reported the miss.
            p: The escalation the decision answered, why it did not land, and
                whether the broker still holds it live.

        Returns:
            The ACK.
        """
        self._clear_inflight(p.escalation_id)
        if p.still_live:
            self.emit(
                Notice(
                    f"decision for escalation {p.escalation_id} did NOT reach "
                    f"session {name}: {p.detail}"
                )
            )
            return self._ack(env, ok=True)
        cleared = self.queue.retract(p.escalation_id)
        if cleared is not None and cleared.escalation_id == self._surfaced_id:
            self._surfaced_id = None
            self.emit(
                Notice(
                    f"escalation {p.escalation_id} from session {name} "
                    f"cleared: {p.detail}"
                )
            )
        self._publish_fleet()
        await self._surface_head()
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
            obtained (not the head, a pane escalation, or the broker declined).
        """
        refused = self._pane_escalation_refusal(escalation_id, "question NOT sent")
        if refused is not None:
            return refused
        active = self.queue.active
        if active is None or active.escalation_id != escalation_id:
            msg = f"question NOT sent — escalation {escalation_id} is no longer live"
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
            A confirmation that the session accepted the prompt, or a
            rejection carrying the session's reason if it NACKed.
        """
        record = self.registry.get(session_id)
        env = self._env(T_SEND_PROMPT, SendPromptPayload(text=text).model_dump())
        rejected = await self._deliver(
            record.socket_path,
            env,
            rejection=f"session {session_id} rejected the prompt",
        )
        if rejected is not None:
            self.emit(Notice(rejected))
            return rejected
        return f"prompt accepted by session {session_id}"

    async def probe_status(self, session_id: str) -> StatusPayload:
        """Ask a session for its status and record the state it reports.

        Args:
            session_id: Registry name of the session to probe.

        Returns:
            The status as reported by the session broker.
        """
        socket_path = Path(self.registry.get(session_id).socket_path)
        env = self._env(T_STATUS, {})
        resp = await client.request(socket_path, env, timeout_s=REQUEST_TIMEOUT_S)
        status = StatusPayload.model_validate(resp.payload)
        self._set_state(session_id, status.state)
        self._note_task_activity(session_id, status.task_activity)
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
        self._retract_stranded_pane_escalations(session_id)
        # A hard stop mid-delivery leaves no delivered/undelivered reply to
        # clear the in-flight marker. Drop it for this session's own head (the
        # broker escalation is intentionally kept) so a re-dispatch is not
        # refused as still being delivered.
        active = self.queue.active
        if active is not None and active.session_id == session_id:
            self._clear_inflight(active.escalation_id)
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
                    f"pane {self.pane_of(name)}"
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
        # structural. -I isolates the subprocess from the master's cwd and
        # PYTHONPATH.
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
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

    async def _retract_stranded_escalation(self, session_id: str) -> None:
        """Clear a queued decision escalation whose broker is gone.

        Args:
            session_id: Session whose broker is gone.
        """
        cleared = self.queue.retract_for_session(session_id)
        if cleared is None:
            return
        self._clear_inflight(cleared.escalation_id)
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

    def _retract_stranded_pane_escalations(self, session_id: str) -> None:
        """Clear the open pane escalations of a session whose broker is gone.

        Args:
            session_id: Session whose broker is gone.
        """
        cleared = self.panes.retract_for_session(session_id)
        for prompt in cleared:
            self.emit(
                Notice(
                    f"{prompt.kind} escalation {prompt.escalation_id} from "
                    f"session {session_id} retracted: its broker is gone"
                )
            )
        if cleared:
            self._publish_fleet()

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
            # The broker exits without withdrawing its live escalations, so
            # retract them all — a stranded head would wedge the FIFO queue,
            # undispatchable to a gone session.
            await self._retract_stranded_escalation(name)
            self._retract_stranded_pane_escalations(name)
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
        self.emit(SessionStateChanged(name, state))
        self._publish_fleet()
        return True

    def _note_identity(self, record: SessionRecord, p: LiveStatusPayload) -> None:
        """Persist the pane, Claude session and transcript a broker reports."""
        pane_id = p.pane_id or record.pane_id
        claude_session_id = p.claude_session_id or record.claude_session_id
        transcript_path = p.transcript_path or record.transcript_path
        if (pane_id, claude_session_id, transcript_path) == (
            record.pane_id,
            record.claude_session_id,
            record.transcript_path,
        ):
            return
        record.pane_id = pane_id
        record.claude_session_id = claude_session_id
        record.transcript_path = transcript_path
        self.registry.upsert(record)

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
            panes=tuple(
                PaneRequest(p.kind, p.session_id, p.escalation_id, pane_label(p))
                for p in self.open_pane_escalations()
            ),
        )

    def _head_request(self) -> HeadRequest | None:
        head = self.queue.active
        if head is None:
            return None
        return HeadRequest(
            head.session_id, head.escalation_id, head.disclosure.what_was_asked
        )

    def open_pane_escalations(self) -> list[PaneEscalationPayload]:
        """Return every open pane escalation, in numeric session order, then kind."""
        return sorted(
            self.panes.entries,
            key=lambda p: (session_sort_key(p.session_id), p.kind.value),
        )

    def _badges_by_session(self) -> dict[str, tuple[Attention, ...]]:
        """Distinct attention badges per session, from the four live sources."""
        acc: dict[str, set[Attention]] = {}
        for entry in self.queue.entries:
            acc.setdefault(entry.session_id, set()).add(Attention.ESCALATION)
        for prompt in self.panes.entries:
            badge = (
                Attention.PERMISSION
                if prompt.kind == PaneKind.PERMISSION
                else Attention.QUESTION
            )
            acc.setdefault(prompt.session_id, set()).add(badge)
        for pending in self.proposals.values():
            acc.setdefault(pending.session_id, set()).add(Attention.PROPOSAL)
        for name in self._permission_prompt_pending:
            acc.setdefault(name, set()).add(Attention.PERMISSION)
        return {
            name: tuple(sorted(kinds, key=lambda a: a.value))
            for name, kinds in acc.items()
        }

    def _register_proposal(self, name: str, payload: PromptProposalPayload) -> None:
        """Store a pending proposal, discarding any the session already had, and surface it to the developer."""
        self._discard_proposals(name)
        self.proposals[payload.proposal_id] = PendingProposal(name, payload)
        self.emit(
            ProposalArrived(name, payload.proposal_id, render_proposal(payload))
        )

    def _discard_proposals(self, name: str) -> None:
        """Drop any pending proposal a session left behind on settling or exit."""
        for pid in [
            pid for pid, p in self.proposals.items() if p.session_id == name
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
