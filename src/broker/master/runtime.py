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
import logging
import sys
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from textual.message import Message

from broker.claude.settings import write_session_permissions
from broker.claude.trust import seed_trust
from broker.config import AdoptedSession, BrokerConfig, SessionBrokerConfig
from broker.paths import BrokerPaths
from broker.master import notifier
from broker.master.messages import (
    CompletionArrived,
    EscalationArrived,
    Notice,
    PermissionEscalationArrived,
    ProposalArrived,
    SessionStatusChanged,
)
from broker.master.registry import Registry, SessionRecord
from broker.protocol import client
from broker.protocol.constants import (
    NACK_MALFORMED,
    NACK_PROTOCOL_VIOLATION,
    NACK_SLOT_OCCUPIED,
    NACK_UNKNOWN_SESSION,
    SessionState,
    T_APPROVE_PROMPT,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_FATAL_ERROR,
    T_GET_DECISION_LOG,
    T_GET_PERMISSION_LOG,
    T_PERMISSION_ESCALATION,
    T_PROMPT_PROPOSAL,
    T_REACTIVATE,
    T_RETRACT,
    T_SEND_PROMPT,
    T_SHUTDOWN,
    T_STATUS,
)
from broker.protocol.schemas import (
    ApprovePromptPayload,
    BudgetUpdatePayload,
    CompletionPayload,
    DecisionLogPayload,
    DispatchDecisionPayload,
    Envelope,
    EscalationPayload,
    FatalErrorPayload,
    PermissionEscalationPayload,
    PermissionLogPayload,
    PermissionSuggestion,
    PromptProposalPayload,
    RaiserIdentity,
    ReactivatePayload,
    Response,
    RetractPayload,
    SendPromptPayload,
    StatusPayload,
)
from broker.protocol.server import serve_unix

logger = logging.getLogger(__name__)

# object, not None: App.post_message returns bool, and the bound method is
# passed here directly.
AppPost = Callable[[Message], object]

REQUEST_TIMEOUT_S = 10.0
STOP_WAIT_S = 10.0
SOCKET_POLL_S = 0.1
SOCKET_PROBE_TIMEOUT_S = 2.0

# Stands in for a pane the registry cannot name. A permission escalation is
# still worth surfacing without it: the developer knows the session.
PANE_UNKNOWN = "(pane unknown)"

# Whatever a session broker or its permission module escalated. Both carry the
# identity the slot rules read: escalation_id, session_id and raiser.
SlotPayload = EscalationPayload | PermissionEscalationPayload

# Failures the on-demand status probe absorbs into a warning line: a session
# that cannot be reached must not fail the whole listing.
PROBE_FAILURES = (OSError, ConnectionError, TimeoutError, ValidationError)


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
        record: Registry record of the session being reassigned.

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
            f"session {record.name} cannot be reassigned: the registry has no "
            + ", ".join(missing)
        )
    return AdoptedSession(
        pane_id=pane_id,
        claude_session_id=claude_session_id,
        transcript_path=transcript_path,
    )


class ProtocolViolation(Exception):
    """A raiser broke the one-outstanding-escalation invariant."""


class CapacityRefusal(Exception):
    """The slot is full, and a different raiser wanted it."""


class EscalationSlot:
    """Single active escalation (accept / retract / resolve / active)."""

    def __init__(self) -> None:
        """Start with no escalation active."""
        self._active: SlotPayload | None = None

    @property
    def active(self) -> SlotPayload | None:
        """Return the escalation awaiting the developer, or ``None``."""
        return self._active

    def accept(self, payload: SlotPayload) -> None:
        """Make ``payload`` the active escalation.

        Args:
            payload: The escalation to surface to the developer.

        Raises:
            ProtocolViolation: The raiser that holds the slot raised again —
                it is expected to wait for its own escalation to resolve.
            CapacityRefusal: Another raiser holds the slot. That is legitimate
                behaviour on the raiser's part, so it is refused rather than
                blamed.
        """
        if self._active is not None:
            held = (
                f"escalation {payload.escalation_id} arrived while "
                f"{self._active.escalation_id} is active"
            )
            if self._active.raiser == payload.raiser:
                raise ProtocolViolation(held)
            raise CapacityRefusal(held)
        self._active = payload

    def retract(self, escalation_id: str) -> SlotPayload | None:
        """Clear an escalation its session has withdrawn.

        Args:
            escalation_id: The escalation being withdrawn.

        Returns:
            The cleared payload, or ``None`` when it was not the active one.
        """
        return self._clear(escalation_id)

    def resolve(self, escalation_id: str) -> SlotPayload | None:
        """Clear an escalation the developer has decided.

        Args:
            escalation_id: The escalation that was answered.

        Returns:
            The cleared payload, or ``None`` when it was not the active one.
        """
        return self._clear(escalation_id)

    def _clear(self, escalation_id: str) -> SlotPayload | None:
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


@dataclass(frozen=True, slots=True)
class PendingProposal:
    """A prompt proposal awaiting the developer's approval."""

    session_name: str
    payload: PromptProposalPayload


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
        self.paths = BrokerPaths(cfg.broker_home)
        self.master_socket_path = self.paths.master_socket
        self.proposals: dict[str, PendingProposal] = {}
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
        if env.type == T_PERMISSION_ESCALATION:
            return await self._on_permission_escalation(env, name)
        if env.type == T_COMPLETION:
            p = CompletionPayload.model_validate(env.payload)
            self._set_state(name, SessionState.COMPLETED)
            self.app_post(CompletionArrived(name, p.summary))
            await self._notify(
                notifier.notify_done, f"Session {name} complete", p.summary
            )
            return self._ack(env, ok=True)
        if env.type == T_FATAL_ERROR:
            p = FatalErrorPayload.model_validate(env.payload)
            self._set_state(name, SessionState.ERROR)
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
            self.proposals[p.proposal_id] = PendingProposal(name, p)
            self._set_state(name, SessionState.AWAITING_APPROVAL)
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
            ``Response(ok=True)`` once live and announced, ``ok=False`` with a
            reason code if rejected.
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
            return self._nack(env, f"malformed escalation: {exc}", NACK_MALFORMED)
        unknown = self._reject_unknown_session(env, p.session_id, "escalation")
        if unknown is not None:
            return unknown
        refused = self._accept_into_slot(env, p, "escalation")
        if refused is not None:
            return refused
        self._set_state(p.session_id, SessionState.ESCALATED)
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

    async def _on_permission_escalation(
        self, env: Envelope, name: str
    ) -> Response:
        """Validate one permission escalation, make it active, and surface it.

        Args:
            env: Envelope carrying the permission-escalation payload.
            name: Session name from the envelope, used for rejection notices.

        Returns:
            ``Response(ok=True)`` once live and announced, ``ok=False`` with a
            reason code if rejected.
        """
        try:
            p = PermissionEscalationPayload.model_validate(env.payload)
        except ValidationError as exc:
            # A permission escalation missing the tool, the reason or the
            # session is not something the developer could act on.
            self.app_post(
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
        refused = self._accept_into_slot(env, p, "permission escalation")
        if refused is not None:
            return refused
        pane_id = self.registry.get(p.session_id).pane_id or PANE_UNKNOWN
        self.app_post(
            PermissionEscalationArrived(
                p.session_id,
                p.escalation_id,
                render_permission_escalation(p, pane_id),
            )
        )
        await self._notify(
            notifier.notify_request,
            f"Permission prompt in session {p.session_id}",
            f"{p.tool_name} — answer it in pane {pane_id}",
        )
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
        self.app_post(Notice(msg))
        return self._nack(env, msg, NACK_UNKNOWN_SESSION)

    def _accept_into_slot(
        self, env: Envelope, payload: SlotPayload, kind: str
    ) -> Response | None:
        """Make ``payload`` active, or build the NACK refusing it.

        Args:
            env: Envelope being answered.
            payload: The escalation offered to the slot.
            kind: Word naming the escalation kind, used in the notice.

        Returns:
            ``None`` once accepted, otherwise the NACK.
        """
        try:
            self.slot.accept(payload)
        except ProtocolViolation as exc:
            self.app_post(Notice(f"PROTOCOL VIOLATION: {exc}"))
            return self._nack(env, str(exc), NACK_PROTOCOL_VIOLATION)
        except CapacityRefusal as exc:
            self.app_post(Notice(f"{kind} refused, slot occupied: {exc}"))
            return self._nack(env, str(exc), NACK_SLOT_OCCUPIED)
        return None

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
        self.app_post(SessionStatusChanged(name, record.state))
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
        record.budget_count = 0
        proc = await self._spawn_broker(record, adopt=adopt)
        record.pid = proc.pid
        self._procs[session_id] = proc
        self.registry.upsert(record)
        self._set_state(session_id, SessionState.SPAWNING)
        return (
            f"session {session_id} reassigned to a new broker (pid {proc.pid})"
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
            self.app_post(Notice(rejected))
            return rejected
        record.intent = intent
        record.approved_prompt = None  # superseded; set again on approval
        record.budget_count = 0
        self.registry.upsert(record)
        self._set_state(session_id, SessionState.GROUNDING)
        return f"session {session_id} reactivated — grounding the new task"

    async def approve_prompt(self, proposal_id: str, prompt: str) -> str:
        """Approve a pending prompt proposal and send it to its session.

        Args:
            proposal_id: Identifier of the proposal being answered.
            prompt: Prompt text to send — the developer's edit of the
                proposal, or the proposal verbatim.

        Returns:
            An outcome line: approved, unknown proposal, or rejected as stale.
        """
        pending = self.proposals.get(proposal_id)
        if pending is None:
            msg = f"unknown proposal {proposal_id!r} — nothing approved"
            self.app_post(Notice(msg))
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
            self.app_post(Notice(rejected))
            return rejected
        del self.proposals[proposal_id]
        record.approved_prompt = prompt  # the AUTHORITATIVE intent
        self.registry.upsert(record)
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
        active = self.slot.active
        if active is None or active.escalation_id != escalation_id:
            msg = (
                f"decision NOT dispatched — escalation {escalation_id} is "
                "no longer live"
            )
            self.app_post(Notice(msg))
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
            self.app_post(Notice(msg))
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
            self.app_post(Notice(rejected))
            return rejected
        self.slot.resolve(escalation_id)
        self._set_state(record.name, SessionState.DRIVING)
        return f"decision dispatched to session {record.name}"

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
            self.app_post(Notice(rejected))
            return rejected
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
        self._retract_stranded_permission_escalation(session_id)
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
        for name in sorted(self.registry.records):
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
        for name in sorted(self.registry.records):
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
        self, record: SessionRecord, *, adopt: AdoptedSession | None
    ) -> asyncio.subprocess.Process:
        """Start a session-broker subprocess for ``record``.

        Args:
            record: Supplies the identity, socket, cwd, intent and budget the
                broker starts from.
            adopt: Pane, Claude session and transcript of a running session the
                broker takes over; ``None`` starts a fresh one.

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
            watchdog_seconds=self.cfg.watchdog_seconds,
            budget_max=self.cfg.budget_max,
            claude_settings_path=str(settings_path),
            adopt=adopt,
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

    def _retract_stranded_permission_escalation(self, session_id: str) -> None:
        """Clear a permission escalation whose broker is gone.

        A permission escalation is withdrawn by the module that raised it once
        the native prompt is answered. With its broker stopped nothing is left
        to withdraw it, and it would hold the slot indefinitely.

        Args:
            session_id: Session whose broker was stopped or replaced.
        """
        active = self.slot.active
        if active is None:
            return
        raiser = RaiserIdentity(component="permission", session_id=session_id)
        if active.raiser != raiser:
            return
        self.slot.retract(active.escalation_id)
        self.app_post(
            Notice(
                f"permission escalation {active.escalation_id} from session "
                f"{session_id} retracted: its broker was stopped"
            )
        )

    def _set_state(self, name: str, state: SessionState) -> None:
        """Record a session's new state and tell the TUI.

        Args:
            name: Registry name of the session.
            state: New state to persist.
        """
        try:
            record = self.registry.get(name)
        except KeyError:
            self.app_post(Notice(f"message from unknown session {name!r}"))
            return
        logger.info("session %s: %s -> %s", name, record.state, state)
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
