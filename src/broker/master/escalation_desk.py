"""Every escalation waiting on the developer, from arrival to answer.

Decision escalations queue for the developer one at a time; pane escalations
are held per session and answered in the pane itself. The desk accepts,
announces, dispatches, clarifies, resolves and retracts both, and renders
what is waiting. It never publishes the fleet view: the runtime does that
once per routed message or runtime tool call. The desk's own dispatch and
clarify tools change nothing the fleet view shows, so they publish nothing.
Its developer-requested actions return a ``ControlResult``, an expected
refusal included.
"""

from broker.master import notifier
from broker.master.broker_link import BrokerLink
from broker.master.control_result import ControlResult, refusal
from broker.master.pane_escalations import PaneEscalations
from broker.master.payload_render import (
    PANE_UNKNOWN,
    pane_label,
    render_escalation,
    render_pane_escalation,
)
from broker.master.queue import EscalationProtocolViolation, EscalationQueue
from broker.master.registry import Registry
from broker.master.viewmodel import (
    EscalationArrived,
    EventSink,
    Notice,
    PaneEscalationArrived,
)
from broker.protocol.constants import NackCode, PaneKind
from broker.protocol.schemas import (
    ClarifyEscalationReplyPayload,
    ClarifyEscalationRequestPayload,
    DecisionDeliveredPayload,
    DecisionUndeliveredPayload,
    DispatchDecisionPayload,
    EscalationPayload,
    EscalationRetractPayload,
    NackPayload,
    PaneEscalationPayload,
    PaneRetractPayload,
)

# The broker runs an LLM call before it can reply, so this is far longer than
# REQUEST_TIMEOUT_S and must exceed the broker's own CLARIFY_TIMEOUT_S.
CLARIFY_ESCALATION_TIMEOUT_S = 60.0

# Opens every dispatch refusal decided before anything was sent to the session.
NOT_DISPATCHED = "decision NOT dispatched"


class EscalationDesk:
    """Owns the lifecycle of every decision and pane escalation."""

    def __init__(
        self,
        queue: EscalationQueue,
        panes: PaneEscalations,
        registry: Registry,
        link: BrokerLink,
        emit: EventSink,
    ) -> None:
        """Wire the desk to the escalation stores, the registry and the brokers.

        Args:
            queue: Decision-escalation queue.
            panes: Open pane escalations.
            registry: Session registry, for each session's pane and socket.
            link: Socket link to the session brokers.
            emit: Receives every view event the desk produces.
        """
        self.queue = queue
        self.panes = panes
        self.registry = registry
        self.link = link
        self.emit = emit

    # ── arrival and withdrawal ───────────────────────────────────────────────

    async def accept(
        self, session_id: str, p: EscalationPayload
    ) -> NackPayload | None:
        """Queue one escalation and surface it if it is next.

        Args:
            session_id: Session that raised the escalation.
            p: The escalation.

        Returns:
            ``None`` once queued, otherwise the refusal.
        """
        try:
            self.queue.accept(p)
        except EscalationProtocolViolation as exc:
            self.emit(Notice(f"PROTOCOL VIOLATION: {exc}"))
            return NackPayload(
                error=str(exc), reason_code=NackCode.PROTOCOL_VIOLATION
            )
        await self.surface_head()
        return None

    async def accept_pane(
        self, session_id: str, p: PaneEscalationPayload
    ) -> NackPayload | None:
        """Hold one pane escalation, superseding its predecessor, and announce it.

        Args:
            session_id: Session whose pane shows the native prompt.
            p: The pane escalation.

        Returns:
            ``None``; a pane escalation is never refused.
        """
        superseded = self.panes.accept(p)
        if superseded is not None:
            self.emit(
                Notice(
                    f"{p.kind} escalation {superseded.escalation_id} from "
                    f"session {session_id} superseded by {p.escalation_id}"
                )
            )
        await self.announce_pane(p)
        return None

    async def retract(
        self, session_id: str, p: EscalationRetractPayload
    ) -> NackPayload | None:
        """Clear a decision escalation its session resolved out of band.

        Args:
            session_id: Session that withdrew the escalation.
            p: The escalation withdrawn and why.

        Returns:
            ``None``; a retract is never refused.
        """
        cleared = self.queue.retract(p.escalation_id)
        if cleared is not None and cleared.was_surfaced:
            # It was surfaced, so the developer must learn it is no longer
            # live; a waiting entry they never saw retracts silently.
            self.emit(
                Notice(
                    f"escalation {p.escalation_id} from session {session_id} "
                    f"retracted: {p.reason}"
                )
            )
        await self.surface_head()
        return None

    async def retract_pane(
        self, session_id: str, p: PaneRetractPayload
    ) -> NackPayload | None:
        """Clear a pane escalation whose native prompt is no longer open.

        Args:
            session_id: Session whose prompt closed.
            p: The pane escalation withdrawn and why.

        Returns:
            ``None``; a retract is never refused.
        """
        cleared = self.panes.retract(p.escalation_id)
        if cleared is not None:
            self.emit(
                Notice(
                    f"{cleared.kind} escalation {p.escalation_id} from session "
                    f"{session_id} retracted: {p.reason}"
                )
            )
        return None

    async def retract_stranded(self, session_id: str) -> None:
        """Clear a queued decision escalation whose broker is gone.

        Args:
            session_id: Session whose broker is gone.
        """
        cleared = self.queue.retract_for_session(session_id)
        if cleared is None:
            return
        self.emit(
            Notice(
                f"escalation {cleared.payload.escalation_id} from session "
                f"{session_id} retracted: its broker is gone"
            )
        )
        await self.surface_head()

    def retract_stranded_panes(self, session_id: str) -> None:
        """Clear the open pane escalations of a session whose broker is gone.

        Args:
            session_id: Session whose broker is gone.
        """
        for prompt in self.panes.retract_for_session(session_id):
            self.emit(
                Notice(
                    f"{prompt.kind} escalation {prompt.escalation_id} from "
                    f"session {session_id} retracted: its broker is gone"
                )
            )

    def session_stopped(self, session_id: str) -> None:
        """Release a stopped session's escalations, keeping its queued decision.

        Args:
            session_id: Session whose broker was stopped.
        """
        self.retract_stranded_panes(session_id)
        # A hard stop mid-delivery leaves no delivered/undelivered reply to
        # clear the in-flight marker. Drop it for this session's own entry
        # (the escalation itself is intentionally kept) so a re-dispatch is
        # not refused as still being delivered.
        self.queue.clear_inflight_for_session(session_id)

    # ── announcing ───────────────────────────────────────────────────────────

    async def surface_head(self) -> None:
        """Render, announce and notify the queue's head, exactly once."""
        head = self.queue.take_unsurfaced_head()
        if head is None:
            return
        self.emit(
            EscalationArrived(
                head.session_id,
                head.escalation_id,
                head.disclosure.escalation_title,
                render_escalation(head),
            )
        )
        await notifier.notify_or_notice(
            self.emit,
            notifier.notify_request,
            f"Escalation from session {head.session_id}",
            head.disclosure.what_was_asked,
        )

    async def announce_pane(self, p: PaneEscalationPayload) -> None:
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
        await notifier.notify_or_notice(
            self.emit,
            notifier.notify_request,
            title,
            f"{pane_label(p)} — answer it in pane {pane_id}",
        )

    async def announce_open_panes(self) -> None:
        """Announce every open pane escalation, in session order."""
        for prompt in self.panes.in_session_order():
            await self.announce_pane(prompt)

    # ── answering ────────────────────────────────────────────────────────────

    async def dispatch(self, escalation_id: str, decision: str) -> ControlResult:
        """Dispatch a decision to the session whose escalation it answers.

        Args:
            escalation_id: The escalation the decision answers.
            decision: The developer's decision, sent verbatim.

        Returns:
            Dispatched to the named session, or refused: the escalation is no
            longer live, it is a pane escalation (naming the pane), a decision
            is already in flight for it, or the session NACKed or never
            replied.
        """
        refused = self._refuse_if_pane(escalation_id, NOT_DISPATCHED)
        if refused is not None:
            return refused
        # Liveness is checked THE INSTANT before the write, not at
        # surface time. A stale dispatch is the worst failure this system
        # can produce.
        active = self.queue.active
        if active is None or active.escalation_id != escalation_id:
            return refusal(
                self.emit,
                f"{NOT_DISPATCHED} — escalation {escalation_id} is "
                "no longer live",
            )
        inflight = self.queue.inflight
        if inflight is not None:
            # A decision is already on its way to the pane; a second would
            # double-submit the same escalation.
            return refusal(
                self.emit,
                f"{NOT_DISPATCHED} — a decision for escalation "
                f"{inflight} is already being delivered",
            )
        record = self.registry.get(active.session_id)
        # The ACK only confirms the broker accepted the decision; resolution
        # waits for T_DECISION_DELIVERED, and until then no second decision
        # may be dispatched. The marker is set before the send because that
        # reply can be handled before the ACK returns.
        self.queue.mark_inflight(escalation_id)
        try:
            rejected = await self.link.deliver(
                record,
                DispatchDecisionPayload(
                    escalation_id=escalation_id, response=decision
                ),
                rejection=(
                    f"session {record.name} rejected the dispatched decision "
                    f"for escalation {escalation_id} (stale)"
                ),
            )
        except BaseException:
            self.queue.clear_inflight(escalation_id)
            raise
        if rejected is not None:
            self.queue.clear_inflight(escalation_id)
            return refusal(self.emit, rejected)
        return ControlResult(True, f"decision dispatched to session {record.name}")

    async def delivered(
        self, session_id: str, p: DecisionDeliveredPayload
    ) -> NackPayload | None:
        """Resolve an escalation now that its decision reached the pane.

        Args:
            session_id: Session that confirmed delivery.
            p: The escalation whose decision landed.

        Returns:
            ``None``; a delivery report is never refused.
        """
        if self.queue.resolve(p.escalation_id) is not None:
            # It was the live head: surface whatever is next. DRIVING is the
            # broker's transition to report; the master never invents it.
            await self.surface_head()
        return None

    async def undelivered(
        self, session_id: str, p: DecisionUndeliveredPayload
    ) -> NackPayload | None:
        """Handle a dispatched decision that did not reach the pane.

        Resolution waits for confirmed delivery, so the escalation was never
        resolved: a still-live miss (a failed pane write) stays surfaced for a
        re-decide, while a stale dispatch (the broker moved past it) drops the
        now orphaned queue entry.

        Args:
            session_id: Session that reported the miss.
            p: The escalation the decision answered, why it did not land, and
                whether the broker still holds it live.

        Returns:
            ``None``; a miss report is never refused.
        """
        if p.still_live:
            self.queue.clear_inflight(p.escalation_id)
            self.emit(
                Notice(
                    f"decision for escalation {p.escalation_id} did NOT reach "
                    f"session {session_id}: {p.detail}"
                )
            )
            return None
        cleared = self.queue.retract(p.escalation_id)
        if cleared is not None and cleared.was_surfaced:
            self.emit(
                Notice(
                    f"escalation {p.escalation_id} from session {session_id} "
                    f"cleared: {p.detail}"
                )
            )
        await self.surface_head()
        return None

    async def clarify(self, escalation_id: str, question: str) -> ControlResult:
        """Relay a read-only question about the live escalation to its broker.

        The escalation stays pending. The broker's answer is shown to the
        developer verbatim (a Notice); this returns only an acknowledgement to
        the tool loop, so the master never rewrites developer-facing text.

        Args:
            escalation_id: The escalation the question is about.
            question: The developer's question, sent verbatim.

        Returns:
            An acknowledgement, or the refusal naming why no answer was
            obtained: not the head, a pane escalation, or the broker declined
            or never replied.
        """
        refused = self._refuse_if_pane(escalation_id, "question NOT sent")
        if refused is not None:
            return refused
        active = self.queue.active
        if active is None or active.escalation_id != escalation_id:
            return refusal(
                self.emit,
                f"question NOT sent — escalation {escalation_id} is no longer live",
            )
        record = self.registry.get(active.session_id)
        reply = await self.link.query(
            record,
            ClarifyEscalationRequestPayload(
                escalation_id=escalation_id, question=question
            ),
            timeout_s=CLARIFY_ESCALATION_TIMEOUT_S,
            rejection=(
                f"no clarification from session {record.name} for escalation "
                f"{escalation_id}"
            ),
        )
        if isinstance(reply, str):
            return refusal(self.emit, reply)
        answer = ClarifyEscalationReplyPayload.model_validate(reply.payload).answer
        self.emit(
            Notice(f"Session {record.name} on escalation {escalation_id}:\n\n{answer}")
        )
        return ControlResult(
            True, f"clarification from session {record.name} shown to the developer"
        )

    def _refuse_if_pane(
        self, escalation_id: str, lead: str
    ) -> ControlResult | None:
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
        return refusal(
            self.emit,
            f"{lead} — escalation {escalation_id} is {what} in session "
            f"{prompt.session_id}. The developer answers it in pane "
            f"{self.pane_of(prompt.session_id)}.",
        )

    # ── read model ───────────────────────────────────────────────────────────

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

    def rendered_head(self) -> str | None:
        """Return the live head escalation rendered verbatim, or ``None``."""
        head = self.queue.active
        return None if head is None else render_escalation(head)

    def rendered_panes(self, kind: PaneKind) -> dict[str, str]:
        """Return each session's open ``kind`` escalation, rendered verbatim."""
        return {
            p.session_id: render_pane_escalation(p, self.pane_of(p.session_id))
            for p in self.panes.in_session_order()
            if p.kind == kind
        }
