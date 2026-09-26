"""Dashboard-only per-session state and the fleet view assembled from it.

Nothing here is persisted: pushed live status, pending prompt proposals and
the master's own activity live in memory and are rebuilt from the brokers
after a master restart.
"""

from dataclasses import dataclass

from broker.master.pane_escalations import PaneEscalations
from broker.master.payload_render import pane_label, render_proposal
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.master.viewmodel import (
    Attention,
    EventSink,
    FleetUpdated,
    FleetView,
    HeadRequest,
    PaneRequest,
    ProposalArrived,
    SessionRow,
)
from broker.protocol.constants import (
    ABSORBING_STATES,
    SETTLED_STATES,
    PaneKind,
    SessionState,
)
from broker.protocol.schemas import (
    LiveStatusPayload,
    PromptProposalPayload,
    StatusPayload,
)


@dataclass(frozen=True, slots=True)
class PendingProposal:
    """A prompt proposal awaiting the developer's approval."""

    session_id: str
    payload: PromptProposalPayload


@dataclass(slots=True)
class _SessionLiveStatus:
    """What a session's broker last reported it is doing right now."""

    activity: str = ""
    task_activity: str = ""
    permission_prompt: bool = False
    pushes: int = 0


class FleetBoard:
    """Owns the per-session dashboard state and publishes the fleet view."""

    def __init__(
        self,
        registry: Registry,
        queue: EscalationQueue,
        panes: PaneEscalations,
        budget_max: int,
        emit: EventSink,
    ) -> None:
        """Wire the board to the stores its view reads and to the frontend sink.

        Args:
            registry: Session registry the rows are built from.
            queue: Decision-escalation queue behind the ESCALATION badges.
            panes: Open pane escalations behind the PERMISSION and QUESTION
                badges.
            budget_max: Autonomous-answer budget shown on every row.
            emit: Receives every view event the board produces.
        """
        self.registry = registry
        self.queue = queue
        self.panes = panes
        self.budget_max = budget_max
        self.emit = emit
        self.proposals: dict[str, PendingProposal] = {}
        self._live: dict[str, _SessionLiveStatus] = {}
        self._master_activity: str | None = None

    def apply_live_status(self, session_id: str, status: LiveStatusPayload) -> bool:
        """Fold a session's pushed live status into the board.

        Args:
            session_id: Registry name of the session that pushed.
            status: The broker's current live status.

        Returns:
            ``True`` when the status applies and its state is the caller's to
            record, ``False`` when the session is absorbed and the push is
            ignored.
        """
        record = self.registry.records[session_id]
        # Outside the absorbing guard: a broker that still answers knows its
        # own pane and Claude session.
        self._note_identity(record, status)
        # Left only by a master-initiated boundary write, never by a push.
        if record.state in ABSORBING_STATES:
            return False
        live = self._live.setdefault(session_id, _SessionLiveStatus())
        live.pushes += 1
        live.activity = status.activity
        live.permission_prompt = status.permission_prompt
        self.note_task_activity(session_id, status.state, status.task_activity)
        return True

    def apply_probed_status(self, session_id: str, status: StatusPayload) -> bool:
        """Fold a session's probed status into the board.

        Args:
            session_id: Registry name of the probed session.
            status: The broker's reply to the probe.

        Returns:
            ``True`` when the status applies and its state is the caller's to
            record, ``False`` when the session is absorbed and the reply is
            ignored.
        """
        if self.registry.get(session_id).state in ABSORBING_STATES:
            return False
        live = self._live.setdefault(session_id, _SessionLiveStatus())
        live.permission_prompt = status.permission_prompt
        self.note_task_activity(session_id, status.state, status.task_activity)
        return True

    def push_count(self, session_id: str) -> int:
        """Return how many pushes from a session the board has applied."""
        live = self._live.get(session_id)
        return live.pushes if live is not None else 0

    def permission_prompt(self, session_id: str) -> bool:
        """Return whether a session last reported a native permission prompt open."""
        live = self._live.get(session_id)
        return live is not None and live.permission_prompt

    def note_task_activity(
        self, session_id: str, state: SessionState, text: str
    ) -> None:
        """Show a session's task activity; one reporting a settled state shows none."""
        live = self._live.setdefault(session_id, _SessionLiveStatus())
        live.task_activity = (
            "" if state in SETTLED_STATES | ABSORBING_STATES else text
        )

    def settle(self, session_id: str) -> None:
        """Clear the live status and pending proposals of a session that settled."""
        live = self._live.get(session_id)
        if live is not None:
            self._live[session_id] = _SessionLiveStatus(pushes=live.pushes)
        self.discard_proposals(session_id)

    def forget(self, session_id: str) -> None:
        """Drop everything the board holds for a session that left the fleet."""
        self._live.pop(session_id, None)
        self.discard_proposals(session_id)

    def register_proposal(
        self, session_id: str, payload: PromptProposalPayload
    ) -> None:
        """Hold a session's proposal in place of any earlier one and announce it."""
        self.discard_proposals(session_id)
        self.proposals[payload.proposal_id] = PendingProposal(session_id, payload)
        self.emit(ProposalArrived(session_id, payload.proposal_id))

    def discard_proposals(self, session_id: str) -> None:
        """Drop every pending proposal a session holds."""
        for proposal_id in [
            proposal_id
            for proposal_id, pending in self.proposals.items()
            if pending.session_id == session_id
        ]:
            del self.proposals[proposal_id]

    def pending_proposals(self) -> list[PendingProposal]:
        """Return every proposal awaiting approval, oldest first."""
        return list(self.proposals.values())

    def rendered_proposals(self) -> dict[str, str]:
        """Return each proposing session's pending proposal, rendered verbatim."""
        return {
            pending.session_id: render_proposal(pending.payload)
            for pending in self.proposals.values()
        }

    def note_master_activity(self, text: str) -> None:
        """Show what the master itself is doing on the dashboard header."""
        self._master_activity = text
        self.publish()

    def clear_master_activity(self) -> None:
        """Return the dashboard header to idle."""
        self._master_activity = None
        self.publish()

    def build_view(self) -> FleetView:
        """Assemble the structured sidebar view from the board and its stores."""
        badges = self._badges_by_session()
        rows: list[SessionRow] = []
        for name in self.registry.names_in_order():
            record = self.registry.records[name]
            live = self._live.get(name, _SessionLiveStatus())
            rows.append(
                SessionRow(
                    session_id=name,
                    state=record.state,
                    title=record.title,
                    task_activity=live.task_activity,
                    broker_activity=live.activity,
                    budget_count=record.budget_count,
                    budget_max=self.budget_max,
                    badges=badges.get(name, ()),
                    pane_id=record.pane_id,
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
                for p in self.panes.in_session_order()
            ),
        )

    def publish(self) -> None:
        """Emit the current structured sidebar snapshot."""
        self.emit(FleetUpdated(self.build_view()))

    def _head_request(self) -> HeadRequest | None:
        head = self.queue.active
        if head is None:
            return None
        return HeadRequest(
            head.session_id, head.escalation_id, head.disclosure.what_was_asked
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
        for name, live in self._live.items():
            if live.permission_prompt:
                acc.setdefault(name, set()).add(Attention.PERMISSION)
        return {
            name: tuple(sorted(kinds, key=lambda a: a.value))
            for name, kinds in acc.items()
        }

    def _note_identity(
        self, record: SessionRecord, status: LiveStatusPayload
    ) -> None:
        """Persist the pane, Claude session and transcript a broker reports."""
        pane_id = status.pane_id or record.pane_id
        claude_session_id = status.claude_session_id or record.claude_session_id
        transcript_path = status.transcript_path or record.transcript_path
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
