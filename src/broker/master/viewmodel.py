"""Renderer-neutral view model: the master runtime produces these plain data
types; each frontend adapts them to its own widgets. No Textual (or any
frontend) import may enter this module — that is what keeps it reusable.

State and attention are structured here; all developer-facing PROSE stays a
runtime-rendered string carried verbatim on the event that delivers it."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from broker.protocol.constants import PaneKind, SessionState

# Lifecycle states a session has settled into: the ones with an outcome to view.
_SETTLED_STATES = frozenset(
    {SessionState.COMPLETED, SessionState.ERROR, SessionState.STOPPED}
)


class Attention(StrEnum):
    """A kind of thing a session needs the developer for. Distinct badges, no
    aggregate. A row may carry more than one."""

    ESCALATION = "escalation"
    PROPOSAL = "proposal"
    PERMISSION = "permission"
    QUESTION = "question"


@dataclass(frozen=True, slots=True)
class SessionRow:
    """One session's sidebar row: lifecycle, activity, budget and badges. No
    prose — disclosure and proposed prompt travel as their own strings."""

    session_id: str
    state: SessionState
    title: str
    task_activity: str
    broker_activity: str
    budget_count: int
    budget_max: int
    badges: tuple[Attention, ...]
    pane_id: str | None

    @property
    def is_settled(self) -> bool:
        """True when the session has reached a state with an outcome to view."""
        return self.state in _SETTLED_STATES


@dataclass(frozen=True, slots=True)
class HeadRequest:
    """The decision escalation the developer is asked about right now. ``text``
    is its question verbatim."""

    session_id: str
    escalation_id: str
    text: str


@dataclass(frozen=True, slots=True)
class PaneRequest:
    """One open native prompt, answered in its session's pane. ``label`` is the
    tool name of a permission prompt, or the first question of a menu."""

    kind: PaneKind
    session_id: str
    escalation_id: str
    label: str


@dataclass(frozen=True, slots=True)
class FleetView:
    """The whole sidebar in one value: master activity, one row per session
    (numeric id order), the decision-escalation queue summary, and every open
    pane escalation (numeric session order, then kind)."""

    master_activity: str | None
    rows: tuple[SessionRow, ...]
    queue_depth: int
    waiting: tuple[str, ...]
    head: HeadRequest | None
    panes: tuple[PaneRequest, ...]


@dataclass(frozen=True, slots=True)
class FleetUpdated:
    view: FleetView


@dataclass(frozen=True, slots=True)
class EscalationArrived:
    session_id: str
    escalation_id: str
    escalation_title: str
    rendered: str


@dataclass(frozen=True, slots=True)
class PaneEscalationArrived:
    kind: PaneKind
    session_id: str
    escalation_id: str
    rendered: str


@dataclass(frozen=True, slots=True)
class ProposalArrived:
    session_id: str
    proposal_id: str
    rendered: str


@dataclass(frozen=True, slots=True)
class CompletionArrived:
    session_id: str
    headline: str
    supporting: str


@dataclass(frozen=True, slots=True)
class Notice:
    text: str


@dataclass(frozen=True, slots=True)
class SessionStatusChanged:
    session_id: str
    state: SessionState


ViewEvent = (
    FleetUpdated
    | EscalationArrived
    | PaneEscalationArrived
    | ProposalArrived
    | CompletionArrived
    | Notice
    | SessionStatusChanged
)

EventSink = Callable[[ViewEvent], None]
