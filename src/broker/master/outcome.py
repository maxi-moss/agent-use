"""Read-only completed/failed outcome, assembled from a session's decision log.

Renderer-neutral: the runtime builds a SessionOutcome from decision-log rows and
each frontend adapts it (mirrors broker.master.viewmodel). Everything here is a
broker report, not verified proof. The frontend formats timestamps."""

from dataclasses import dataclass, replace
from typing import Literal

from broker.decision_log import DecisionKind, DecisionRow
from broker.protocol.constants import SessionState

OUTCOME_STATES = frozenset(
    {SessionState.COMPLETED, SessionState.ERROR, SessionState.STOPPED}
)

_NO_FOLLOW_UP = "(no recorded follow-up)"

OutcomeStatus = Literal["completed", "error", "stopped", "other"]
EventKind = Literal["action", "escalation", "developer", "reactivated", "terminal"]


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    """One history line. `detail` is an escalation's Reason; `resolution` is an
    escalation's Solution or a developer prompt's result."""

    kind: EventKind
    ts: str
    label: str
    detail: str | None = None
    resolution: str | None = None


@dataclass(frozen=True, slots=True)
class SessionOutcome:
    """The whole read-only outcome for one settled session."""

    session_id: str
    title: str
    status: OutcomeStatus
    headline: str
    supporting: str
    history: tuple[OutcomeEvent, ...]


def build_outcome(
    *,
    session_id: str,
    title: str,
    state: SessionState,
    rows: list[DecisionRow],
) -> SessionOutcome:
    """Assemble a SessionOutcome from a session's decision-log rows.

    Args:
        session_id: Registry name of the session.
        title: The session's approval-time title, or "" when cleared.
        state: The session's current lifecycle state.
        rows: Decision-log rows, oldest first.

    Returns:
        The outcome the modal renders.
    """
    history: list[OutcomeEvent] = []
    # Events whose Solution/Result is the next kept action (answered/completed).
    # Sound for escalations because triage runs only while DRIVING, so nothing is
    # kept between a raise and its resolution; a developer prompt's result is the
    # next kept action by definition.
    pending: list[int] = []
    headline = ""
    supporting = ""
    fatal: DecisionRow | None = None

    def resolve(summary: str) -> None:
        for idx in pending:
            history[idx] = replace(history[idx], resolution=summary)
        pending.clear()

    for row in rows:
        if row.kind is DecisionKind.ANSWERED:
            if row.task_summary:
                resolve(row.task_summary)
                history.append(OutcomeEvent("action", row.ts, row.task_summary))
        elif row.kind is DecisionKind.COMPLETED:
            if row.task_summary:
                resolve(row.task_summary)
            headline = row.headline or headline
            supporting = row.supporting or supporting
            history.append(OutcomeEvent("terminal", row.ts, "Task completed"))
        elif row.kind is DecisionKind.ESCALATION_RAISED:
            pending.append(len(history))
            history.append(
                OutcomeEvent(
                    "escalation", row.ts, "Escalation", detail=row.task_summary or ""
                )
            )
        elif row.kind is DecisionKind.DEVELOPER_PROMPT:
            pending.append(len(history))
            history.append(OutcomeEvent("developer", row.ts, "Developer instruction"))
        elif row.kind is DecisionKind.REACTIVATED:
            history.append(OutcomeEvent("reactivated", row.ts, "New task"))
        elif row.kind is DecisionKind.ERROR:
            fatal = row  # last fatal wins; stale-dispatch noise is overwritten

    for idx in pending:
        history[idx] = replace(history[idx], resolution=_NO_FOLLOW_UP)

    status, headline, supporting = _status(state, fatal, headline, supporting)
    return SessionOutcome(
        session_id=session_id,
        title=title,
        status=status,
        headline=headline,
        supporting=supporting,
        history=tuple(history),
    )


def _status(
    state: SessionState,
    fatal: DecisionRow | None,
    headline: str,
    supporting: str,
) -> tuple[OutcomeStatus, str, str]:
    """Derive status + headline/supporting for the top of the modal."""
    if state == SessionState.COMPLETED:
        return "completed", headline, supporting
    if state == SessionState.ERROR:
        detail = (
            f"{fatal.reasoning}: {fatal.detail}"
            if fatal
            else "no error detail recorded"
        )
        return "error", "Session ended with an error", detail
    if state == SessionState.STOPPED:
        return "stopped", "Session ended without completing", ""
    return "other", headline or "In progress", supporting
