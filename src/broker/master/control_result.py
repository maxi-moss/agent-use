"""What a developer-requested session-control action reports back.

An expected refusal is a result with ``ok`` false, never an exception; a raised
exception from a session-control action means a bug.
"""

from dataclasses import dataclass

from broker.master.viewmodel import EventSink, Notice


@dataclass(frozen=True, slots=True)
class ControlResult:
    """The outcome of one session-control action, as the master LLM reads it."""

    ok: bool
    text: str


def refusal(emit: EventSink, text: str) -> ControlResult:
    """Show a refusal to the developer and return it as a failed result.

    Args:
        emit: Receives the Notice carrying ``text``.
        text: Why the action was refused.

    Returns:
        The failed result carrying ``text``.
    """
    emit(Notice(text))
    return ControlResult(False, text)
