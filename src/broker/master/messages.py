"""Runtime → TUI messages: the runtime posts pre-rendered text.

Every string a developer reads is rendered by the runtime before it gets here;
widgets display these payloads verbatim and never see a raw protocol payload.
"""

from textual.message import Message


class EscalationArrived(Message):
    def __init__(
        self, session_id: str, escalation_id: str, rendered: str
    ) -> None:
        """Carry one runtime-rendered escalation to the TUI."""
        super().__init__()
        self.session_id = session_id
        self.escalation_id = escalation_id
        self.rendered = rendered  # runtime-rendered, verbatim — display as-is


class ProposalArrived(Message):
    def __init__(self, session_id: str, proposal_id: str, rendered: str) -> None:
        """Carry one runtime-rendered prompt proposal to the TUI."""
        super().__init__()
        self.session_id = session_id
        self.proposal_id = proposal_id
        self.rendered = rendered


class CompletionArrived(Message):
    def __init__(self, session_id: str, summary: str) -> None:
        """Carry a session's completion summary to the TUI, verbatim."""
        super().__init__()
        self.session_id = session_id
        self.summary = summary  # broker's summary, verbatim


class Notice(Message):
    def __init__(self, text: str) -> None:
        """Carry one line of runtime news to the TUI."""
        super().__init__()
        self.text = text


class SessionStatusChanged(Message):
    def __init__(self, session_id: str, state: str) -> None:
        """Carry a session's new state to the TUI."""
        super().__init__()
        self.session_id = session_id
        self.state = state


class LLMReply(Message):
    def __init__(self, text: str) -> None:
        """Carry the LLM layer's reply text to the TUI."""
        super().__init__()
        self.text = text
