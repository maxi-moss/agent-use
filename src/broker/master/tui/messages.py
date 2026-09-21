"""Textual-only messages for the master frontend adapter. The view model
(broker.master.viewmodel) never imports this; the app wraps each ViewEvent in
ViewEventMessage to pump it through Textual's message loop."""

from textual.message import Message

from broker.master.viewmodel import ViewEvent


class ViewEventMessage(Message):
    def __init__(self, event: ViewEvent) -> None:
        """Carry one renderer-neutral view event through the Textual loop."""
        super().__init__()
        self.event = event


class LLMReply(Message):
    def __init__(self, text: str) -> None:
        """Carry the master LLM's reply text to the app (app-internal)."""
        super().__init__()
        self.text = text
