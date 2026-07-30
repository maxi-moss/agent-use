"""PromptArea — a multi-line developer input.

Enter submits (matching the single-line Input it replaces); ctrl+j inserts a
newline for manual composition. Pasted text lands verbatim, newlines and all
— TextArea's own paste handling, unlike Input's, does not truncate to the
first line.
"""

from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.widgets import TextArea


class PromptArea(TextArea):
    """Multi-line prompt box that submits on Enter."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            """Carry the submitted prompt text to the app."""
            super().__init__()
            self.text = text

    BINDINGS = [
        Binding("ctrl+j", "insert_newline", "Newline", show=False),
    ]

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        await super()._on_key(event)

    def action_insert_newline(self) -> None:
        self._replace_via_keyboard("\n", *self.selection)
