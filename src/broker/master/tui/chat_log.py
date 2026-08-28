"""Chat-log widgets for the master TUI.

ChatMessage renders one entry as a role-styled accent bar and label above the
body; the body is a ``rich.Text`` so its content is never markup-parsed.
ThinkingIndicator is the transient italic line shown while the master's LLM
turn runs. Appearance rides with each widget as DEFAULT_CSS; app.py keeps only
layout.
"""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class ChatMessage(Vertical):
    """One chat entry: an optional role label above a verbatim body."""

    DEFAULT_CSS = """
    ChatMessage {
        height: auto;
        width: 1fr;
        margin: 0 0 1 0;
        padding: 0 0 0 1;
        border-left: thick $foreground 30%;
    }
    ChatMessage .role { height: 1; text-style: bold; color: $text-muted; }
    ChatMessage .body { color: $text; }
    ChatMessage.user { border-left: thick $success; }
    ChatMessage.user .role { color: $success; }
    ChatMessage.master { border-left: thick $primary; }
    ChatMessage.master .role { color: $primary; }
    """

    def __init__(self, role: str, body: str, label: str | None) -> None:
        """Build a chat entry styled for ``role`` (user, master, or system)."""
        super().__init__(classes=role)
        self._label = label
        self._body = body

    def compose(self) -> ComposeResult:
        if self._label is not None:
            yield Static(Text(self._label), classes="role")
        yield Static(Text(self._body), classes="body")


class ThinkingIndicator(Static):
    """Transient italic line shown while the master's LLM turn runs."""

    DEFAULT_CSS = """
    ThinkingIndicator {
        height: auto;
        margin: 0 0 1 2;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self) -> None:
        """Render the fixed italic thinking line."""
        super().__init__(Text("⋯ Thinking…"))
