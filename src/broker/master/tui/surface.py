"""FocusedSurface: a single-item focus view (escalation or proposal). It shows a
heading and a verbatim runtime-rendered body; the composer that drives it is the
master composer, so this widget never routes anything itself."""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static


class FocusedSurface(VerticalScroll):
    """Heading, rule and verbatim body for one escalation or proposal."""

    DEFAULT_CSS = """
    FocusedSurface { height: 1fr; }
    FocusedSurface .surface-heading { text-style: bold; color: $warning; }
    FocusedSurface .surface-rule { color: $warning; }
    FocusedSurface .surface-body { color: $text; }
    """

    def compose(self) -> ComposeResult:
        yield Static(Text(""), classes="surface-heading")
        yield Static(Text("─" * 40), classes="surface-rule")
        yield Static(Text(""), classes="surface-body")

    def show(self, heading: str, body: str) -> None:
        """Display ``body`` verbatim under ``heading``."""
        self.query(".surface-heading").first(Static).update(Text(heading))
        self.query(".surface-body").first(Static).update(Text(body))
