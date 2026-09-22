"""OutcomeModal: read-only completed/failed outcome over the dimmed master view.

The Textual adapter for broker.master.outcome.SessionOutcome (mirrors the
viewmodel -> fleet split). Opened by /outcome sN; Close or Escape dismisses.
First and only ModalScreen in the app; it never routes through the master LLM."""

from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from broker.master.outcome import OutcomeEvent, SessionOutcome

_SUBTITLE: dict[str, str] = {
    "completed": "Completed · session broker report",
    "error": "Ended with an error · session broker report",
    "stopped": "Stopped · session broker report",
    "other": "In progress · session broker report",
}


def _fmt_time(ts: str) -> str:
    """Local wall-clock HH:MM for a stored ISO-UTC timestamp; '' if unparseable."""
    if not ts:
        return ""
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%H:%M")
    except ValueError:
        return ""


def _render_event(ev: OutcomeEvent) -> Text:
    """One history line as rich.Text: marker, label, time, Reason/Solution."""
    if ev.kind == "reactivated":
        return Text(f"— {ev.label} —", style="dim")
    line = Text()
    line.append("✓ ", style="green")
    line.append(ev.label)
    time = _fmt_time(ev.ts)
    if time:
        line.append(f"   {time}", style="dim")
    if ev.detail is not None:
        line.append(f"\n    Reason: {ev.detail}", style="dim")
    if ev.resolution is not None:
        key = "Solution" if ev.kind == "escalation" else "Result"
        line.append(f"\n    {key}: {ev.resolution}", style="dim")
    return line


class OutcomeModal(ModalScreen[None]):
    """A centered read-only panel showing one session's outcome."""

    DEFAULT_CSS = """
    OutcomeModal { align: center middle; background: $background 60%; }
    OutcomeModal #panel {
        width: 80%; max-width: 96; height: auto; max-height: 90%;
        border: round $primary; background: $surface; padding: 1 2;
    }
    OutcomeModal #subtitle { color: $text-muted; }
    OutcomeModal #headline { text-style: bold; margin: 1 0 0 0; }
    OutcomeModal #supporting { color: $text-muted; margin: 0 0 1 0; }
    OutcomeModal .hist-title { text-style: bold; color: $text-muted; margin: 1 0 0 0; }
    OutcomeModal #history { height: auto; max-height: 18; }
    OutcomeModal #close { margin: 1 0 0 0; }
    """

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, outcome: SessionOutcome) -> None:
        """Render ``outcome`` in a read-only modal."""
        super().__init__()
        self._outcome = outcome

    def compose(self) -> ComposeResult:
        o = self._outcome
        with Vertical(id="panel"):
            yield Static(Text(f"{o.title or o.session_id} · {o.session_id}"))
            yield Static(Text(_SUBTITLE[o.status]), id="subtitle")
            yield Static(Text(o.headline), id="headline")
            yield Static(Text(o.supporting), id="supporting")
            yield Static(Text("HISTORY"), classes="hist-title")
            with VerticalScroll(id="history"):
                for ev in o.history:
                    yield Static(_render_event(ev))
            yield Button("Close", id="close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()

    def action_close(self) -> None:
        self.dismiss()
