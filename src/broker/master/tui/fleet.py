"""FleetSidebar: the Textual frontend's fleet renderer. It adapts a FleetView
(the durable contract) into a header line plus one row widget per session;
clicking a settled row opens its outcome overview."""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from broker.master.viewmodel import Attention, FleetView, SessionRow
from broker.protocol.constants import SessionState

# Content cells every sidebar line must fit in; the app sizes the pane as
# this plus its scrollbar.
FLEET_WIDTH = 54

_DOT = "●"
_MUTED = "dim"
_INDENT = "    "
_SUB_WIDTH = FLEET_WIDTH - len(_INDENT)

# (theme colour token, lifecycle label) per state; a token of ``_MUTED`` is a
# plain Rich style, the rest resolve through the app's current theme.
_STATE_STYLE: dict[SessionState, tuple[str, str]] = {
    SessionState.SPAWNING: ("primary", "starting"),
    SessionState.GROUNDING: ("primary", "grounding"),
    SessionState.AWAITING_APPROVAL: ("warning", "needs prompt"),
    SessionState.DRIVING: ("primary", "working"),
    SessionState.ESCALATED: ("warning", "needs decision"),
    SessionState.COMPLETED: ("success", "completed"),
    SessionState.ERROR: ("error", "error"),
    SessionState.STOPPED: (_MUTED, "stopped"),
    SessionState.UNMANAGED: (_MUTED, "unmanaged"),
}

_BADGE_TEXT: dict[Attention, str] = {
    Attention.ESCALATION: "needs decision",
    Attention.PROPOSAL: "approve prompt",
    Attention.PERMISSION: "permission",
    Attention.QUESTION: "question",
}


def _one_line(text: str, width: int) -> str:
    """Collapse ``text`` to a single line of at most ``width`` cells."""
    line = " ".join(text.split())
    if len(line) <= width:
        return line
    return line[: width - 1] + "…"


def _badge_labels(row: SessionRow) -> list[str]:
    """Return the row's badge labels, naming the pane on a pane badge."""
    labels: list[str] = []
    for badge in row.badges:
        text = _BADGE_TEXT[badge]
        if badge in (Attention.PERMISSION, Attention.QUESTION) and row.pane_id:
            text = f"{text} · {row.pane_id}"
        labels.append(text)
    return labels


def _theme_style(widget: Widget, token: str) -> str:
    """Resolve a theme colour token to a Rich style string for ``widget``."""
    if token == _MUTED:
        return _MUTED
    # Textual types Widget.app as App[Unknown]; the dict itself is concrete.
    variables: dict[str, str] = (
        widget.app.theme_variables  # pyright: ignore[reportUnknownMemberType]
    )
    return variables.get(token, "")


class SessionRowWidget(Static):
    """One session's row block. Clicking it opens the outcome overview, but only
    once the session has reached an outcome state (completed, error or stopped)."""

    DEFAULT_CSS = """
    SessionRowWidget { height: auto; margin: 0 0 1 0; }
    """

    class Selected(Message):
        """A settled row was clicked."""

        def __init__(self, session_id: str) -> None:
            self.session_id = session_id
            super().__init__()

    def __init__(self, row: SessionRow) -> None:
        """Render ``row``."""
        super().__init__()
        self._row = row
        self.update(self._render_row(row))

    @property
    def session_id(self) -> str:
        """The session this row renders."""
        return self._row.session_id

    def update_row(self, row: SessionRow) -> None:
        """Refresh this row's content in place."""
        self._row = row
        self.update(self._render_row(row))

    def on_click(self) -> None:
        """Click: ask the app to open this row's outcome."""
        if self._row.is_settled:
            self.post_message(self.Selected(self._row.session_id))

    def _render_row(self, row: SessionRow) -> Text:
        token, label = _STATE_STYLE[row.state]
        colour = _theme_style(self, token)
        out = Text()
        out.append(f"{_DOT} ", style=colour)
        out.append(f"{row.session_id}  ", style="bold")
        out.append(label, style=colour)
        out.append(f"   {row.budget_count}/{row.budget_max}", style=_MUTED)
        if row.title:
            out.append(f"\n{_INDENT}{_one_line(row.title, _SUB_WIDTH)}")
        primary = row.task_activity or row.broker_activity
        if primary:
            out.append(
                f"\n{_INDENT}{_one_line(primary, _SUB_WIDTH)}", style=_MUTED
            )
        if row.task_activity and row.broker_activity:
            out.append(
                f"\n{_INDENT}{_one_line(row.broker_activity, _SUB_WIDTH)}",
                style=_MUTED,
            )
        for badge in _badge_labels(row):
            out.append(
                f"\n{_INDENT}⚠ {badge}", style=_theme_style(self, "warning")
            )
        if row.is_settled:
            out.append(
                f"\n{_INDENT}View outcome ›", style=_theme_style(self, "primary")
            )
        return out


class FleetSidebar(VerticalScroll):
    """The session sidebar: a master/queue header plus one clickable row per
    session, reconciled from each FleetView."""

    def __init__(self, id: str) -> None:
        super().__init__(id=id)
        self._rows: dict[str, SessionRowWidget] = {}

    def compose(self) -> ComposeResult:
        yield Static(id="fleet-header")

    def update_view(self, view: FleetView) -> None:
        """Update the header, then reconcile the row widgets against ``view``."""
        self.query_one("#fleet-header", Static).update(self._render_header(view))
        self._drop_departed(view)
        self._add_or_refresh(view)

    def _drop_departed(self, view: FleetView) -> None:
        """Remove row widgets for sessions no longer in the view."""
        live = {row.session_id for row in view.rows}
        for session_id in self._rows.keys() - live:
            self._rows.pop(session_id).remove()

    def _add_or_refresh(self, view: FleetView) -> None:
        """Mount a widget for each new session; refresh the rest in place."""
        for row in view.rows:
            widget = self._rows.get(row.session_id)
            if widget is None:
                widget = SessionRowWidget(row)
                self._rows[row.session_id] = widget
                # New sessions always carry the highest id, so mounting at the
                # end keeps the rows in the view's ascending order.
                self.mount(widget)
            else:
                widget.update_row(row)

    def _render_header(self, view: FleetView) -> Text:
        out = Text()
        out.append(f"Master — {view.master_activity or 'idle'}", style="bold")
        if view.queue_depth == 0:
            out.append("\nall clear", style=_theme_style(self, "success"))
        else:
            n = view.queue_depth
            out.append(
                f"\n{n} request{'s' if n != 1 else ''} waiting",
                style=_theme_style(self, "warning"),
            )
        if not view.rows:
            out.append("\n\n(no sessions)", style=_MUTED)
        return out
