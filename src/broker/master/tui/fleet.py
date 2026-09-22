"""FleetSidebar: the Textual frontend's fleet renderer. It adapts a FleetView
(the durable contract) into this frontend's widgets."""

from rich.text import Text
from textual.widgets import Static

from broker.master.outcome import OUTCOME_STATES
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
}


def _one_line(text: str, width: int) -> str:
    """Collapse ``text`` to a single line of at most ``width`` cells."""
    line = " ".join(text.split())
    if len(line) <= width:
        return line
    return line[: width - 1] + "…"


def _badge_labels(row: SessionRow) -> list[str]:
    """Return the row's badge labels, naming the pane on a permission badge."""
    labels: list[str] = []
    for badge in row.badges:
        text = _BADGE_TEXT[badge]
        if badge is Attention.PERMISSION and row.pane_id:
            text = f"{text} · {row.pane_id}"
        labels.append(text)
    return labels


class FleetSidebar(Static):
    """One Static whose content is rebuilt from each FleetView."""

    def update_view(self, view: FleetView) -> None:
        """Replace the sidebar content with a rendering of ``view``."""
        self.update(self._render_view(view))

    def _style(self, token: str) -> str:
        """Resolve a theme colour token to a Rich style string."""
        if token == _MUTED:
            return _MUTED
        # Textual types Widget.app as App[Unknown]; the dict itself is concrete.
        variables: dict[str, str] = (
            self.app.theme_variables  # pyright: ignore[reportUnknownMemberType]
        )
        return variables.get(token, "")

    def _render_view(self, view: FleetView) -> Text:
        """Render the master line, the queue line and one block per session."""
        out = Text()
        out.append(f"Master — {view.master_activity or 'idle'}\n", style="bold")
        if view.queue_depth == 0:
            out.append("all clear\n", style=self._style("success"))
        else:
            n = view.queue_depth
            out.append(
                f"{n} request{'s' if n != 1 else ''} waiting\n",
                style=self._style("warning"),
            )
        out.append("\n")
        if not view.rows:
            out.append("(no sessions)", style=_MUTED)
            return out
        warning = self._style("warning")
        for row in view.rows:
            token, label = _STATE_STYLE[row.state]
            colour = self._style(token)
            out.append(f"{_DOT} ", style=colour)
            out.append(f"{row.session_id}  ", style="bold")
            out.append(label, style=colour)
            out.append(f"   {row.budget_count}/{row.budget_max}\n", style=_MUTED)
            if row.title:
                out.append(f"{_INDENT}{_one_line(row.title, _SUB_WIDTH)}\n")
            primary = row.task_activity or row.broker_activity
            if primary:
                out.append(
                    f"{_INDENT}{_one_line(primary, _SUB_WIDTH)}\n", style=_MUTED
                )
            if row.task_activity and row.broker_activity:
                out.append(
                    f"{_INDENT}{_one_line(row.broker_activity, _SUB_WIDTH)}\n",
                    style=_MUTED,
                )
            for badge in _badge_labels(row):
                out.append(f"{_INDENT}⚠ {badge}\n", style=warning)
            if row.state in OUTCOME_STATES:
                out.append(
                    f"{_INDENT}/outcome {row.session_id} — view outcome\n",
                    style=_MUTED,
                )
            out.append("\n")
        return out
