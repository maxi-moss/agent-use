"""AttentionNotice: the pinned strip above the master chat naming what needs
the developer right now — the queue head and any proposals awaiting approval.
It renders from FleetView alone and hides itself when nothing is waiting; the
disclosures themselves enter the chat only on /escalation or /proposal."""

from rich.text import Text
from textual.widgets import Static

from broker.master.viewmodel import Attention, FleetView


class AttentionNotice(Static):
    """One line for the queue head, one per pending proposal."""

    DEFAULT_CSS = """
    AttentionNotice {
        height: auto;
        color: $warning;
        border-bottom: solid $warning 50%;
        padding: 0 1;
    }
    """

    def update_view(self, view: FleetView) -> None:
        """Rebuild the strip from ``view``; hide it when nothing is waiting."""
        lines = self._lines(view)
        self.display = bool(lines)
        self.update(Text("\n".join(lines)))

    @staticmethod
    def _lines(view: FleetView) -> list[str]:
        lines: list[str] = []
        head_line = AttentionNotice._head_line(view)
        if head_line is not None:
            lines.append(head_line)
        for row in view.rows:
            if Attention.PROPOSAL in row.badges:
                lines.append(
                    f"⚠ {row.session_id} waiting for opening-prompt approval"
                    f" — /proposal {row.session_id} shows it"
                )
        return lines

    @staticmethod
    def _head_line(view: FleetView) -> str | None:
        head = view.head
        if head is None:
            return None
        n = view.queue_depth
        count = f"{n} request{'s' if n != 1 else ''} waiting"
        if head.kind is Attention.PERMISSION:
            pane = next(
                (r.pane_id for r in view.rows if r.session_id == head.session_id),
                None,
            )
            where = f"pane {pane}" if pane else "its pane"
            return (
                f"⚠ {count} · {head.session_id} permission prompt for "
                f"{head.text} — answer it in {where}"
            )
        return (
            f"⚠ {count} · {head.session_id} needs a decision: {head.text}"
            " — /escalation shows the disclosure"
        )
