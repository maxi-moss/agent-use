"""Persisted pane escalations: broker_home/pane-escalations.json.

A pane escalation reports a native prompt — a permission prompt or an
AskUserQuestion menu — that the developer answers in the session's own pane;
the master only shows it and never answers it. At most one of each kind is
live per session, and none waits behind another. Every mutation persists
before it returns, so an open prompt survives a master restart.
"""

import json
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from broker.atomic_json import atomic_update_json
from broker.protocol.schemas import PANE_ESCALATION_ADAPTER, PaneEscalationPayload


class PaneStoreError(Exception):
    """The persisted pane-escalation file could not be read or validated."""


class PaneProtocolViolation(Exception):
    """A session raised a pane escalation while its previous one of that kind is live."""


class PaneEscalations:
    """Persisted per-session store of open pane escalations."""

    def __init__(
        self, path: Path, entries: list[PaneEscalationPayload] | None = None
    ) -> None:
        """Hold the store file path and the payloads loaded from it."""
        self._path = path
        self._entries: list[PaneEscalationPayload] = list(entries or [])

    @classmethod
    def load(cls, path: Path) -> "PaneEscalations":
        """Load the store from ``path``, starting empty if it is absent.

        Args:
            path: Store JSON file, normally ``broker_home/pane-escalations.json``.

        Returns:
            The loaded store, empty when the file does not exist.

        Raises:
            PaneStoreError: The file is not valid JSON, is not a JSON object,
                has a non-list ``pane_escalations`` entry, or holds an entry
                that fails validation. A skipped entry is an open prompt the
                developer is never shown, so none is skipped.
        """
        if not path.exists():
            return cls(path)
        try:
            parsed: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PaneStoreError(f"{path} is not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise PaneStoreError(f"{path} is not a JSON object")
        raw_entries = cast(dict[str, Any], parsed).get("pane_escalations", [])
        if not isinstance(raw_entries, list):
            raise PaneStoreError(f"{path}: 'pane_escalations' is not a list")
        entries: list[PaneEscalationPayload] = []
        for i, raw in enumerate(cast(list[Any], raw_entries)):
            try:
                entries.append(PANE_ESCALATION_ADAPTER.validate_python(raw))
            except ValidationError as exc:
                raise PaneStoreError(
                    f"{path}: invalid pane escalation {i}: {exc}"
                ) from exc
        return cls(path, entries)

    @property
    def entries(self) -> tuple[PaneEscalationPayload, ...]:
        """Return every open pane escalation, in arrival order."""
        return tuple(self._entries)

    def find(self, escalation_id: str) -> PaneEscalationPayload | None:
        """Return the open pane escalation ``escalation_id``, or ``None``."""
        return next(
            (p for p in self._entries if p.escalation_id == escalation_id), None
        )

    def accept(self, payload: PaneEscalationPayload) -> None:
        """Hold ``payload`` and persist it.

        Args:
            payload: The pane escalation to show the developer.

        Raises:
            PaneProtocolViolation: The session already has a live pane
                escalation of this kind; its raiser retracts the old one
                before raising the next.
        """
        for entry in self._entries:
            if entry.session_id == payload.session_id and entry.kind == payload.kind:
                raise PaneProtocolViolation(
                    f"{payload.kind} escalation {payload.escalation_id} arrived "
                    f"while {entry.escalation_id} is live"
                )
        self._entries.append(payload)
        self.save()

    def retract(self, escalation_id: str) -> PaneEscalationPayload | None:
        """Clear a pane escalation whose native prompt is no longer open.

        Args:
            escalation_id: The pane escalation being withdrawn.

        Returns:
            The cleared payload, or ``None`` when no live entry matched.
        """
        for i, entry in enumerate(self._entries):
            if entry.escalation_id == escalation_id:
                cleared = self._entries.pop(i)
                self.save()
                return cleared
        return None

    def retract_for_session(self, session_id: str) -> list[PaneEscalationPayload]:
        """Clear every live pane escalation of ``session_id``.

        Args:
            session_id: Session whose live entries — at most one per kind —
                are to be withdrawn.

        Returns:
            The cleared payloads, empty when the session had none live.
        """
        cleared = [p for p in self._entries if p.session_id == session_id]
        if cleared:
            self._entries = [p for p in self._entries if p.session_id != session_id]
            self.save()
        return cleared

    def save(self) -> None:
        """Write the in-memory entries back to the store file."""
        entries = [p.model_dump() for p in self._entries]

        def mutate(data: dict[str, Any]) -> dict[str, Any]:
            """Replace ``pane_escalations``, leaving the rest intact."""
            data["pane_escalations"] = entries
            return data

        atomic_update_json(self._path, mutate, backup=False)
