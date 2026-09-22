"""Persisted permission escalations: broker_home/permission-escalations.json.

A permission escalation reports a native permission prompt that the developer
answers in the session's own pane; the master only shows it and never answers
it. At most one is live per session, and none waits behind another. Every
mutation persists before it returns, so an open prompt survives a master
restart.
"""

import json
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from broker.claude.atomic import atomic_update_json
from broker.protocol.schemas import PermissionEscalationPayload


class PermissionStoreError(Exception):
    """The persisted permission-escalation file could not be read or validated."""


class PermissionProtocolViolation(Exception):
    """A session raised a permission escalation while its previous one is live."""


class PermissionEscalations:
    """Persisted per-session store of open permission escalations."""

    def __init__(
        self, path: Path, entries: list[PermissionEscalationPayload] | None = None
    ) -> None:
        """Hold the store file path and the payloads loaded from it."""
        self._path = path
        self._entries: list[PermissionEscalationPayload] = list(entries or [])

    @classmethod
    def load(cls, path: Path) -> "PermissionEscalations":
        """Load the store from ``path``, starting empty if it is absent.

        Args:
            path: Store JSON file, normally
                ``broker_home/permission-escalations.json``.

        Returns:
            The loaded store, empty when the file does not exist.

        Raises:
            PermissionStoreError: The file is not valid JSON, is not a JSON
                object, has a non-list ``permission_escalations`` entry, or
                holds an entry that fails validation. A skipped entry is an
                open prompt the developer is never shown, so none is skipped.
        """
        if not path.exists():
            return cls(path)
        try:
            parsed: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PermissionStoreError(
                f"{path} is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(parsed, dict):
            raise PermissionStoreError(f"{path} is not a JSON object")
        raw_entries = cast(dict[str, Any], parsed).get("permission_escalations", [])
        if not isinstance(raw_entries, list):
            raise PermissionStoreError(
                f"{path}: 'permission_escalations' is not a list"
            )
        entries: list[PermissionEscalationPayload] = []
        for i, raw in enumerate(cast(list[Any], raw_entries)):
            try:
                entries.append(PermissionEscalationPayload.model_validate(raw))
            except ValidationError as exc:
                raise PermissionStoreError(
                    f"{path}: invalid permission escalation {i}: {exc}"
                ) from exc
        return cls(path, entries)

    @property
    def entries(self) -> tuple[PermissionEscalationPayload, ...]:
        """Return every open permission escalation, in arrival order."""
        return tuple(self._entries)

    def find(self, escalation_id: str) -> PermissionEscalationPayload | None:
        """Return the open permission escalation ``escalation_id``, or ``None``."""
        return next(
            (p for p in self._entries if p.escalation_id == escalation_id), None
        )

    def accept(self, payload: PermissionEscalationPayload) -> None:
        """Hold ``payload`` and persist it.

        Args:
            payload: The permission escalation to show the developer.

        Raises:
            PermissionProtocolViolation: The session already has a live
                permission escalation; its permission module retracts the old
                one before raising the next.
        """
        for entry in self._entries:
            if entry.session_id == payload.session_id:
                raise PermissionProtocolViolation(
                    f"permission escalation {payload.escalation_id} arrived "
                    f"while {entry.escalation_id} is live"
                )
        self._entries.append(payload)
        self.save()

    def retract(self, escalation_id: str) -> PermissionEscalationPayload | None:
        """Clear a permission escalation whose prompt is no longer open.

        Args:
            escalation_id: The permission escalation being withdrawn.

        Returns:
            The cleared payload, or ``None`` when no live entry matched.
        """
        for i, entry in enumerate(self._entries):
            if entry.escalation_id == escalation_id:
                cleared = self._entries.pop(i)
                self.save()
                return cleared
        return None

    def retract_for_session(
        self, session_id: str
    ) -> PermissionEscalationPayload | None:
        """Clear the live permission escalation of ``session_id``, if any.

        Args:
            session_id: Session whose live entry — there is at most one — is
                to be withdrawn.

        Returns:
            The cleared payload, or ``None`` when the session had none live.
        """
        for i, entry in enumerate(self._entries):
            if entry.session_id == session_id:
                cleared = self._entries.pop(i)
                self.save()
                return cleared
        return None

    def save(self) -> None:
        """Write the in-memory entries back to the store file."""
        entries = [p.model_dump() for p in self._entries]

        def mutate(data: dict[str, Any]) -> dict[str, Any]:
            """Replace ``permission_escalations``, leaving the rest intact."""
            data["permission_escalations"] = entries
            return data

        atomic_update_json(self._path, mutate)
