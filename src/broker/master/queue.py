"""Persisted strict-FIFO decision-escalation queue: broker_home/escalation-queue.json.

One escalation is live per session; the head is the one surfaced to the
developer. Order is the file's list order — every mutation persists before it
returns, so a pending escalation survives a master restart. Pane
escalations are never queued here: they live in
``broker.master.pane_escalations``.
"""

import json
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from broker.atomic_json import atomic_update_json
from broker.protocol.schemas import EscalationPayload


class QueueError(Exception):
    """The persisted queue file could not be read or validated."""


class EscalationProtocolViolation(Exception):
    """A session broke the one-outstanding-escalation invariant."""


class EscalationQueue:
    """Persisted strict-FIFO decision-escalation store (accept / retract /
    resolve / active)."""

    def __init__(
        self, path: Path, entries: list[EscalationPayload] | None = None
    ) -> None:
        """Hold the queue file path and the payloads loaded from it."""
        self._path = path
        self._entries: list[EscalationPayload] = list(entries or [])

    @classmethod
    def load(cls, path: Path) -> "EscalationQueue":
        """Load the queue from ``path``, starting empty if it is absent.

        Args:
            path: Queue JSON file, normally ``broker_home/escalation-queue.json``.

        Returns:
            The loaded queue, empty when the file does not exist.

        Raises:
            QueueError: The file is not valid JSON, is not a JSON object, has
                a non-list ``queue`` entry, or holds an entry that fails
                validation. A silently dropped escalation is a silently
                dropped decision, so no entry is ever skipped.
        """
        if not path.exists():
            return cls(path)
        try:
            parsed: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise QueueError(f"{path} is not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise QueueError(f"{path} is not a JSON object")
        raw_queue = cast(dict[str, Any], parsed).get("queue", [])
        if not isinstance(raw_queue, list):
            raise QueueError(f"{path}: 'queue' is not a list")
        entries: list[EscalationPayload] = []
        for i, raw in enumerate(cast(list[Any], raw_queue)):
            try:
                entries.append(EscalationPayload.model_validate(raw))
            except ValidationError as exc:
                raise QueueError(
                    f"{path}: invalid queue entry {i}: {exc}"
                ) from exc
        return cls(path, entries)

    @property
    def active(self) -> EscalationPayload | None:
        """Return the escalation awaiting the developer, or ``None``."""
        return self._entries[0] if self._entries else None

    @property
    def depth(self) -> int:
        """Return the number of live escalations, the head included."""
        return len(self._entries)

    @property
    def waiting(self) -> tuple[str, ...]:
        """Return the session ids of the entries behind the head, in order."""
        return tuple(p.session_id for p in self._entries[1:])

    @property
    def entries(self) -> tuple[EscalationPayload, ...]:
        """Return every live entry, head first, as a read-only snapshot."""
        return tuple(self._entries)

    def accept(self, payload: EscalationPayload) -> None:
        """Append ``payload`` to the queue and persist it.

        Args:
            payload: The escalation to queue for the developer.

        Raises:
            EscalationProtocolViolation: The session already has a live
                escalation, surfaced or waiting — it is expected to wait for
                its own escalation to resolve.
        """
        for entry in self._entries:
            if entry.session_id == payload.session_id:
                raise EscalationProtocolViolation(
                    f"escalation {payload.escalation_id} arrived while "
                    f"{entry.escalation_id} is live"
                )
        self._entries.append(payload)
        self.save()

    def retract(self, escalation_id: str) -> EscalationPayload | None:
        """Clear an escalation its session has withdrawn, wherever it waits.

        Args:
            escalation_id: The escalation being withdrawn.

        Returns:
            The cleared payload, or ``None`` when no live entry matched.
        """
        for i, entry in enumerate(self._entries):
            if entry.escalation_id == escalation_id:
                cleared = self._entries.pop(i)
                self.save()
                return cleared
        return None

    def resolve(self, escalation_id: str) -> EscalationPayload | None:
        """Clear an escalation the developer has decided.

        Only a matching head is cleared: a stale resolve must be a no-op so a
        retraction crossing a dispatch cannot mis-pop the queue.

        Args:
            escalation_id: The escalation that was answered.

        Returns:
            The cleared payload, or ``None`` when it was not the head.
        """
        if self._entries and self._entries[0].escalation_id == escalation_id:
            cleared = self._entries.pop(0)
            self.save()
            return cleared
        return None

    def retract_for_session(self, session_id: str) -> EscalationPayload | None:
        """Clear the live escalation raised by ``session_id``, if any.

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
        """Write the in-memory entries back to the queue file."""
        queue = [p.model_dump() for p in self._entries]

        def mutate(data: dict[str, Any]) -> dict[str, Any]:
            """Replace ``queue``, leaving the rest of the file intact."""
            data["queue"] = queue
            return data

        atomic_update_json(self._path, mutate, backup=False)
