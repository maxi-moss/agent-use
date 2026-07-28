"""Persisted session registry: broker_home/registry.json.

The registry — not the transcript, not the conversation — is the durable
memory: socket paths, approved prompts (the authoritative
intent), and budget counters.
"""

import json
import re
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from broker.claude.atomic import atomic_update_json
from broker.protocol.constants import SessionState

NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")


class SessionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str  # herdr agent name, master-allocated
    socket_path: str  # allocated ONCE, persisted, never re-derived
    cwd: str
    anchor_pane: str
    state: SessionState = SessionState.SPAWNING
    pane_id: str | None = None
    claude_session_id: str | None = None
    transcript_path: str | None = None
    intent: str = ""  # raw intent at spawn
    approved_prompt: str | None = None  # the AUTHORITATIVE intent record
    budget_count: int = 0  # persisted; survives broker death
    pid: int | None = None


class RegistryError(Exception):
    """The registry file exists but cannot be read. Fail loud, never clobber."""


class Registry:
    def __init__(self, path: Path, records: dict[str, SessionRecord]) -> None:
        """Hold the registry file path and the records loaded from it."""
        self.path = path
        self.records = records

    @classmethod
    def load(cls, path: Path) -> "Registry":
        """Load the registry from ``path``, starting empty if it is absent.

        Args:
            path: Registry JSON file, normally ``broker_home/registry.json``.

        Returns:
            The loaded registry, empty when the file does not exist.

        Raises:
            RegistryError: The file is not valid JSON, is not a JSON object,
                has a non-object ``sessions`` entry, or holds a session record
                that fails validation.
        """
        if not path.exists():
            return cls(path, {})
        try:
            parsed: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{path} is not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise RegistryError(f"{path} is not a JSON object")
        raw_sessions = cast(dict[str, Any], parsed).get("sessions", {})
        if not isinstance(raw_sessions, dict):
            raise RegistryError(f"{path}: 'sessions' is not an object")
        records: dict[str, SessionRecord] = {}
        for name, raw in cast(dict[str, Any], raw_sessions).items():
            try:
                records[name] = SessionRecord.model_validate(raw)
            except ValidationError as exc:
                raise RegistryError(
                    f"{path}: invalid session record {name!r}: {exc}"
                ) from exc
        return cls(path, records)

    def save(self) -> None:
        """Write the in-memory records back to the registry file."""
        sessions = {
            name: record.model_dump() for name, record in self.records.items()
        }

        def mutate(data: dict[str, Any]) -> dict[str, Any]:
            """Replace ``sessions``, leaving the rest of the file intact."""
            data["sessions"] = sessions
            return data

        atomic_update_json(self.path, mutate)

    def allocate_name(self) -> str:
        """Allocate the next free session name.

        Returns:
            The allocated name, guaranteed to match ``NAME_RE``.
        """
        i = 1
        while f"s{i}" in self.records:
            i += 1
        name = f"s{i}"
        assert NAME_RE.fullmatch(name)
        return name

    def get(self, name: str) -> SessionRecord:
        """Return the record stored under ``name``.

        Args:
            name: Registry name of the session.

        Returns:
            The stored record.

        Raises:
            KeyError: No session by that name — fail loud rather than hand
                back a default the caller would act on.
        """
        if name not in self.records:
            raise KeyError(f"unknown session {name!r}")  # fail loud
        return self.records[name]

    def upsert(self, record: SessionRecord) -> None:
        """Store a record under its own name and persist the registry.

        Args:
            record: Session record to store; replaces any record already held
                under the same name.

        Raises:
            ValueError: ``record.name`` does not match ``NAME_RE``.
        """
        if not NAME_RE.fullmatch(record.name):
            raise ValueError(f"invalid session name {record.name!r}")
        self.records[record.name] = record
        self.save()
