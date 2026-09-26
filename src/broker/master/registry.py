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

from broker.atomic_json import atomic_update_json
from broker.herdr.driver import AGENT_NAME_RE
from broker.protocol.constants import SessionState

_SESSION_NUM = re.compile(r"s(\d+)\Z")


def session_sort_key(name: str) -> tuple[int, str]:
    """Order sessions by numeric id (s2 before s10); any non-'sN' name last."""
    m = _SESSION_NUM.match(name)
    return (int(m.group(1)), "") if m else (10**9, name)


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
    title: str = ""  # short task label set at approval; shown in the fleet
    budget_count: int = 0  # persisted; survives broker death
    pid: int | None = None


class RegistryError(Exception):
    """The registry file exists but cannot be read. Fail loud, never clobber."""


class Registry:
    def __init__(
        self, path: Path, records: dict[str, SessionRecord], name_seq: int
    ) -> None:
        """Hold the registry file path, the records, and the name counter."""
        self.path = path
        self.records = records
        self._name_seq = name_seq

    @classmethod
    def load(cls, path: Path) -> "Registry":
        """Load the registry from ``path``, starting empty if it is absent.

        Args:
            path: Registry JSON file, normally ``broker_home/registry.json``.

        Returns:
            The loaded registry, empty when the file does not exist.

        Raises:
            RegistryError: The file is not valid JSON, is not a JSON object,
                has a non-object ``sessions`` entry, a non-integer
                ``name_seq``, or holds a session record that fails validation.
        """
        if not path.exists():
            return cls(path, {}, 0)
        try:
            parsed: Any = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{path} is not valid JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise RegistryError(f"{path} is not a JSON object")
        obj = cast(dict[str, Any], parsed)
        raw_sessions = obj.get("sessions", {})
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
        name_seq = obj.get("name_seq", 0)
        if not isinstance(name_seq, int) or isinstance(name_seq, bool):
            raise RegistryError(f"{path}: 'name_seq' is not an integer")
        return cls(path, records, name_seq)

    def save(self) -> None:
        """Write the in-memory records back to the registry file."""
        sessions = {
            name: record.model_dump() for name, record in self.records.items()
        }

        def mutate(data: dict[str, Any]) -> dict[str, Any]:
            """Replace ``sessions`` and the counter, leaving the rest intact."""
            data["sessions"] = sessions
            data["name_seq"] = self._name_seq
            return data

        atomic_update_json(self.path, mutate, backup=False)

    def allocate_name(self) -> str:
        """Allocate the next session name, never reusing a freed one.

        Returns:
            The allocated name, guaranteed to match ``AGENT_NAME_RE``.
        """
        # Monotonic, never a scan of `records`: a finished session is removed,
        # so a free-slot scan would rehand its name while its socket and logs
        # still exist on disk.
        self._name_seq += 1
        name = f"s{self._name_seq}"
        assert AGENT_NAME_RE.fullmatch(name)
        return name

    def names_in_order(self) -> list[str]:
        """Return every session name, in numeric order (s2 before s10)."""
        return sorted(self.records, key=session_sort_key)

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
            ValueError: ``record.name`` does not match ``AGENT_NAME_RE``.
        """
        if not AGENT_NAME_RE.fullmatch(record.name):
            raise ValueError(f"invalid session name {record.name!r}")
        self.records[record.name] = record
        self.save()

    def remove(self, name: str) -> None:
        """Drop a finished session from the registry and persist the removal.

        Args:
            name: Session to forget; a no-op when it is already absent.
        """
        if self.records.pop(name, None) is None:
            return
        self.save()
