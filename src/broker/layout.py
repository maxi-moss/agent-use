"""Every path under ``broker_home``, in one place.

No other module joins a path inside the broker home: changing the on-disk
layout means changing this file and nothing else.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    """On-disk locations for one broker home."""

    home: Path

    @property
    def registry(self) -> Path:
        """The persisted session registry."""
        return self.home / "registry.json"

    @property
    def master_socket(self) -> Path:
        """The master's listening socket."""
        return self.home / "master.sock"

    def session_socket(self, name: str) -> Path:
        """The listening socket for session ``name``."""
        return self.home / "s" / f"{name}.sock"

    @property
    def logs(self) -> Path:
        """Root of the log tree."""
        return self.home / "logs"

    @property
    def master_log(self) -> Path:
        """The master's diagnostic log."""
        return self.logs / "master.log"

    @property
    def master_conversation(self) -> Path:
        """The master's append-only conversation log."""
        return self.logs / "master-log.ndjson"

    def session_logs(self, name: str) -> Path:
        """Log directory for session ``name``."""
        return self.logs / "sessions" / name

    def session_log(self, name: str) -> Path:
        """Diagnostic log for session ``name``."""
        return self.session_logs(name) / "broker.log"

    def session_decisions(self, name: str) -> Path:
        """Append-only triage decision log for session ``name``."""
        return self.session_logs(name) / "decisions.ndjson"
