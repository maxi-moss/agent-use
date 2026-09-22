"""Every path under ``broker_home``, in one place.

No other module joins a path inside the broker home: moving a file or socket
on disk means changing this file and nothing else.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BrokerPaths:
    """On-disk locations for one broker home."""

    home: Path

    @property
    def registry(self) -> Path:
        """The persisted session registry."""
        return self.home / "registry.json"

    @property
    def escalation_queue(self) -> Path:
        """The persisted escalation queue."""
        return self.home / "escalation-queue.json"

    @property
    def permission_escalations(self) -> Path:
        """The persisted open permission escalations."""
        return self.home / "permission-escalations.json"

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

    @property
    def llm_timings(self) -> Path:
        """Shared append-only LLM-call timing log, written by master and sessions."""
        return self.logs / "llm-timings.ndjson"

    def session_logs(self, name: str) -> Path:
        """Log directory for session ``name``."""
        return self.logs / "sessions" / name

    def session_log(self, name: str) -> Path:
        """Diagnostic log for session ``name``."""
        return self.session_logs(name) / "broker.log"

    def session_decisions(self, name: str) -> Path:
        """Append-only triage decision log for session ``name``."""
        return self.session_logs(name) / "decisions.ndjson"

    def session_claude_settings(self, name: str) -> Path:
        """Claude Code settings file loaded by session ``name``."""
        return self.session_logs(name) / "claude-settings.json"

    def session_permissions(self, name: str) -> Path:
        """Append-only permission decision log for session ``name``."""
        return self.session_logs(name) / "permissions.ndjson"

    @property
    def index_dir(self) -> Path:
        """Root of the per-repository code indexes."""
        return self.home / "index"

    def index_db(self, repo: Path) -> Path:
        """The code index for the repository rooted at ``repo``.

        Args:
            repo: Absolute, resolved repository root. Callers resolve it; the
                same string must hash identically from the CLI and a spawn.
        """
        digest = hashlib.sha256(str(repo).encode("utf-8")).hexdigest()
        return self.index_dir / f"{digest}.sqlite"

    @property
    def index_log(self) -> Path:
        """Diagnostic log for the index CLI."""
        return self.logs / "index.log"
