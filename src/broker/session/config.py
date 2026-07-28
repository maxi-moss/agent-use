"""Session broker process configuration, passed as --config-json at spawn."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class AdoptedSession(BaseModel):
    """A live Claude session a replacement broker takes over instead of starting.

    Every field is required: a broker that only partly knows the session it is
    adopting would drive a pane it cannot read, or read a transcript it cannot
    drive.
    """

    model_config = ConfigDict(extra="forbid")

    pane_id: str
    claude_session_id: str
    transcript_path: str


class SessionBrokerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str
    socket_path: str
    master_socket_path: str
    broker_home: Path  # sent by the master; never re-derived from the env
    cwd: str
    anchor_pane: str
    intent: str
    budget_count: int = 0  # persisted count resumes across broker death
    model_id: str
    max_tokens: int
    watchdog_seconds: float
    budget_max: int
    adopt: AdoptedSession | None = None  # set only when reassigned
