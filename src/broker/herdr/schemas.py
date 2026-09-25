"""Pydantic models for herdr CLI JSON results (verified against herdr 0.8.2).

Every model here is extra="ignore", never extra="forbid": a new key in herdr's
JSON output is a herdr upgrade, not a typo, and must not fail the parse.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HerdrClientInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: str


class HerdrServerInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: str
    compatible: bool


class HerdrStatus(BaseModel):
    """`herdr status --json`. Trust `.server.compatible`, never `herdr --version`
    — the Homebrew symlink can be rewritten mid-session."""

    model_config = ConfigDict(extra="ignore")

    client: HerdrClientInfo
    server: HerdrServerInfo

    @property
    def compatible(self) -> bool:
        """Return whether the running server is compatible with the client."""
        return self.server.compatible


class Pane(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pane_id: str


class PaneResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pane: Pane


class PaneInfo(BaseModel):
    """`herdr pane split` result — pane id at .result.pane.pane_id."""

    model_config = ConfigDict(extra="ignore")

    result: PaneResult

    @property
    def pane_id(self) -> str:
        """Return the new pane's id from the nested result."""
        return self.result.pane.pane_id


class AgentSession(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent: str
    kind: str
    source: str
    value: str


AgentStatus = Literal["idle", "working", "blocked", "done", "unknown"]


class AgentInfo(BaseModel):
    """`agent start`/`agent get` result, from ``result.agent`` (installed
    herdr 0.8.2, live capture). agent_status is the schema-verified
    AgentStatus enum."""

    model_config = ConfigDict(extra="ignore")

    pane_id: str | None = None
    name: str | None = None
    agent_status: AgentStatus = "unknown"
    agent_session: AgentSession | None = None
