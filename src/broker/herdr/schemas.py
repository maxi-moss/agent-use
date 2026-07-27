"""Pydantic models for herdr CLI JSON results (verified against herdr 0.7.5)."""

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
    — the Homebrew symlink can be rewritten mid-session (spikes/README.md §1)."""

    model_config = ConfigDict(extra="ignore")

    client: HerdrClientInfo
    server: HerdrServerInfo

    @property
    def compatible(self) -> bool:
        return self.server.compatible


class Pane(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pane_id: str


class PaneResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pane: Pane


class PaneInfo(BaseModel):
    """`herdr pane split` result — pane id at .result.pane.pane_id (spike-verified)."""

    model_config = ConfigDict(extra="ignore")

    result: PaneResult

    @property
    def pane_id(self) -> str:
        return self.result.pane.pane_id


class AgentSession(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent: str
    kind: str
    source: str
    value: str


class AgentStartResult(BaseModel):
    """`herdr agent start` result. agent_session is OPTIONAL — absent when the
    trust dialog blocked session init (spike delta vs spec §6/§9.11)."""

    model_config = ConfigDict(extra="ignore")

    interactive_ready: bool = False
    agent_session: AgentSession | None = None
