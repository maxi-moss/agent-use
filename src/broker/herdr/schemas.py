"""Pydantic models for herdr CLI JSON results (verified against herdr 0.7.5)."""

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, model_validator


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


class AgentInfo(BaseModel):
    """AgentInfo per `herdr api schema --json` on the installed 0.7.5.
    agent_status enum (schema-verified): idle | working | blocked | done | unknown."""

    model_config = ConfigDict(extra="ignore")

    pane_id: str | None = None
    name: str | None = None
    interactive_ready: bool = False
    agent_status: str = "unknown"
    agent_session: AgentSession | None = None


class AgentStartResult(BaseModel):
    """`herdr agent start` result.

    The installed binary's API schema says the result is
    {"type": "agent_started", "agent": AgentInfo, "argv": [...]} — fields nested
    under `agent`, NOT flat as originally assumed. The validator lifts the
    nested shape; the flat shape still validates too, tolerated until a live
    capture settles it. agent_session stays OPTIONAL — absent when the trust
    dialog blocked session init."""

    model_config = ConfigDict(extra="ignore")

    interactive_ready: bool = False
    agent_session: AgentSession | None = None

    @model_validator(mode="before")
    @classmethod
    def _lift_nested_agent(cls, data: Any) -> Any:
        """Lift the nested ``agent`` object to the top level before validation.

        Args:
            data: Raw input handed to the model; only dicts are inspected.

        Returns:
            The nested ``agent`` object if present, otherwise ``data`` unchanged.
        """
        if not isinstance(data, dict):
            return data
        typed = cast(dict[str, Any], data)
        agent = typed.get("agent")
        if isinstance(agent, dict):
            return cast(dict[str, Any], agent)
        return typed
