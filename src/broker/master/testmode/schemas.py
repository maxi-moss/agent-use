"""Scenario step models: a discriminated union on ``op``, strict by design.

Unknown keys in a scenario file are typos, not a newer peer, so every model
forbids extras — the opposite of the wire models, which ignore unknown fields.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from broker.protocol.constants import SessionState


class ScenarioError(Exception):
    """A scenario could not be loaded or names something that does not exist."""


class SeedSession(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["seed_session"]
    name: str
    state: SessionState = SessionState.DRIVING
    pane_id: str | None = None
    claude_session_id: str | None = None
    transcript_path: str | None = None
    approved_prompt: str | None = None


class StartFakeSocket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["start_socket"]
    session: str


class StopFakeSocket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["stop_socket"]
    session: str


class Escalate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["escalate"]
    session: str
    escalation_id: str
    expect: str = "ack"


class PermissionEscalate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["permission_escalate"]
    session: str
    escalation_id: str
    tool_name: str
    tool_input: dict[str, Any]
    raised_at: str
    expect: str = "ack"


class EscalationRetract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["escalation_retract"]
    session: str
    escalation_id: str
    reason: str


class PermissionRetract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["permission_retract"]
    session: str
    escalation_id: str
    reason: str


class Dispatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["dispatch"]
    escalation_id: str
    decision: str
    expect: Literal["dispatched", "refused"]


class Deliver(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["deliver"]
    session: str
    escalation_id: str


class Attach(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["attach"]
    session: str
    # "refused" is the only outcome a scenario uses: an "attached" expectation
    # would spawn a real broker, which stays out of the scenario runner.
    expect: Literal["attached", "refused"] = "refused"


class AssertSurfaced(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["assert_surfaced"]
    escalation_id: str


class AssertNeverSurfaced(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["assert_never_surfaced"]
    escalation_id: str


class AssertDepth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["assert_depth"]
    depth: int
    waiting: list[str] = Field(default_factory=list[str])


class AssertOpenPermissions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["assert_open_permissions"]
    escalation_ids: list[str] = Field(default_factory=list[str])


class AssertIsolated(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["assert_isolated"]


class AssertNoDispatchWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["assert_no_dispatch_write"]
    session: str


class AssertUnreachable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["assert_unreachable"]
    session: str


Step = Annotated[
    SeedSession
    | StartFakeSocket
    | StopFakeSocket
    | Escalate
    | PermissionEscalate
    | EscalationRetract
    | PermissionRetract
    | Dispatch
    | Deliver
    | Attach
    | AssertSurfaced
    | AssertNeverSurfaced
    | AssertDepth
    | AssertOpenPermissions
    | AssertIsolated
    | AssertNoDispatchWrite
    | AssertUnreachable,
    Field(discriminator="op"),
]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    steps: list[Step]


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    op: str
    passed: bool
    detail: str


class ScenarioReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    results: list[StepResult] = Field(default_factory=list[StepResult])

    @property
    def passed(self) -> bool:
        """True when every step passed."""
        return all(r.passed for r in self.results)
