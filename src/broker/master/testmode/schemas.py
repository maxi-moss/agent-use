"""Scenario step models: a discriminated union on ``op``, strict by design.

Unknown keys in a scenario file are typos, not a newer peer, so every model
forbids extras — the opposite of the wire models, which ignore unknown fields.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from broker.protocol.constants import PaneKind, SessionState


class ScenarioError(Exception):
    """A scenario could not be loaded or names something that does not exist."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


NackExpectation = Annotated[str, StringConstraints(pattern=r"^nack:[a-z_]+$")]


class SeedSessionStep(_StrictModel):
    op: Literal["seed_session"]
    name: str
    state: SessionState = SessionState.DRIVING
    pane_id: str | None = None
    claude_session_id: str | None = None
    transcript_path: str | None = None
    approved_prompt: str | None = None


class StartSocketStep(_StrictModel):
    op: Literal["start_socket"]
    session: str


class StopSocketStep(_StrictModel):
    op: Literal["stop_socket"]
    session: str


class EscalateStep(_StrictModel):
    op: Literal["escalate"]
    session: str
    escalation_id: str
    expect: Literal["ack"] | NackExpectation = "ack"


class PermissionEscalateStep(_StrictModel):
    op: Literal["permission_escalate"]
    session: str
    escalation_id: str
    tool_name: str
    tool_input: dict[str, Any]
    expect: Literal["ack"] | NackExpectation = "ack"


class QuestionEscalateStep(_StrictModel):
    op: Literal["question_escalate"]
    session: str
    escalation_id: str
    question: str
    expect: Literal["ack"] | NackExpectation = "ack"


class EscalationRetractStep(_StrictModel):
    op: Literal["escalation_retract"]
    session: str
    escalation_id: str
    reason: str


class PaneRetractStep(_StrictModel):
    op: Literal["pane_retract"]
    session: str
    escalation_id: str
    reason: str


class DispatchStep(_StrictModel):
    op: Literal["dispatch"]
    escalation_id: str
    decision: str
    expect: Literal["dispatched", "refused"]


class DeliverStep(_StrictModel):
    op: Literal["deliver"]
    session: str
    escalation_id: str


class AttachStep(_StrictModel):
    op: Literal["attach"]
    session: str
    # "refused" is the only outcome a scenario uses: an "attached" expectation
    # would spawn a real broker, which stays out of the scenario runner.
    expect: Literal["attached", "refused"] = "refused"


class AssertSurfacedStep(_StrictModel):
    op: Literal["assert_surfaced"]
    escalation_id: str


class AssertNeverSurfacedStep(_StrictModel):
    op: Literal["assert_never_surfaced"]
    escalation_id: str


class AssertDepthStep(_StrictModel):
    op: Literal["assert_depth"]
    depth: int
    waiting: list[str] = Field(default_factory=list[str])


class AssertOpenPanesStep(_StrictModel):
    op: Literal["assert_open_panes"]
    kind: PaneKind
    session_ids: list[str] = Field(default_factory=list[str])
    ids: list[str] = Field(default_factory=list[str])


class AssertIsolatedStep(_StrictModel):
    op: Literal["assert_isolated"]


class AssertNoDispatchWriteStep(_StrictModel):
    op: Literal["assert_no_dispatch_write"]
    session: str


class AssertUnreachableStep(_StrictModel):
    op: Literal["assert_unreachable"]
    session: str


StepModel = (
    SeedSessionStep
    | StartSocketStep
    | StopSocketStep
    | EscalateStep
    | PermissionEscalateStep
    | QuestionEscalateStep
    | EscalationRetractStep
    | PaneRetractStep
    | DispatchStep
    | DeliverStep
    | AttachStep
    | AssertSurfacedStep
    | AssertNeverSurfacedStep
    | AssertDepthStep
    | AssertOpenPanesStep
    | AssertIsolatedStep
    | AssertNoDispatchWriteStep
    | AssertUnreachableStep
)

Step = Annotated[StepModel, Field(discriminator="op")]


class Scenario(_StrictModel):
    name: str
    steps: list[Step]


class StepResult(_StrictModel):
    index: int
    op: str
    passed: bool
    detail: str


class ScenarioReport(_StrictModel):
    name: str
    results: list[StepResult] = Field(default_factory=list[StepResult])

    @property
    def passed(self) -> bool:
        """True when every step passed."""
        return all(r.passed for r in self.results)
