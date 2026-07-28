"""Socket-protocol pydantic models. Brokers only — the hook must never import this.

Every model here is extra="ignore", never extra="forbid": an unknown wire field
means a newer peer, not a typo. Config models are the strict half of that pair.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from broker.protocol.constants import PROTOCOL_VERSION, SessionState


class Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    v: int = PROTOCOL_VERSION
    id: str  # uuid4 hex
    type: str
    session_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class Response(BaseModel):
    model_config = ConfigDict(extra="ignore")

    v: int = PROTOCOL_VERSION
    id: str  # reuses request id
    type: Literal["response"] = "response"
    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)


class PermissionRequestPayload(BaseModel):
    """hook -> broker; mirrors the PreToolUse stdin payload."""

    model_config = ConfigDict(extra="ignore")

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    cwd: str
    transcript_path: str
    permission_mode: str | None = None


class PermissionDecisionPayload(BaseModel):
    """broker -> hook reply payload."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["allow", "escalated"]


class HookEventPayload(BaseModel):
    """Fire-and-forget wrapper for every non-PreToolUse hook event."""

    model_config = ConfigDict(extra="ignore")

    hook_event_name: str
    raw: dict[str, Any]  # full stdin payload, untyped by design


class DispatchDecisionPayload(BaseModel):
    """master -> broker: developer's resolution of an escalation."""

    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    response: str


class SendPromptPayload(BaseModel):
    """master -> broker: relay a developer prompt into the session."""

    model_config = ConfigDict(extra="ignore")

    text: str


class DecisionLogPayload(BaseModel):
    """broker -> master reply payload carrying a rendered decision log."""

    model_config = ConfigDict(extra="ignore")

    text: str


class StatusPayload(BaseModel):
    """broker -> master reply payload for the attach/liveness probe."""

    model_config = ConfigDict(extra="ignore")

    state: SessionState
    pane_id: str | None = None
    claude_session_id: str | None = None
    transcript_path: str | None = None


class Alternative(BaseModel):
    model_config = ConfigDict(extra="ignore")

    option: str
    pros: str
    cons: str


class EscalationPayload(BaseModel):
    """broker -> master structured escalation object."""

    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    session_id: str
    task_context: str
    situation: str
    what_was_asked: str
    what_is_at_stake: str
    alternatives: list[Alternative]
    recommendation: str
    uncertainty: str
    what_would_change_my_mind: str


class CompletionPayload(BaseModel):
    """broker -> master completion notice with summary."""

    model_config = ConfigDict(extra="ignore")

    summary: str


class FatalErrorPayload(BaseModel):
    """broker -> master fatal error."""

    model_config = ConfigDict(extra="ignore")

    error_class: str
    detail: str


class RetractPayload(BaseModel):
    """broker -> master: escalation resolved out of band."""

    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    reason: str


class PromptProposalPayload(BaseModel):
    """broker -> master: grounded prompt awaiting developer approval."""

    model_config = ConfigDict(extra="ignore")

    proposal_id: str
    proposed_prompt: str
    grounding_summary: str


class ApprovePromptPayload(BaseModel):
    """master -> broker: the developer-approved (possibly revised) prompt."""

    model_config = ConfigDict(extra="ignore")

    proposal_id: str
    prompt: str


class ReactivatePayload(BaseModel):
    """master -> broker: a new task for a session that already completed one."""

    model_config = ConfigDict(extra="ignore")

    intent: str


class BudgetUpdatePayload(BaseModel):
    """broker -> master: autonomous-answer counter for registry persistence."""

    model_config = ConfigDict(extra="ignore")

    count: int
