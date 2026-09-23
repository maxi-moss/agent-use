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


class AddDirectoriesSuggestion(BaseModel):
    """Claude Code's offer to widen the accessible directory set."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["addDirectories"]
    directories: list[str] = Field(default_factory=list[str])
    destination: str | None = None


class SetModeSuggestion(BaseModel):
    """Claude Code's offer to switch the session's permission mode."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["setMode"]
    mode: str | None = None
    destination: str | None = None


# The suggestion set is undocumented and grows with the binary, so an arm we
# do not recognise must survive as its raw object rather than fail the
# permission request that carries it.
PermissionSuggestion = AddDirectoriesSuggestion | SetModeSuggestion | dict[str, Any]


class PermissionRequestPayload(BaseModel):
    """hook -> broker; mirrors the PermissionRequest stdin payload."""

    model_config = ConfigDict(extra="ignore")

    tool_name: str
    tool_input: dict[str, Any]
    cwd: str
    transcript_path: str
    permission_mode: str | None = None
    permission_suggestions: list[PermissionSuggestion] = Field(
        default_factory=list[PermissionSuggestion]
    )


class PermissionDecisionPayload(BaseModel):
    """broker -> hook reply payload."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["allow", "escalated"]


class AskQuestionRequestPayload(BaseModel):
    """hook -> broker: a pending AskUserQuestion awaiting a decision."""

    model_config = ConfigDict(extra="ignore")

    tool_input: dict[str, Any]
    tool_use_id: str


class AskQuestionDecisionPayload(BaseModel):
    """broker -> hook reply for a pending AskUserQuestion."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["answer", "escalated"]
    updated_input: dict[str, Any] | None = None


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


class DecisionDeliveredPayload(BaseModel):
    """broker -> master: a dispatched decision reached the pane.

    The master resolves the escalation only on this confirmation — the dispatch
    ACK means the broker accepted the decision for processing, not that it
    landed. Binding resolution to delivery keeps the two sides from diverging.
    """

    model_config = ConfigDict(extra="ignore")

    escalation_id: str


class DecisionUndeliveredPayload(BaseModel):
    """broker -> master: a dispatched decision did not reach the pane.

    Sent when a pane write fails or the dispatch was stale. Since resolution
    waits for delivery, the escalation was never resolved: ``still_live`` says
    whether the broker still holds it (a failed write — keep it surfaced for a
    re-decide) or has moved past it (a stale dispatch — drop the queue entry).
    """

    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    detail: str = ""
    still_live: bool = True


class SendPromptPayload(BaseModel):
    """master -> broker: relay a developer prompt into the session."""

    model_config = ConfigDict(extra="ignore")

    text: str


class DecisionLogPayload(BaseModel):
    """broker -> master reply payload carrying a rendered decision log."""

    model_config = ConfigDict(extra="ignore")

    text: str


class RetrievedSymbol(BaseModel):
    """One symbol grounding retrieval surfaced, for the developer to judge."""

    model_config = ConfigDict(extra="ignore")

    name: str  # path::Scope.name
    score: float | None = None  # cosine similarity for seeds; None for expansion nodes


class PromptProposalPayload(BaseModel):
    """broker -> master: grounded prompt awaiting developer approval."""

    model_config = ConfigDict(extra="ignore")

    proposal_id: str
    proposed_prompt: str
    grounding_summary: str
    retrieved: list[RetrievedSymbol] = Field(default_factory=list[RetrievedSymbol])


class StatusPayload(BaseModel):
    """broker -> master reply payload for the attach/liveness probe."""

    model_config = ConfigDict(extra="ignore")

    state: SessionState
    pane_id: str | None = None
    claude_session_id: str | None = None
    transcript_path: str | None = None
    permission_prompt: bool = False
    task_activity: str = ""
    pending_proposal: PromptProposalPayload | None = None


class ClarifyEscalationRequestPayload(BaseModel):
    """master -> broker: a read-only question about a live escalation."""

    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    question: str


class ClarifyEscalationReplyPayload(BaseModel):
    """broker -> master reply payload carrying a clarification answer."""

    model_config = ConfigDict(extra="ignore")

    answer: str


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
    escalation_title: str
    situation: str
    what_was_asked: str
    what_is_at_stake: str
    alternatives: list[Alternative]
    recommendation: str
    uncertainty: str
    what_would_change_my_mind: str


class PermissionEscalationPayload(BaseModel):
    """broker -> master: a tool call the developer must answer in the pane.

    Every field is required. A permission escalation the developer cannot act
    on — no tool, no pane, no reason — is worse than no escalation at all.
    """

    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    session_id: str
    tool_name: str
    tool_input: dict[str, Any]
    task_intent: str
    reason: str
    raised_at: str  # ISO timestamp; every resolution signal is dated against it
    permission_suggestions: list[PermissionSuggestion] = Field(
        default_factory=list[PermissionSuggestion]
    )


class PermissionLogPayload(BaseModel):
    """broker -> master reply payload carrying a rendered permission log."""

    model_config = ConfigDict(extra="ignore")

    text: str


class CompletionPayload(BaseModel):
    """broker -> master completion notice: split outcome for the modal and chat."""

    model_config = ConfigDict(extra="ignore")

    headline: str
    supporting: str


class FatalErrorPayload(BaseModel):
    """broker -> master fatal error."""

    model_config = ConfigDict(extra="ignore")

    error_class: str
    detail: str


class EscalationRetractPayload(BaseModel):
    """broker -> master: a decision escalation resolved out of band."""

    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    reason: str


class PermissionRetractPayload(BaseModel):
    """permission module -> master: a permission prompt that is no longer open."""

    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    reason: str


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


class LiveStatusPayload(BaseModel):
    """broker -> master: the broker's current live status, pushed on any change."""

    model_config = ConfigDict(extra="ignore")

    state: SessionState
    activity: str = ""  # active gerund phrases joined, "" when idle
    permission_prompt: bool = False
    task_activity: str = ""  # last per-turn task description; persists across turns
