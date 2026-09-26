"""Socket-protocol pydantic models. Brokers only — the hook must never import this.

Every model here is extra="ignore", never extra="forbid": an unknown wire field
means a newer peer, not a typo. Config models are the strict half of that pair.
"""

from collections.abc import Callable, Mapping
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from broker.protocol.constants import (
    T_APPROVE_PROMPT,
    T_ASK_QUESTION,
    T_BUDGET_UPDATE,
    T_CLARIFY_ESCALATION,
    T_COMPLETION,
    T_DECISION_DELIVERED,
    T_DECISION_UNDELIVERED,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_ESCALATION_RETRACT,
    T_FATAL_ERROR,
    T_GET_DECISION_LOG,
    T_GET_PERMISSION_LOG,
    T_HOOK_EVENT,
    T_LIVE_STATUS,
    T_PANE_ESCALATION,
    T_PANE_RETRACT,
    T_PERMISSION_REQUEST,
    T_PROMPT_PROPOSAL,
    T_PROMPT_UNDELIVERED,
    T_REACTIVATE,
    T_SEND_PROMPT,
    T_SESSION_ENDED,
    T_SHUTDOWN,
    T_STATUS,
    NackCode,
    PaneKind,
    SessionState,
)


class Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str  # uuid4 hex
    type: str
    session_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class Response(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str  # reuses request id
    type: Literal["response"] = "response"
    ok: bool
    payload: dict[str, Any] = Field(default_factory=dict)


class NackPayload(BaseModel):
    """The refusal a negative ``Response`` carries."""

    model_config = ConfigDict(extra="ignore")

    error: str
    reason_code: NackCode | None = None


def nack_response(env: Envelope, error: str, code: NackCode | None) -> Response:
    """Build the refusal answering ``env``.

    Args:
        env: Envelope being answered.
        error: Human-readable reason, carried for the developer.
        code: Machine-readable reason, or ``None`` when no code fits.

    Returns:
        The negative response.
    """
    return Response(
        id=env.id,
        ok=False,
        payload=NackPayload(error=error, reason_code=code).model_dump(
            exclude_none=True
        ),
    )


def parse_nack(resp: Response) -> NackPayload:
    """Read the refusal a negative response carries.

    Args:
        resp: A response with ``ok=False``.

    Returns:
        The refusal's error string and reason code.

    Raises:
        ValidationError: The payload is not a refusal.
    """
    return NackPayload.model_validate(resp.payload)


class WireMessage(BaseModel):
    """A request or notice payload, bound to the one message type it travels as."""

    model_config = ConfigDict(extra="ignore")

    MESSAGE_TYPE: ClassVar[str]


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


class PermissionRequestPayload(WireMessage):
    """hook -> broker; mirrors the PermissionRequest stdin payload."""

    MESSAGE_TYPE = T_PERMISSION_REQUEST
    model_config = ConfigDict(extra="ignore")

    tool_name: str
    tool_input: dict[str, Any]
    permission_suggestions: list[PermissionSuggestion] = Field(
        default_factory=list[PermissionSuggestion]
    )


class PermissionDecisionPayload(BaseModel):
    """broker -> hook reply payload."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["allow", "escalated"]


class AskQuestionRequestPayload(WireMessage):
    """hook -> broker: a pending AskUserQuestion awaiting a decision."""

    MESSAGE_TYPE = T_ASK_QUESTION
    model_config = ConfigDict(extra="ignore")

    tool_input: dict[str, Any]
    tool_use_id: str


class AskQuestionDecisionPayload(BaseModel):
    """broker -> hook reply for a pending AskUserQuestion."""

    model_config = ConfigDict(extra="ignore")

    decision: Literal["answer", "escalated"]
    updated_input: dict[str, Any] | None = None


class HookEventPayload(WireMessage):
    """Fire-and-forget wrapper for every hook event except PreToolUse and PermissionRequest."""

    MESSAGE_TYPE = T_HOOK_EVENT
    model_config = ConfigDict(extra="ignore")

    hook_event_name: str
    raw: dict[str, Any]  # full stdin payload, untyped by design


class DispatchDecisionPayload(WireMessage):
    """master -> broker: developer's resolution of an escalation."""

    MESSAGE_TYPE = T_DISPATCH_DECISION
    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    response: str


class DecisionDeliveredPayload(WireMessage):
    """broker -> master: a dispatched decision reached the pane.

    The master resolves the escalation only on this confirmation — the dispatch
    ACK means the broker accepted the decision for processing, not that it
    landed. Binding resolution to delivery keeps the two sides from diverging.
    """

    MESSAGE_TYPE = T_DECISION_DELIVERED
    model_config = ConfigDict(extra="ignore")

    escalation_id: str


class DecisionUndeliveredPayload(WireMessage):
    """broker -> master: a dispatched decision did not reach the pane.

    Sent when a pane write fails or the dispatch was stale. Since resolution
    waits for delivery, the escalation was never resolved: ``still_live`` says
    whether the broker still holds it (a failed write — keep it surfaced for a
    re-decide) or has moved past it (a stale dispatch — drop the queue entry).
    """

    MESSAGE_TYPE = T_DECISION_UNDELIVERED
    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    detail: str = ""
    still_live: bool = True


class SendPromptPayload(WireMessage):
    """master -> broker: relay a developer prompt into the session."""

    MESSAGE_TYPE = T_SEND_PROMPT
    model_config = ConfigDict(extra="ignore")

    text: str


class StatusRequestPayload(WireMessage):
    """master -> broker: the attach/liveness probe."""

    MESSAGE_TYPE = T_STATUS
    model_config = ConfigDict(extra="ignore")


class ShutdownPayload(WireMessage):
    """master -> broker: stop the session."""

    MESSAGE_TYPE = T_SHUTDOWN
    model_config = ConfigDict(extra="ignore")


class DecisionLogRequestPayload(WireMessage):
    """master -> broker: fetch the rendered decision log."""

    MESSAGE_TYPE = T_GET_DECISION_LOG
    model_config = ConfigDict(extra="ignore")


class DecisionLogPayload(BaseModel):
    """broker -> master reply payload carrying a rendered decision log."""

    model_config = ConfigDict(extra="ignore")

    text: str


class RetrievedSymbol(BaseModel):
    """One symbol grounding retrieval surfaced, for the developer to judge."""

    model_config = ConfigDict(extra="ignore")

    name: str  # path::Scope.name
    score: float | None = None  # cosine similarity for seeds; None for expansion nodes


class PromptProposalPayload(WireMessage):
    """broker -> master: grounded prompt awaiting developer approval."""

    MESSAGE_TYPE = T_PROMPT_PROPOSAL
    model_config = ConfigDict(extra="ignore")

    proposal_id: str
    proposed_prompt: str
    grounding_summary: str
    retrieved: list[RetrievedSymbol] = Field(default_factory=list[RetrievedSymbol])


class StatusPayload(BaseModel):
    """broker -> master reply payload for the attach/liveness probe."""

    model_config = ConfigDict(extra="ignore")

    state: SessionState
    permission_prompt: bool = False
    task_activity: str = ""
    pending_proposal: PromptProposalPayload | None = None


class ClarifyEscalationRequestPayload(WireMessage):
    """master -> broker: a read-only question about a live escalation."""

    MESSAGE_TYPE = T_CLARIFY_ESCALATION
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


class EscalationDisclosure(BaseModel):
    """The broker's analysis of a decision, as the developer reads it."""

    model_config = ConfigDict(extra="ignore")

    escalation_title: str
    situation: str
    what_was_asked: str
    what_is_at_stake: str
    alternatives: list[Alternative]
    recommendation: str
    uncertainty: str
    what_would_change_my_mind: str


class EscalationPayload(WireMessage):
    """broker -> master: a decision the developer answers in the master chat."""

    MESSAGE_TYPE = T_ESCALATION
    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    session_id: str
    task_context: str
    disclosure: EscalationDisclosure


class PermissionEscalationPayload(WireMessage):
    """broker -> master: a tool call the developer must answer in the pane.

    Every field is required. A permission escalation the developer cannot act
    on — no tool, no pane, no reason — is worse than no escalation at all.
    """

    MESSAGE_TYPE = T_PANE_ESCALATION
    model_config = ConfigDict(extra="ignore")

    kind: Literal[PaneKind.PERMISSION] = PaneKind.PERMISSION
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


class QuestionEscalationPayload(WireMessage):
    """broker -> master: an AskUserQuestion menu the developer answers in the pane."""

    MESSAGE_TYPE = T_PANE_ESCALATION
    model_config = ConfigDict(extra="ignore")

    kind: Literal[PaneKind.QUESTION] = PaneKind.QUESTION
    escalation_id: str
    session_id: str
    task_context: str
    # Both empty when the tool input was unusable.
    menu: str  # rendered by the broker, shown verbatim
    first_question: str
    reason: str  # why the broker did not answer the menu itself
    analysis: EscalationDisclosure | None = None


PaneEscalationPayload = Annotated[
    PermissionEscalationPayload | QuestionEscalationPayload,
    Field(discriminator="kind"),
]

PANE_ESCALATION_ADAPTER: TypeAdapter[PaneEscalationPayload] = TypeAdapter(
    PaneEscalationPayload
)


class PermissionLogRequestPayload(WireMessage):
    """master -> broker: fetch the rendered permission log."""

    MESSAGE_TYPE = T_GET_PERMISSION_LOG
    model_config = ConfigDict(extra="ignore")


class PermissionLogPayload(BaseModel):
    """broker -> master reply payload carrying a rendered permission log."""

    model_config = ConfigDict(extra="ignore")

    text: str


class CompletionPayload(WireMessage):
    """broker -> master completion notice: split outcome for the modal and chat."""

    MESSAGE_TYPE = T_COMPLETION
    model_config = ConfigDict(extra="ignore")

    headline: str
    supporting: str


class FatalErrorPayload(WireMessage):
    """broker -> master fatal error."""

    MESSAGE_TYPE = T_FATAL_ERROR
    model_config = ConfigDict(extra="ignore")

    error_class: str
    detail: str


class EscalationRetractPayload(WireMessage):
    """broker -> master: a decision escalation resolved out of band."""

    MESSAGE_TYPE = T_ESCALATION_RETRACT
    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    reason: str


class PaneRetractPayload(WireMessage):
    """raiser -> master: a pane escalation whose native prompt is no longer open."""

    MESSAGE_TYPE = T_PANE_RETRACT
    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    reason: str


class PromptUndeliveredPayload(WireMessage):
    """broker -> master: an accepted developer prompt that never reached the pane."""

    MESSAGE_TYPE = T_PROMPT_UNDELIVERED
    model_config = ConfigDict(extra="ignore")

    detail: str


class SessionEndedPayload(WireMessage):
    """broker -> master: SessionEnd fired and the broker is exiting."""

    MESSAGE_TYPE = T_SESSION_ENDED
    model_config = ConfigDict(extra="ignore")


class ApprovePromptPayload(WireMessage):
    """master -> broker: the developer-approved (possibly revised) prompt."""

    MESSAGE_TYPE = T_APPROVE_PROMPT
    model_config = ConfigDict(extra="ignore")

    proposal_id: str
    prompt: str


class ReactivatePayload(WireMessage):
    """master -> broker: a new task for a session that already completed one."""

    MESSAGE_TYPE = T_REACTIVATE
    model_config = ConfigDict(extra="ignore")

    intent: str


class BudgetUpdatePayload(WireMessage):
    """broker -> master: autonomous-answer counter for registry persistence."""

    MESSAGE_TYPE = T_BUDGET_UPDATE
    model_config = ConfigDict(extra="ignore")

    count: int


class LiveStatusPayload(WireMessage):
    """broker -> master: the broker's current live status, pushed on any change."""

    MESSAGE_TYPE = T_LIVE_STATUS
    model_config = ConfigDict(extra="ignore")

    state: SessionState
    activity: str = ""  # active gerund phrases joined, "" when idle
    permission_prompt: bool = False
    task_activity: str = ""  # last per-turn task description; persists across turns
    pane_id: str | None = None
    claude_session_id: str | None = None
    transcript_path: str | None = None


SESSION_SOCKET_PAYLOADS: Mapping[str, Callable[[Any], WireMessage]] = {
    model.MESSAGE_TYPE: model.model_validate
    for model in (
        HookEventPayload,
        PermissionRequestPayload,
        AskQuestionRequestPayload,
        DispatchDecisionPayload,
        SendPromptPayload,
        StatusRequestPayload,
        DecisionLogRequestPayload,
        PermissionLogRequestPayload,
        ShutdownPayload,
        ApprovePromptPayload,
        ReactivatePayload,
        ClarifyEscalationRequestPayload,
    )
}

MASTER_SOCKET_PAYLOADS: Mapping[str, Callable[[Any], WireMessage]] = {
    **{
        model.MESSAGE_TYPE: model.model_validate
        for model in (
            EscalationPayload,
            CompletionPayload,
            FatalErrorPayload,
            EscalationRetractPayload,
            PaneRetractPayload,
            PromptProposalPayload,
            BudgetUpdatePayload,
            DecisionDeliveredPayload,
            DecisionUndeliveredPayload,
            LiveStatusPayload,
            SessionEndedPayload,
            PromptUndeliveredPayload,
        )
    },
    T_PANE_ESCALATION: PANE_ESCALATION_ADAPTER.validate_python,
}
