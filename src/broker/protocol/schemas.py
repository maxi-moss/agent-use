"""Socket-protocol pydantic models. Brokers only — the hook must never import this."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from broker.protocol.constants import PROTOCOL_VERSION


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
    """master -> broker: developer's resolution of an escalation (spec §9.7 step 6)."""

    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    response: str


class SendPromptPayload(BaseModel):
    """master -> broker: relay a developer prompt into the session (spec §8.4)."""

    model_config = ConfigDict(extra="ignore")

    text: str


class StatusPayload(BaseModel):
    """broker -> master reply payload for the attach/liveness probe (spec §9.11)."""

    model_config = ConfigDict(extra="ignore")

    state: str


class Alternative(BaseModel):
    model_config = ConfigDict(extra="ignore")

    option: str
    pros: str
    cons: str


class EscalationPayload(BaseModel):
    """broker -> master structured escalation object (spec §9.7)."""

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
    """broker -> master completion notice with summary (spec §9.8)."""

    model_config = ConfigDict(extra="ignore")

    summary: str


class FatalErrorPayload(BaseModel):
    """broker -> master fatal error (spec §9.10)."""

    model_config = ConfigDict(extra="ignore")

    error_class: str
    detail: str


class RetractPayload(BaseModel):
    """broker -> master: escalation resolved out of band (spec §8.6)."""

    model_config = ConfigDict(extra="ignore")

    escalation_id: str
    reason: str
