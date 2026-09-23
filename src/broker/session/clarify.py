"""Clarify: answer one read-only question about a live escalation.

Reuses the session broker's own LLM seam and triage's context assembly — one
forced-tool call, exactly like triage. Never resolves the escalation and never
writes to the pane.
"""

import logging

from pydantic import BaseModel, ConfigDict, ValidationError

from broker import llm_timing
from broker import prompts
from broker.config import BrokerConfig
from broker.llm import LLMCaller, LLMCallError, ToolCall, strict_tool
from broker.protocol.schemas import EscalationPayload
from broker.session.triage import FORCED_ONE, assemble_context
from broker.transcript.schemas import TranscriptEvent

logger = logging.getLogger(__name__)


class ClarifyCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    answer: str


CLARIFY_TOOLS = [
    strict_tool(
        "answer_clarification",
        "Answer the developer's question about the escalation, from the "
        "escalation and the session transcript only. If the record does not "
        "contain the answer, say so plainly rather than inferring.",
        ClarifyCall,
    )
]

_CLARIFY_PROMPT = prompts.load("clarify")


def render_disclosure(p: EscalationPayload) -> str:
    """Render a raised escalation as prompt text for the clarify call.

    Args:
        p: The escalation the broker raised and still holds.

    Returns:
        The escalation's analysis as a plain block. Omits the id/session
        header (the broker knows both), task_context (carried as intent) and
        the title.
    """
    lines = [
        "## Situation",
        p.situation,
        "",
        "## What was asked",
        p.what_was_asked,
        "",
        "## What is at stake",
        p.what_is_at_stake,
        "",
        "## Alternatives",
    ]
    for alt in p.alternatives:
        lines += [f"- {alt.option}", f"  pros: {alt.pros}", f"  cons: {alt.cons}"]
    lines += [
        "",
        "## Recommendation",
        p.recommendation,
        "",
        "## Uncertainty",
        p.uncertainty,
        "",
        "## What would change my mind",
        p.what_would_change_my_mind,
    ]
    return "\n".join(lines)


@llm_timing.timed("clarify")
async def clarify(
    llm_call: LLMCaller[ToolCall],
    cfg: BrokerConfig,
    *,
    intent: str,
    escalation: EscalationPayload,
    question: str,
    events: list[TranscriptEvent],
) -> ClarifyCall:
    """Answer one read-only question about a live escalation.

    Args:
        llm_call: The broker's own injected tool-calling seam.
        cfg: Supplies the pinned model id and token cap.
        intent: Authoritative task intent, from the registry.
        escalation: The escalation the question is about.
        question: The developer's question, verbatim.
        events: Cleaned transcript events, as surrounding context.

    Returns:
        The validated answer call.

    Raises:
        LLMCallError: The model called the wrong tool, or its input failed
            validation.
    """
    working = (
        "# The escalation the developer is asking about\n"
        + render_disclosure(escalation)
        + "\n\n# The developer's question (answer THIS)\n"
        + question
    )
    system, messages = assemble_context(_CLARIFY_PROMPT, intent, events, working)
    call: ToolCall = await llm_call(
        model=cfg.model_id,
        max_tokens=cfg.max_tokens,
        system=system,
        messages=messages,
        tools=CLARIFY_TOOLS,
        tool_choice=FORCED_ONE,
    )
    if call.name != "answer_clarification":
        raise LLMCallError(f"unknown clarify tool {call.name!r}")
    try:
        return ClarifyCall.model_validate(call.input)
    except ValidationError as exc:
        raise LLMCallError(f"invalid answer_clarification input: {exc}") from exc
