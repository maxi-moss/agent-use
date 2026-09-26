"""Clarify: answer one read-only question about a live escalation.

Reuses the session broker's own LLM seam and the session stack's context
assembly — one forced-tool call, exactly like triage. Never resolves the
escalation and never writes to the pane.

Class names, field names, `Field` descriptions and docstrings of these models
are sent to the model.
"""

import logging

from pydantic import BaseModel, ConfigDict

from broker import llm_timing
from broker import prompts
from broker.config import SessionModelConfig
from broker.llm import LLMCaller, ToolCall, strict_tool
from broker.protocol.schemas import EscalationDisclosure, EscalationPayload
from broker.session.llm_stack import assemble_context, forced_call
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

_TOOL_MODELS: dict[str, type[ClarifyCall]] = {"answer_clarification": ClarifyCall}

_CLARIFY_PROMPT = prompts.load("clarify")


def render_disclosure(p: EscalationDisclosure) -> str:
    """Render a raised escalation's disclosure as prompt text for the clarify call.

    Args:
        p: The disclosure of the escalation the broker raised and still holds.

    Returns:
        The analysis as a plain block. Omits the title.
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
    model_cfg: SessionModelConfig,
    *,
    intent: str,
    escalation: EscalationPayload,
    question: str,
    events: list[TranscriptEvent],
) -> ClarifyCall:
    """Answer one read-only question about a live escalation.

    Args:
        llm_call: The broker's own injected tool-calling seam.
        model_cfg: Supplies the pinned model id and token cap.
        intent: Authoritative task intent, from the registry.
        escalation: The escalation the question is about.
        question: The developer's question, verbatim.
        events: Cleaned transcript events, as surrounding context.

    Returns:
        The validated answer call.

    Raises:
        LLMCallError: The model called the wrong tool, or its input failed
            validation. Uncaught here; the caller NACKs the developer's
            clarify request rather than resolving it.
    """
    working = (
        "# The escalation the developer is asking about\n"
        + render_disclosure(escalation.disclosure)
        + "\n\n# The developer's question (answer THIS)\n"
        + question
    )
    system, messages = assemble_context(_CLARIFY_PROMPT, intent, events, working)
    return await forced_call(
        llm_call,
        model_cfg,
        system=system,
        messages=messages,
        tools=CLARIFY_TOOLS,
        models=_TOOL_MODELS,
        label="clarify",
    )
