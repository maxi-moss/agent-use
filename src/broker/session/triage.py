"""Triage: four strict tools, one LLM call per turn boundary.

Pydantic models are the single source of truth for tool inputs; JSON schemas
are DERIVED (model_json_schema) then strictified.

Class names, field names, `Field` descriptions and docstrings of these models
are sent to the model.
"""

from anthropic.types import ToolParam
from pydantic import BaseModel, ConfigDict, Field

from broker import llm_timing
from broker import prompts
from broker.config import SessionModelConfig
from broker.llm import LLMCaller, ToolCall, strict_tool
from broker.session.llm_stack import (
    EscalateCall,
    HasTaskSummary,
    assemble_context,
    forced_call,
)
from broker.transcript.schemas import TranscriptEvent


class _HasTaskActivity(BaseModel):
    task_activity: str


class AnswerCall(_HasTaskActivity, HasTaskSummary):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    answer: str


class CompleteCall(_HasTaskActivity, HasTaskSummary):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    headline: str = Field(
        description=(
            "The one-line outcome shown to the developer, e.g. 'Recovered the"
            " session after broker loss'."
        )
    )
    supporting: str = Field(
        description="One sentence of the key evidence behind the headline."
    )


class NoActionCall(_HasTaskActivity):
    model_config = ConfigDict(extra="forbid")

    reasoning: str


TriageResult = AnswerCall | EscalateCall | CompleteCall | NoActionCall


TRIAGE_TOOLS: list[ToolParam] = [
    strict_tool(
        "answer",
        "Answer the coding agent yourself. The `answer` text is typed into the"
        " session verbatim as an instruction. Use when a well-supported answer"
        " follows from the stated intent and the conversation.",
        AnswerCall,
    ),
    strict_tool(
        "escalate",
        "Raise the decision to the developer. Use when the situation is"
        " irreversible or high blast-radius, architecturally significant, or"
        " you cannot ground an answer. Every field must carry real analysis.",
        EscalateCall,
    ),
    strict_tool(
        "complete",
        "The task is finished. `headline` and `supporting` are shown to the"
        " developer as the outcome; `task_summary` is its final history line.",
        CompleteCall,
    ),
    strict_tool(
        "no_action",
        "Nothing needs doing at this turn boundary.",
        NoActionCall,
    ),
]

_TOOL_MODELS: dict[str, type[TriageResult]] = {
    "answer": AnswerCall,
    "escalate": EscalateCall,
    "complete": CompleteCall,
    "no_action": NoActionCall,
}

_TRIAGE_PROMPT = prompts.load("triage")


def triage_working_text(last_assistant_message: str) -> str:
    """Build the working block for a triage call: the message classified THIS turn."""
    return (
        "# What just happened\nhook event: Stop\n\n"
        "# The coding agent's last message (triage THIS)\n"
        + last_assistant_message
    )


@llm_timing.timed("triage")
async def triage(
    llm_call: LLMCaller[ToolCall],
    model_cfg: SessionModelConfig,
    *,
    intent: str,
    events: list[TranscriptEvent],
    last_assistant_message: str,
) -> TriageResult:
    """Classify one turn boundary into exactly one triage tool call.

    The classification input is ``last_assistant_message``; the transcript is
    surrounding context only. One tool call is forced, and its input is
    validated against the pydantic model backing that tool.

    Args:
        llm_call: The injected tool-calling seam.
        model_cfg: Supplies the model id and the token cap.
        intent: Authoritative task intent, taken from the registry.
        events: Transcript events, as surrounding context.
        last_assistant_message: The coding agent's last message — the text
            being classified.

    Returns:
        The validated call model for the tool the LLM chose.

    Raises:
        LLMCallError: The LLM called an unknown tool, or the tool input
            failed validation. Uncaught here; the caller treats this turn
            boundary as a fatal session error.
    """
    working = triage_working_text(last_assistant_message)
    system, messages = assemble_context(_TRIAGE_PROMPT, intent, events, working)
    return await forced_call(
        llm_call,
        model_cfg,
        system=system,
        messages=messages,
        tools=TRIAGE_TOOLS,
        models=_TOOL_MODELS,
        label="triage",
    )
