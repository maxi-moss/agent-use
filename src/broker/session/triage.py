"""Triage: four strict tools, one LLM call per turn boundary.

Pydantic models are the single source of truth for tool inputs; JSON schemas
are DERIVED (model_json_schema) then strictified.

Class names, field names, `Field` descriptions and docstrings of these models
are sent to the model.
"""

from anthropic.types import ToolParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from broker import llm_timing
from broker import prompts
from broker.config import SessionModelConfig
from broker.llm import LLMCaller, LLMCallError, ToolCall, strict_tool
from broker.session.llm_stack import (
    FORCED_ONE,
    EscalateCall,
    HasTaskSummary,
    assemble_context,
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


_tool = strict_tool  # derivation lives in broker.llm (neutral module)


TRIAGE_TOOLS: list[ToolParam] = [
    _tool(
        "answer",
        "Answer the coding agent yourself. The `answer` text is typed into the"
        " session verbatim as an instruction. Use when a well-supported answer"
        " follows from the stated intent and the conversation.",
        AnswerCall,
    ),
    _tool(
        "escalate",
        "Raise the decision to the developer. Use when the situation is"
        " irreversible or high blast-radius, architecturally significant, or"
        " you cannot ground an answer. Every field must carry real analysis.",
        EscalateCall,
    ),
    _tool(
        "complete",
        "The task is finished. `headline` and `supporting` are shown to the"
        " developer as the outcome; `task_summary` is its final history line.",
        CompleteCall,
    ),
    _tool(
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


@llm_timing.timed("triage")
async def triage(
    llm_call: LLMCaller[ToolCall],
    model_cfg: SessionModelConfig,
    *,
    intent: str,
    events: list[TranscriptEvent],
    event_name: str,
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
        event_name: The hook event that opened this turn boundary.
        last_assistant_message: The coding agent's last message — the text
            being classified.

    Returns:
        The validated call model for the tool the LLM chose.

    Raises:
        LLMCallError: The LLM called an unknown tool, or the tool input
            failed validation.
    """
    working = (
        f"# What just happened\nhook event: {event_name}\n\n"
        "# The coding agent's last message (triage THIS)\n"
        + last_assistant_message
    )
    system, messages = assemble_context(_TRIAGE_PROMPT, intent, events, working)
    call: ToolCall = await llm_call(
        model=model_cfg.model_id,
        max_tokens=model_cfg.max_tokens,
        system=system,
        messages=messages,
        tools=TRIAGE_TOOLS,
        tool_choice=FORCED_ONE,
    )
    model = _TOOL_MODELS.get(call.name)
    if model is None:
        raise LLMCallError(f"unknown triage tool {call.name!r}")
    try:
        return model.model_validate(call.input)
    except ValidationError as exc:
        raise LLMCallError(f"invalid {call.name} input: {exc}") from exc
