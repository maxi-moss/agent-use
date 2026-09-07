"""Triage: four strict tools, context assembly, one LLM call per turn boundary.

Pydantic models are the single source of truth for tool inputs; JSON schemas
are DERIVED (model_json_schema) then strictified. The escalate tool's field
set is pinned against EscalationPayload by a unit test so the wire schema and
the tool schema cannot drift apart.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from broker import llm_timing
from broker import prompts
from broker.config import BrokerConfig
from broker.index.render import fit_to_budget, render_relevant_code
from broker.index.schemas import GroundingContext
from broker.llm import LLMCaller, LLMCallError, ToolCall, strict_tool
from broker.protocol.schemas import Alternative
from broker.transcript.adapter import render
from broker.transcript.schemas import TranscriptEvent

logger = logging.getLogger(__name__)


class AnswerCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    answer: str


class EscalateCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    situation: str
    what_was_asked: str
    what_is_at_stake: str
    alternatives: list[Alternative]
    recommendation: str
    uncertainty: str
    what_would_change_my_mind: str


class CompleteCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    summary: str


class NoActionCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str


class ProposePromptCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    prompt: str


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
        "The task is finished. `summary` is shown to the developer verbatim.",
        CompleteCall,
    ),
    _tool(
        "no_action",
        "Nothing needs doing at this turn boundary.",
        NoActionCall,
    ),
]

GROUNDING_TOOLS: list[ToolParam] = [
    _tool(
        "propose_prompt",
        "Propose the initial task prompt for the coding session. The developer"
        " reviews and may revise it before submission.",
        ProposePromptCall,
    ),
]

_TOOL_MODELS: dict[str, type[TriageResult]] = {
    "answer": AnswerCall,
    "escalate": EscalateCall,
    "complete": CompleteCall,
    "no_action": NoActionCall,
}

FORCED_ONE: ToolChoiceParam = {"type": "any", "disable_parallel_tool_use": True}

_TRIAGE_PROMPT = prompts.load("triage")
_GROUNDING_PROMPT = prompts.load("grounding")


def assemble_context(
    system_prompt: str,
    intent: str,
    transcript_events: list[TranscriptEvent],
    working: str,
) -> tuple[list[TextBlockParam], list[MessageParam]]:
    """Assemble the system blocks and messages for one session-stack call.

    Args:
        system_prompt: Static system prompt text; cached with a 1h TTL.
        intent: Authoritative task intent, taken from the registry.
        transcript_events: Cleaned transcript events; an empty render becomes
            ``(transcript empty)``.
        working: Turn-specific text, placed last and left uncached.

    Returns:
        The system blocks and the single user message, ready to pass to the
        LLM.
    """
    rendered = render(transcript_events) or "(transcript empty)"
    system: list[TextBlockParam] = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "# Authoritative task intent\n" + intent,
                },
                {
                    "type": "text",
                    "text": rendered,
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                },
                {"type": "text", "text": working},
            ],
        }
    ]
    return system, messages


@llm_timing.timed("triage")
async def triage(
    llm_call: LLMCaller[ToolCall],
    cfg: BrokerConfig,
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
        cfg: Supplies the model id and the token cap.
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
        model=cfg.model_id,
        max_tokens=cfg.max_tokens,
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


class Retriever(Protocol):
    """The injected retrieval seam: tests pass a fake, production binds the index."""

    async def __call__(self, intent: str, cwd: Path) -> GroundingContext:
        """Return the intent's code neighbourhood for the repository at ``cwd``."""
        ...


@dataclass
class Grounding:
    """A proposed prompt with the code neighbourhood it was grounded in."""

    proposal: ProposePromptCall
    context: GroundingContext


@llm_timing.timed("retrieval")
async def _retrieve(retrieve: Retriever, intent: str, cwd: Path) -> GroundingContext:
    """Retrieve and trim the neighbourhood so the rendered block fits its budget."""
    return fit_to_budget(await retrieve(intent, cwd))


@llm_timing.timed("grounding")
async def _propose(
    llm_call: LLMCaller[ToolCall], cfg: BrokerConfig, parts: list[str]
) -> ProposePromptCall:
    """Make the one grounding call and validate its forced tool use.

    Raises:
        LLMCallError: The LLM called a tool other than ``propose_prompt``, or
            the tool input failed validation.
    """
    system: list[TextBlockParam] = [{"type": "text", "text": _GROUNDING_PROMPT}]
    messages: list[MessageParam] = [{"role": "user", "content": "\n\n".join(parts)}]
    call: ToolCall = await llm_call(
        model=cfg.model_id,
        max_tokens=cfg.max_tokens,
        system=system,
        messages=messages,
        tools=GROUNDING_TOOLS,
        tool_choice=FORCED_ONE,
    )
    if call.name != "propose_prompt":
        raise LLMCallError(f"unknown grounding tool {call.name!r}")
    try:
        return ProposePromptCall.model_validate(call.input)
    except ValidationError as exc:
        raise LLMCallError(f"invalid propose_prompt input: {exc}") from exc


async def ground_intent(
    llm_call: LLMCaller[ToolCall],
    cfg: BrokerConfig,
    *,
    retrieve: Retriever,
    intent: str,
    cwd: Path,
) -> Grounding:
    """Propose the initial task prompt for a session from the stated intent.

    Args:
        llm_call: The injected tool-calling seam.
        cfg: Supplies the model id and the token cap.
        retrieve: The injected code-retrieval seam.
        intent: The developer's intent, passed verbatim.
        cwd: Session working directory: the repository root.

    Returns:
        The validated ``propose_prompt`` call and the neighbourhood it saw.

    Raises:
        LLMCallError: The grounding call failed or returned the wrong tool.
        Exception: Whatever ``retrieve`` raises — retrieval failures abort
            grounding; there is no degraded path.
    """
    context = await _retrieve(retrieve, intent, cwd)
    relevant_code = render_relevant_code(context)
    logger.info("relevant code for grounding in %s:\n%s", cwd, relevant_code)
    claude_md = ""
    claude_md_path = cwd / "CLAUDE.md"
    if claude_md_path.exists():
        claude_md = claude_md_path.read_text(encoding="utf-8")
    parts = [f"# Developer intent (verbatim)\n{intent}"]
    if claude_md:
        parts.append(f"# The codebase's CLAUDE.md\n{claude_md}")
    parts.append(relevant_code)
    proposal = await _propose(llm_call, cfg, parts)
    return Grounding(proposal=proposal, context=context)
