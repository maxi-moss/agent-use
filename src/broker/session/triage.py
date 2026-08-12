"""Triage: four strict tools, context assembly, one LLM call per turn boundary.

Pydantic models are the single source of truth for tool inputs; JSON schemas
are DERIVED (model_json_schema) then strictified. The escalate tool's field
set is pinned against EscalationPayload by a unit test so the wire schema and
the tool schema cannot drift apart.
"""

import asyncio
import subprocess
from pathlib import Path

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from broker import diagnostics
from broker import prompts
from broker.config import BrokerConfig
from broker.llm import LLMCaller, LLMCallError, ToolCall, strict_tool
from broker.protocol.schemas import Alternative
from broker.transcript.adapter import render
from broker.transcript.schemas import TranscriptEvent

_TRACKED_FILES_TIMEOUT_S = 10.0
_MAX_TRACKED_FILES = 200


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
    intent: str, transcript_events: list[TranscriptEvent], working: str
) -> tuple[list[TextBlockParam], list[MessageParam]]:
    """Assemble the system blocks and messages for one triage call.

    Args:
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
            "text": _TRIAGE_PROMPT,
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


@diagnostics.timed("triage")
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
    system, messages = assemble_context(intent, events, working)
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


def _git_ls_files(cwd: Path) -> str:
    """List the repository's tracked files, truncated to a fixed maximum.

    Args:
        cwd: Directory to run ``git ls-files`` in.

    Returns:
        Newline-joined paths, with a trailing count line when the listing was
        truncated; empty when git produced nothing usable.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TRACKED_FILES_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    lines = proc.stdout.splitlines()
    head = lines[:_MAX_TRACKED_FILES]
    if len(lines) > _MAX_TRACKED_FILES:
        extra = len(lines) - _MAX_TRACKED_FILES
        head.append(f"... ({extra} more tracked files)")
    return "\n".join(head)


@diagnostics.timed("grounding")
async def ground_intent(
    llm_call: LLMCaller[ToolCall],
    cfg: BrokerConfig,
    *,
    intent: str,
    cwd: Path,
) -> ProposePromptCall:
    """Propose the initial task prompt for a session from the stated intent.

    Args:
        llm_call: The injected tool-calling seam.
        cfg: Supplies the model id and the token cap.
        intent: The developer's intent, passed verbatim.
        cwd: Session working directory, read for grounding context.

    Returns:
        The validated ``propose_prompt`` call, for the developer to review.

    Raises:
        LLMCallError: The LLM called a tool other than ``propose_prompt``, or
            the tool input failed validation.
    """
    claude_md = ""
    claude_md_path = cwd / "CLAUDE.md"
    if claude_md_path.exists():
        claude_md = claude_md_path.read_text(encoding="utf-8")
    listing = await asyncio.to_thread(_git_ls_files, cwd)
    parts = [f"# Developer intent (verbatim)\n{intent}"]
    if claude_md:
        parts.append(f"# The codebase's CLAUDE.md\n{claude_md}")
    if listing:
        parts.append(f"# Tracked files (head)\n{listing}")
    system: list[TextBlockParam] = [
        {"type": "text", "text": _GROUNDING_PROMPT}
    ]
    messages: list[MessageParam] = [
        {"role": "user", "content": "\n\n".join(parts)}
    ]
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
