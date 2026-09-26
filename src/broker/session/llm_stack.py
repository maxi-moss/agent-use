"""The session stack's kernel: context assembly, the escalate tool, the client seam.

Triage, ask and clarify each make their own forced-tool call on top of this.
The escalate tool's field set is pinned against EscalationDisclosure by a unit
test so the wire schema and the tool schema cannot drift apart.

Class names, field names, `Field` descriptions and docstrings of these models
are sent to the model.
"""

import functools

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlockParam, ToolChoiceParam
from pydantic import BaseModel, ConfigDict, Field

from broker.llm import LLMCaller, ToolCall, call_tool
from broker.protocol.schemas import Alternative, EscalationDisclosure
from broker.transcript.adapter import render
from broker.transcript.schemas import TranscriptEvent


class HasTaskSummary(BaseModel):
    task_summary: str = Field(
        description=(
            "One short past-tense line for the completed-outcome history: what"
            " you did this turn and why it mattered, e.g. 'Checked restart"
            " behavior'. Developer-facing narrative, not the raw instruction."
        )
    )


class EscalateCall(HasTaskSummary):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    escalation_title: str = Field(
        description=(
            "A short noun-phrase title naming the decision, e.g. 'Queue"
            " replacement policy'. Shown to the developer as the escalation's"
            " one-line label."
        )
    )
    situation: str = Field(
        description=(
            "What is happening in the session right now that led to this"
            " decision: what the coding agent was doing and what it ran into."
        )
    )
    what_was_asked: str = Field(
        description=(
            "The question or questions the developer must answer. When the"
            " coding agent asked them, copy its questions verbatim."
        )
    )
    what_is_at_stake: str = Field(
        description="What goes wrong, and how badly, if the decision is wrong."
    )
    alternatives: list[Alternative] = Field(
        description=(
            "The genuine options open to the developer, each with its pros and"
            " cons."
        )
    )
    recommendation: str = Field(
        description="The option you would choose, and why."
    )
    uncertainty: str = Field(
        description="What you are unsure about in this analysis or recommendation."
    )
    what_would_change_my_mind: str = Field(
        description=(
            "What fact or evidence would make you recommend a different option."
        )
    )


def disclosure_of(call: EscalateCall) -> EscalationDisclosure:
    """Return the developer-facing disclosure an escalate call carries."""
    return EscalationDisclosure(
        escalation_title=call.escalation_title,
        situation=call.situation,
        what_was_asked=call.what_was_asked,
        what_is_at_stake=call.what_is_at_stake,
        alternatives=call.alternatives,
        recommendation=call.recommendation,
        uncertainty=call.uncertainty,
        what_would_change_my_mind=call.what_would_change_my_mind,
    )


FORCED_ONE: ToolChoiceParam = {"type": "any", "disable_parallel_tool_use": True}


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


def bind_call_tool(client: AsyncAnthropic) -> LLMCaller[ToolCall]:
    """Bind an Anthropic client into the session stack's tool-calling seam.

    Args:
        client: Anthropic client every session-stack call goes through.

    Returns:
        A callable that forwards each forced-tool call to that one client.
    """
    return functools.partial(call_tool, client)
