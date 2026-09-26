"""The ONLY renderers of broker payloads.

The runtime/LLM split is load-bearing: everything the developer reads is
rendered HERE, verbatim, and handed to the TUI (and to the LLM layer as an
opaque block). Re-summarising happens nowhere — structurally.
"""

import json

from broker.protocol.schemas import (
    EscalationDisclosure,
    EscalationPayload,
    PaneEscalationPayload,
    PermissionEscalationPayload,
    PermissionSuggestion,
    PromptProposalPayload,
    QuestionEscalationPayload,
    RetrievedSymbol,
)

# Stands in for a pane the registry cannot name. A pane escalation is still
# worth surfacing without it: the developer knows the session.
PANE_UNKNOWN = "(pane unknown)"


def _render_disclosure_sections(d: EscalationDisclosure) -> list[str]:
    """Render a broker's disclosure as block lines, verbatim."""
    lines = [
        "## Title",
        d.escalation_title,
        "",
        "## Situation",
        d.situation,
        "",
        "## What was asked",
        d.what_was_asked,
        "",
        "## What is at stake",
        d.what_is_at_stake,
        "",
        "## Alternatives",
    ]
    for alt in d.alternatives:
        lines += [
            f"- {alt.option}",
            f"  pros: {alt.pros}",
            f"  cons: {alt.cons}",
        ]
    lines += [
        "",
        "## Recommendation",
        d.recommendation,
        "",
        "## Uncertainty",
        d.uncertainty,
        "",
        "## What would change my mind",
        d.what_would_change_my_mind,
    ]
    return lines


def render_escalation(p: EscalationPayload) -> str:
    """Render the decision-escalation block, deterministic and verbatim.

    Args:
        p: Validated escalation payload from a session broker.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    lines = [
        f"Escalation {p.escalation_id} — session {p.session_id}",
        "",
        "## Task context",
        p.task_context,
        "",
        *_render_disclosure_sections(p.disclosure),
    ]
    return "\n".join(lines)


def _render_suggestion(suggestion: PermissionSuggestion) -> str:
    """Render one of Claude Code's permission suggestions as its raw object."""
    data = suggestion if isinstance(suggestion, dict) else suggestion.model_dump()
    return json.dumps(data, sort_keys=True)


def render_permission_escalation(
    p: PermissionEscalationPayload, pane_id: str
) -> str:
    """Render the permission-escalation block, deterministic and verbatim.

    Args:
        p: Validated permission-escalation payload from a session broker.
        pane_id: Pane holding the native prompt, or ``PANE_UNKNOWN``.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    lines = [
        f"Permission escalation {p.escalation_id} — session {p.session_id}",
        "",
        f"The developer answers this in pane {pane_id}, on the native "
        "permission prompt already waiting there. It cannot be answered "
        "here, and no decision sent from here reaches it.",
        "",
        "## Tool",
        p.tool_name,
        "",
        "## Tool input",
        json.dumps(p.tool_input, indent=2, sort_keys=True),
        "",
        "## Why it was escalated",
        p.reason,
        "",
        "## Task intent it was judged against",
        p.task_intent,
        "",
        "## Permission suggestions",
    ]
    if p.permission_suggestions:
        lines += [
            f"- {_render_suggestion(s)}" for s in p.permission_suggestions
        ]
    else:
        lines.append("(none)")
    return "\n".join(lines)


def render_question_escalation(p: QuestionEscalationPayload, pane_id: str) -> str:
    """Render the question-escalation block, deterministic and verbatim.

    Args:
        p: Validated question-escalation payload from a session broker.
        pane_id: Pane holding the AskUserQuestion menu, or ``PANE_UNKNOWN``.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    lines = [
        f"Question escalation {p.escalation_id} — session {p.session_id}",
        "",
        f"The developer answers this in pane {pane_id}, on the AskUserQuestion "
        "menu already waiting there. It cannot be answered here, and no "
        "decision sent from here reaches it.",
        "",
        "## Task context",
        p.task_context,
        "",
        "## Menu",
    ]
    lines += [
        p.menu or "(the menu could not be read — see the pane)",
        "",
        "## Why the broker did not answer",
        p.reason,
    ]
    if p.analysis is not None:
        lines += ["", *_render_disclosure_sections(p.analysis)]
    return "\n".join(lines)


def render_pane_escalation(p: PaneEscalationPayload, pane_id: str) -> str:
    """Render a pane-escalation block of either kind, deterministic and verbatim.

    Args:
        p: Validated pane-escalation payload.
        pane_id: Pane holding the native prompt, or ``PANE_UNKNOWN``.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    if isinstance(p, PermissionEscalationPayload):
        return render_permission_escalation(p, pane_id)
    return render_question_escalation(p, pane_id)


def pane_label(p: PaneEscalationPayload) -> str:
    """Name a pane escalation in one line: the tool, or the menu's first question."""
    if isinstance(p, PermissionEscalationPayload):
        return p.tool_name
    return p.first_question or "(unreadable menu)"


def render_proposal(p: PromptProposalPayload) -> str:
    """Render the proposal block: prompt, grounding, and retrieved code, verbatim.

    Args:
        p: Validated prompt-proposal payload from a session broker.

    Returns:
        The rendered block, to be displayed and passed on unchanged.
    """
    lines = [
        f"Prompt proposal {p.proposal_id}",
        "",
        "## Proposed prompt",
        p.proposed_prompt,
        "",
        "## Grounding summary",
        p.grounding_summary,
        "",
        "## Retrieved code",
    ]
    if p.retrieved:
        lines += [_render_retrieved(s) for s in p.retrieved]
    else:
        lines.append("(none)")
    return "\n".join(lines)


def _render_retrieved(s: RetrievedSymbol) -> str:
    """Render one retrieved symbol as a bullet line."""
    if s.score is None:
        return f"- {s.name}"
    return f"- {s.name} (seed {s.score:.2f})"
