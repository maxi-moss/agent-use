"""Anthropic client discipline for the permission classifier, standalone.

The strict-tool derivation and the client construction here duplicate the
shapes the session broker uses, deliberately and without sharing them: the
classifier runs on its own small model, and a shared helper is exactly how that
model would get quietly re-pinned to the session's.

Rules encoded here:
- AsyncAnthropic(max_retries=0) — the SDK default of 2 silently retries.
- Catch the ROOT anthropic.AnthropicError once; APITimeoutError subclasses
  APIConnectionError, so a discriminating chain mis-sorts timeouts.
- stop_reason "refusal" arrives as HTTP 200 with empty/partial content — it is
  checked BEFORE content is touched.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, cast

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)
from pydantic import BaseModel, ValidationError

from broker import prompts
from broker.config import ClassifierConfig
from broker.permission.schemas import AllowCall, EscalateCall, PermissionResult
from broker.protocol.schemas import PermissionSuggestion


class PermissionCallError(Exception):
    """Any classifier-call failure. One class: every failure escalates."""


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]


class PermissionCaller(Protocol):
    """The one injected seam: tests pass a fake, production binds a client."""

    async def __call__(
        self,
        *,
        model: str,
        max_tokens: int,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        tools: list[ToolParam],
        tool_choice: ToolChoiceParam,
    ) -> ToolCall:
        """Make one call against the model and return its tool call."""
        ...


def strictify(schema: dict[str, Any]) -> dict[str, Any]:
    """Enforce strict-tool constraints on a derived JSON schema, in place.

    Args:
        schema: A schema derived from a pydantic model. Mutated in place.

    Returns:
        The same schema object, for call-site convenience.
    """

    def walk(node: Any) -> None:
        """Tighten every object schema reachable from ``node``, recursively."""
        if isinstance(node, dict):
            typed = cast(dict[str, Any], node)
            if typed.get("type") == "object" or "properties" in typed:
                props_raw = typed.get("properties")
                props = (
                    cast(dict[str, Any], props_raw)
                    if isinstance(props_raw, dict)
                    else {}
                )
                typed["additionalProperties"] = False
                typed["required"] = list(props.keys())
            for value in typed.values():
                walk(value)
        elif isinstance(node, list):
            for item in cast(list[Any], node):
                walk(item)

    walk(schema)
    return schema


def strict_tool(
    name: str, description: str, model: type[BaseModel]
) -> ToolParam:
    """Build a strict tool definition from a pydantic model.

    Args:
        name: Tool name the model will call.
        description: Tool description sent to the model.
        model: Model whose JSON schema becomes the tool's input schema, after
            ``strictify``.

    Returns:
        The tool parameter block, with ``strict`` set.
    """
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": strictify(model.model_json_schema()),
    }


PERMISSION_TOOLS: list[ToolParam] = [
    strict_tool(
        "allow",
        "Let the tool call execute without interrupting the developer. Use"
        " when the call is reversible, small in blast radius, and plainly"
        " serves the stated task.",
        AllowCall,
    ),
    strict_tool(
        "escalate",
        "Hand the tool call to the developer, who answers the prompt already"
        " on screen. Use whenever the call is irreversible, wide in blast"
        " radius, significant, unrelatable to the stated task, or you are"
        " unsure.",
        EscalateCall,
    ),
]

_TOOL_MODELS: dict[str, type[PermissionResult]] = {
    "allow": AllowCall,
    "escalate": EscalateCall,
}

FORCED_ONE: ToolChoiceParam = {"type": "any", "disable_parallel_tool_use": True}

_PERMISSION_PROMPT = prompts.load("permission")


def build_classifier_client(cfg: ClassifierConfig) -> AsyncAnthropic:
    """Construct the classifier's own Anthropic client.

    Args:
        cfg: Unused by the client itself; accepted so the classifier's
            construction site stands on its own.

    Returns:
        A client that never retries.
    """
    del cfg  # unused; accepted so this construction site stands on its own
    return AsyncAnthropic(max_retries=0)


def render_suggestions(suggestions: Iterable[PermissionSuggestion]) -> str:
    """Render the session's permission suggestions as sorted JSON.

    Args:
        suggestions: Suggestions carried by the permission request.

    Returns:
        A JSON array, or ``(none)`` when nothing was suggested.
    """
    rendered: list[dict[str, Any]] = [
        s.model_dump() if isinstance(s, BaseModel) else s for s in suggestions
    ]
    if not rendered:
        return "(none)"
    return json.dumps(rendered, sort_keys=True, indent=2)


async def call_tool(
    client: AsyncAnthropic,
    *,
    model: str,
    max_tokens: int,
    system: Iterable[TextBlockParam],
    messages: Iterable[MessageParam],
    tools: Iterable[ToolParam],
    tool_choice: ToolChoiceParam,
) -> ToolCall:
    """Make one forced tool call and return the single tool_use block.

    Args:
        client: Client to send the request with.
        model: Model id.
        max_tokens: Cap on the response.
        system: System prompt blocks.
        messages: Conversation sent to the model.
        tools: Tool definitions offered.
        tool_choice: How the model may use them; a forcing choice is expected.

    Returns:
        The tool the model called, with its input.

    Raises:
        PermissionCallError: The SDK raised, the model refused, or the
            response did not stop on exactly one tool use.
    """
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=list(system),
            messages=list(messages),
            tools=list(tools),
            tool_choice=tool_choice,
        )
    except anthropic.AnthropicError as exc:
        raise PermissionCallError(f"{type(exc).__name__}: {exc}") from exc
    if response.stop_reason == "refusal":
        raise PermissionCallError(
            f"model refused (stop_reason=refusal, request={response._request_id})"  # pyright: ignore[reportPrivateUsage]
        )
    if response.stop_reason != "tool_use":
        raise PermissionCallError(
            f"expected stop_reason tool_use, got {response.stop_reason!r} "
            f"(request={response._request_id})"  # pyright: ignore[reportPrivateUsage]
        )
    blocks = [b for b in response.content if b.type == "tool_use"]
    if len(blocks) != 1:
        raise PermissionCallError(
            f"expected exactly one tool_use block, got {len(blocks)} "
            f"(request={response._request_id})"  # pyright: ignore[reportPrivateUsage]
        )
    return ToolCall(
        name=blocks[0].name, input=cast(dict[str, Any], blocks[0].input)
    )


def bind(client: AsyncAnthropic) -> PermissionCaller:
    """Adapt an Anthropic client into the keyword-only caller shape.

    Args:
        client: Anthropic client every classifier call is forwarded to.

    Returns:
        A callable matching the injected seam.
    """

    async def call(
        *,
        model: str,
        max_tokens: int,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        tools: list[ToolParam],
        tool_choice: ToolChoiceParam,
    ) -> ToolCall:
        """Invoke the bound client and return its single tool call."""
        return await call_tool(
            client,
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

    return call


def assemble_context(
    intent: str, tool_name: str, tool_input: dict[str, Any], suggestions: str
) -> tuple[list[TextBlockParam], list[MessageParam]]:
    """Assemble the system blocks and messages for one classifier call.

    Args:
        intent: Authoritative task intent the call is judged against.
        tool_name: Name of the tool the session is asking to run.
        tool_input: Arguments the session passed to it.
        suggestions: Rendered permission suggestions.

    Returns:
        The system blocks and the single user message, ready to pass to the
        LLM.
    """
    system: list[TextBlockParam] = [
        {"type": "text", "text": _PERMISSION_PROMPT}
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
                    "text": (
                        "# The tool call to judge\n"
                        f"tool: {tool_name}\n"
                        "input:\n"
                        + json.dumps(tool_input, sort_keys=True, indent=2)
                    ),
                },
                {
                    "type": "text",
                    "text": "# Permission suggestions offered\n" + suggestions,
                },
            ],
        }
    ]
    return system, messages


async def classify(
    llm_call: PermissionCaller,
    cfg: ClassifierConfig,
    *,
    intent: str,
    tool_name: str,
    tool_input: dict[str, Any],
    suggestions: str,
) -> PermissionResult:
    """Judge one tool call into exactly one of allow or escalate.

    Args:
        llm_call: The injected tool-calling seam.
        cfg: Supplies the model id and the token cap.
        intent: Authoritative task intent the call is judged against.
        tool_name: Name of the tool the session is asking to run.
        tool_input: Arguments the session passed to it.
        suggestions: Rendered permission suggestions.

    Returns:
        The validated call model for the tool the LLM chose.

    Raises:
        PermissionCallError: The LLM called an unknown tool, or the tool input
            failed validation.
    """
    system, messages = assemble_context(
        intent, tool_name, tool_input, suggestions
    )
    call = await llm_call(
        model=cfg.model_id,
        max_tokens=cfg.max_tokens,
        system=system,
        messages=messages,
        tools=PERMISSION_TOOLS,
        tool_choice=FORCED_ONE,
    )
    model = _TOOL_MODELS.get(call.name)
    if model is None:
        raise PermissionCallError(f"unknown permission tool {call.name!r}")
    try:
        return model.model_validate(call.input)
    except ValidationError as exc:
        raise PermissionCallError(f"invalid {call.name} input: {exc}") from exc
