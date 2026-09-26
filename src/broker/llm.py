"""Anthropic client discipline, shared by triage and routing.

Rules encoded here:
- AsyncAnthropic(max_retries=0) — the SDK default of 2 silently violates the
  no-retry rule.
- Catch the ROOT anthropic.AnthropicError once; APITimeoutError subclasses
  APIConnectionError, so a discriminating chain mis-sorts timeouts.
- stop_reason "refusal" arrives as HTTP 200 with empty/partial content — it is
  checked BEFORE content is touched.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import anthropic
from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)

from pydantic import BaseModel


class LLMCallError(Exception):
    """Any LLM-call failure. One class: every failure takes the same
    escalation path, no retry."""


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]


@dataclass
class TurnResult:
    """One assistant turn under tool_choice auto: text and/or tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])


class LLMCaller[R](Protocol):
    """The one injected seam: tests pass a fake, production binds a client.

    ``R`` is the call's result type — ``ToolCall`` for triage's forced-tool
    calls, ``TurnResult`` for the master's ``tool_choice`` auto turns.
    """

    async def __call__(
        self,
        *,
        model: str,
        max_tokens: int,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        tools: list[ToolParam],
        tool_choice: ToolChoiceParam,
    ) -> R:
        """Make one call against the model and return its result."""
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


def build_client() -> AsyncAnthropic:
    """Construct the Anthropic client every caller shares.

    Returns:
        A client that never retries.
    """
    return AsyncAnthropic(max_retries=0)


async def _create(
    client: AsyncAnthropic,
    *,
    model: str,
    max_tokens: int,
    system: Iterable[TextBlockParam],
    messages: Iterable[MessageParam],
    tools: Iterable[ToolParam],
    tool_choice: ToolChoiceParam,
    timeout_s: float,
) -> anthropic.types.Message:
    """Issue one Messages request and return it only if it is usable.

    Args:
        client: Client to send the request with.
        model: Model id.
        max_tokens: Cap on the response.
        system: System prompt blocks.
        messages: Conversation sent to the model.
        tools: Tool definitions offered.
        tool_choice: How the model may use them.
        timeout_s: Deadline passed straight to the SDK call.

    Returns:
        The response message.

    Raises:
        LLMCallError: The SDK raised, or the model refused.
    """
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=list(system),
            messages=list(messages),
            tools=list(tools),
            tool_choice=tool_choice,
            timeout=timeout_s,
        )
    except anthropic.AnthropicError as exc:
        raise LLMCallError(f"{type(exc).__name__}: {exc}") from exc
    if response.stop_reason == "refusal":
        raise LLMCallError(
            "model refused (stop_reason=refusal, "
            f"request={response._request_id})"  # pyright: ignore[reportPrivateUsage]
        )
    return response


async def call_tool(
    client: AsyncAnthropic,
    *,
    model: str,
    max_tokens: int,
    system: Iterable[TextBlockParam],
    messages: Iterable[MessageParam],
    tools: Iterable[ToolParam],
    tool_choice: ToolChoiceParam,
    timeout_s: float,
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
        timeout_s: Deadline passed straight to the SDK call.

    Returns:
        The tool the model called, with its input.

    Raises:
        LLMCallError: The call failed, or didn't stop on exactly one tool use.
    """
    response = await _create(
        client,
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        timeout_s=timeout_s,
    )
    if response.stop_reason != "tool_use":
        raise LLMCallError(
            f"expected stop_reason tool_use, got {response.stop_reason!r} "
            f"(request={response._request_id})"  # pyright: ignore[reportPrivateUsage]
        )
    blocks = [b for b in response.content if b.type == "tool_use"]
    if len(blocks) != 1:
        raise LLMCallError(
            f"expected exactly one tool_use block, got {len(blocks)} "
            f"(request={response._request_id})"  # pyright: ignore[reportPrivateUsage]
        )
    return ToolCall(
        name=blocks[0].name, input=cast(dict[str, Any], blocks[0].input)
    )


async def call_turn(
    client: AsyncAnthropic,
    *,
    model: str,
    max_tokens: int,
    system: Iterable[TextBlockParam],
    messages: Iterable[MessageParam],
    tools: Iterable[ToolParam],
    tool_choice: ToolChoiceParam,
    timeout_s: float,
) -> TurnResult:
    """Take one assistant turn under tool_choice auto.

    Args:
        client: Client to send the request with.
        model: Model id.
        max_tokens: Cap on the response.
        system: System prompt blocks.
        messages: Conversation sent to the model.
        tools: Tool definitions offered.
        tool_choice: How the model may use them; auto is the intended choice.
        timeout_s: Deadline passed straight to the SDK call.

    Returns:
        The turn's concatenated text and its tool calls, in order.

    Raises:
        LLMCallError: The call failed, or the turn carried neither text nor a
            tool call.
    """
    response = await _create(
        client,
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        timeout_s=timeout_s,
    )
    result = TurnResult()
    text_parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            result.tool_calls.append(
                ToolCall(name=block.name, input=cast(dict[str, Any], block.input))
            )
    result.text = "".join(text_parts)
    if not result.text and not result.tool_calls:
        raise LLMCallError(
            f"empty response (stop_reason={response.stop_reason!r}, "
            f"request={response._request_id})"  # pyright: ignore[reportPrivateUsage]
        )
    return result
