#!/usr/bin/env python
"""One-shot LIVE smoke test — RUN MANUALLY, costs two API calls:

    uv run python scripts/llm_smoke.py

Verifies the strict + tool_choice "any" + disable_parallel_tool_use trio
against the real pinned models (the docs only recommend the combo
pairwise), once for the session broker's triage surface and once for the
permission classifier's. Each call fails loud unless EXACTLY one tool_use
block returns, so a clean exit is the verification.

The classifier call goes through the classifier's own client builder and its
own tools, never the session broker's, and reports
cache_creation_input_tokens: the classifier's prompt sits far below its
model's minimum cacheable prefix, so 0 is the expected reading and padding
the prompt to cross that floor is not a trade worth making.
"""

import asyncio
import os
import sys

from anthropic.types import MessageParam, TextBlockParam

from broker.config import BrokerConfig, ClassifierConfig
from broker.llm import build_client, call_tool
from broker.permission.llm import (
    FORCED_ONE as PERMISSION_FORCED_ONE,
    PERMISSION_TOOLS,
    assemble_context,
    build_classifier_client,
    render_suggestions,
)
from broker.session.triage import FORCED_ONE, TRIAGE_TOOLS

SYSTEM: list[TextBlockParam] = [
    {
        "type": "text",
        "text": (
            "You are a triage broker supervising a coding agent. Classify "
            "the agent's last message by calling exactly one tool."
        ),
    }
]

MESSAGES: list[MessageParam] = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "# Authoritative task intent\n"
                    "Add OAuth login to the app.\n\n"
                    "# The coding agent's last message (triage THIS)\n"
                    "Should I store refresh tokens in the existing session "
                    "table, or create a dedicated table for them?"
                ),
            }
        ],
    }
]


async def session_call() -> None:
    """Call the session broker's triage surface against its pinned model."""
    cfg = BrokerConfig()  # the pinned production model — deliberately
    client = build_client()
    call = await call_tool(
        client,
        model=cfg.model_id,
        max_tokens=cfg.max_tokens,
        system=SYSTEM,
        messages=MESSAGES,
        tools=TRIAGE_TOOLS,
        tool_choice=FORCED_ONE,
    )
    print(f"triage  ({cfg.model_id}): one tool_use block -> {call.name}")
    print(f"    input keys: {sorted(call.input)}")


async def classifier_call() -> None:
    """Call the permission classifier's surface against its pinned model.

    The response is read directly rather than through the classifier's
    ``call_tool``, which returns the tool call alone: the token usage is the
    point of this call.
    """
    cfg = ClassifierConfig()  # the pinned classifier model — deliberately
    client = build_classifier_client(cfg)
    system, messages = assemble_context(
        "Add OAuth login to the app.",
        "Bash",
        {"command": "rm -rf ~/.ssh", "description": "clean up keys"},
        render_suggestions([]),
    )
    response = await client.messages.create(
        model=cfg.model_id,
        max_tokens=cfg.max_tokens,
        system=system,
        messages=messages,
        tools=PERMISSION_TOOLS,
        tool_choice=PERMISSION_FORCED_ONE,
    )
    blocks = [b for b in response.content if b.type == "tool_use"]
    if len(blocks) != 1:
        raise SystemExit(
            f"expected exactly one tool_use block, got {len(blocks)}"
        )
    print(f"classify ({cfg.model_id}): one tool_use block -> {blocks[0].name}")
    print(
        "    cache_creation_input_tokens: "
        f"{response.usage.cache_creation_input_tokens}"
    )


async def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1
    await session_call()
    await classifier_call()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
