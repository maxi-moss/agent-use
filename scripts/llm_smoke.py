#!/usr/bin/env python
"""One-shot LIVE smoke test — RUN MANUALLY, costs one API call:

    uv run python scripts/llm_smoke.py

Verifies the strict + tool_choice "any" + disable_parallel_tool_use trio
against the real pinned model (the docs only recommend the combo
pairwise). call_tool already fails loud unless EXACTLY one tool_use block
returns, so a clean exit is the verification.
"""

import asyncio
import os
import sys

from anthropic.types import MessageParam, TextBlockParam

from broker.config import BrokerConfig
from broker.llm import build_client, call_tool
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


async def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1
    cfg = BrokerConfig()  # the pinned production model — deliberately
    client = build_client(cfg)
    call = await call_tool(
        client,
        model=cfg.model_id,
        max_tokens=cfg.max_tokens,
        system=SYSTEM,
        messages=MESSAGES,
        tools=TRIAGE_TOOLS,
        tool_choice=FORCED_ONE,
    )
    print(f"OK: exactly one tool_use block returned -> {call.name}")
    print(f"    input keys: {sorted(call.input)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
