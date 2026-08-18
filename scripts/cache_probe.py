#!/usr/bin/env python
"""Prompt-cache probe — RUN MANUALLY, makes real API calls:

    uv run python scripts/cache_probe.py

Assembles a real triage context and issues the identical request twice, then
reports the cache usage of each call so the 1-hour-TTL break-even is measured
rather than assumed. call 1 is expected to write the prefix and call 2 to read
it back — if the triage prefix clears the model's minimum cacheable size. A
zero read is a measurement, not a defect to engineer around.

call_tool discards response.usage, so both calls go through a raw
messages.create to read the token counts directly.

The permission prefix is probed once for contrast: it sits far below its
model's minimum cacheable size, so cache_creation_input_tokens: 0 is the
expected reading there.
"""

import asyncio
import os
import sys

from broker.config import BrokerConfig, ClassifierConfig
from broker import prompts
from broker.llm import build_client
from broker.permission.llm import (
    FORCED_ONE as PERMISSION_FORCED_ONE,
    PERMISSION_TOOLS,
    assemble_context as assemble_permission_context,
    build_classifier_client,
    render_suggestions,
)
from broker.session.triage import (
    FORCED_ONE,
    TRIAGE_TOOLS,
    assemble_context as assemble_triage_context,
)

_INTENT = "Add OAuth login to the app."
_WORKING = (
    "# What just happened\nhook event: Stop\n\n"
    "# The coding agent's last message (triage THIS)\n"
    "Where should the token store live?"
)


async def probe_triage() -> None:
    """Issue two identical triage calls and report each call's cache usage."""
    cfg = BrokerConfig()
    client = build_client(cfg)
    system, messages = assemble_triage_context(
        prompts.load("triage"), _INTENT, [], _WORKING
    )
    print(f"triage ({cfg.model_id}): two identical calls")
    for i in (1, 2):
        response = await client.messages.create(
            model=cfg.model_id,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=messages,
            tools=TRIAGE_TOOLS,
            tool_choice=FORCED_ONE,
        )
        u = response.usage
        print(
            f"  call {i}: create={u.cache_creation_input_tokens} "
            f"read={u.cache_read_input_tokens} "
            f"input={u.input_tokens} output={u.output_tokens}"
        )


async def probe_permission() -> None:
    """Issue one classifier call and report its cache-creation count."""
    cfg = ClassifierConfig()
    client = build_classifier_client(cfg)
    system, messages = assemble_permission_context(
        _INTENT,
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
    u = response.usage
    print(f"permission ({cfg.model_id}): one call")
    print(
        f"  call 1: create={u.cache_creation_input_tokens} "
        f"read={u.cache_read_input_tokens} "
        f"input={u.input_tokens} output={u.output_tokens}"
    )


async def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1
    await probe_triage()
    await probe_permission()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
