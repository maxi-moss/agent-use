"""assemble_context: block order, cache breakpoints and the context byte pin."""

import hashlib
import json
from typing import Any, cast

from broker import prompts
from broker.session.llm_stack import assemble_context
from broker.transcript.adapter import render
from broker.transcript.schemas import AssistantText, UserPrompt

EVENTS = [
    UserPrompt(kind="user_prompt", text="add a login page"),
    AssistantText(kind="assistant_text", text="Which auth provider?"),
]


def test_context_order_intent_transcript_working() -> None:
    system, messages = assemble_context(
        prompts.load("triage"), "the intent", list(EVENTS), "the working context"
    )
    assert len(messages) == 1
    content = cast(list[dict[str, Any]], messages[0]["content"])
    assert len(content) == 3
    assert content[0]["text"].startswith("# Authoritative task intent\n")
    assert "the intent" in content[0]["text"]
    assert content[1]["text"] == render(list(EVENTS))
    assert content[2]["text"] == "the working context"
    assert system[0]["text"]  # triage prompt, non-empty


def test_exactly_two_cache_breakpoints() -> None:
    system, messages = assemble_context(prompts.load("triage"), "i", list(EVENTS), "w")
    blocks = cast(list[dict[str, Any]], list(system)) + cast(
        list[dict[str, Any]], messages[0]["content"]
    )
    with_cache = [b for b in blocks if "cache_control" in b]
    assert len(with_cache) == 2
    for block in with_cache:
        assert block["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # system prefix and transcript tail carry them; intent and working do not
    assert "cache_control" in system[0]
    content = cast(list[dict[str, Any]], messages[0]["content"])
    assert "cache_control" in content[1]


def test_assembled_context_schema_pin() -> None:
    system, messages = assemble_context(prompts.load("triage"), "i", list(EVENTS), "w")
    payload = {"system": system, "messages": messages}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert digest == (
        "e50f429144d92cb925a44e5c132df69689fd3b6ccacfaaa9ab472d2e814f733a"
    ), "LLM-visible context bytes changed; review assemble_context and update the pin"
