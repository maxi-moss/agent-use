"""assemble_context: block order, cache breakpoints and the context byte pin.

Also forced_call: the one forced-tool call every session-stack call site
shares (triage, ask, clarify, grounding each pin their own call site's label
and tool table separately).
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from anthropic.types import MessageParam, TextBlockParam, ToolChoiceParam, ToolParam
from pydantic import BaseModel, ConfigDict

from broker import prompts
from broker.config import SessionModelConfig
from broker.llm import LLMCallError, ToolCall
from broker.session.llm_stack import (
    FORCED_ONE,
    SESSION_CALL_TIMEOUT_S,
    assemble_context,
    bind_call_tool,
    forced_call,
)
from broker.transcript.adapter import render
from broker.transcript.schemas import AssistantText, UserPrompt

EVENTS = [
    UserPrompt(kind="user_prompt", text="add a login page"),
    AssistantText(kind="assistant_text", text="Which auth provider?"),
]

MODEL_CFG = SessionModelConfig(model_id="test-model", max_tokens=8192)


class _ProbeCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


class _FakeLLM:
    def __init__(self, result: ToolCall) -> None:
        self.result = result

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
        return self.result


async def test_forced_call_unknown_tool_name_raises() -> None:
    fake = _FakeLLM(ToolCall(name="surprise", input={}))
    with pytest.raises(LLMCallError):
        await forced_call(
            fake,
            MODEL_CFG,
            system=[],
            messages=[],
            tools=[],
            models={"probe": _ProbeCall},
            label="probe",
        )


async def test_forced_call_invalid_input_raises() -> None:
    fake = _FakeLLM(ToolCall(name="probe", input={}))
    with pytest.raises(LLMCallError):
        await forced_call(
            fake,
            MODEL_CFG,
            system=[],
            messages=[],
            tools=[],
            models={"probe": _ProbeCall},
            label="probe",
        )


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


@dataclass
class _FakeBlock:
    type: str
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict[str, Any])


class _FakeResponse:
    def __init__(self) -> None:
        self.stop_reason = "tool_use"
        self.content = [_FakeBlock(type="tool_use", name="probe", input={})]
        self._request_id = "req_test"


class _FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse()


class _FakeClient:
    def __init__(self, messages: _FakeMessages) -> None:
        self.messages = messages


async def test_bind_call_tool_passes_the_session_timeout() -> None:
    messages = _FakeMessages()
    llm_call = bind_call_tool(cast(Any, _FakeClient(messages)))
    await llm_call(
        model="test-model",
        max_tokens=8192,
        system=[],
        messages=[],
        tools=[],
        tool_choice=FORCED_ONE,
    )
    assert messages.calls[0]["timeout"] == SESSION_CALL_TIMEOUT_S


def test_assembled_context_schema_pin() -> None:
    system, messages = assemble_context(prompts.load("triage"), "i", list(EVENTS), "w")
    payload = {"system": system, "messages": messages}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert digest == (
        "e50f429144d92cb925a44e5c132df69689fd3b6ccacfaaa9ab472d2e814f733a"
    ), "LLM-visible context bytes changed; review assemble_context and update the pin"
