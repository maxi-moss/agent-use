"""llm.py: zero retries, refusal handling, exactly-one-tool-block discipline."""

from dataclasses import dataclass, field
from typing import Any, cast

import anthropic
import pytest
from anthropic import AsyncAnthropic

from broker.config import BrokerConfig
from broker.llm import LLMCallError, ToolCall, build_client, call_tool, call_turn


@dataclass
class FakeBlock:
    type: str
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict[str, Any])
    text: str = ""


class FakeResponse:
    def __init__(self, stop_reason: str, content: list[FakeBlock]) -> None:
        self.stop_reason = stop_reason
        self.content = content
        self._request_id = "req_test"


class FakeMessages:
    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class FakeClient:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


def as_client(fake: FakeClient) -> AsyncAnthropic:
    return cast(AsyncAnthropic, fake)


CALL_KWARGS: dict[str, Any] = {
    "model": "claude-opus-5",
    "max_tokens": 1024,
    "system": [],
    "messages": [],
    "tools": [],
    "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
}


def test_client_has_zero_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = build_client(BrokerConfig())
    assert client.max_retries == 0


async def test_refusal_raises() -> None:
    fake = FakeClient(FakeMessages(FakeResponse("refusal", [])))
    with pytest.raises(LLMCallError, match="refus"):
        await call_tool(as_client(fake), **CALL_KWARGS)


async def test_refusal_raises_before_content_indexing() -> None:
    """Empty content on refusal must raise LLMCallError, never IndexError."""
    fake = FakeClient(FakeMessages(FakeResponse("refusal", [])))
    with pytest.raises(LLMCallError):
        await call_turn(as_client(fake), **CALL_KWARGS)


async def test_multiple_tool_blocks_raise() -> None:
    blocks = [
        FakeBlock(type="tool_use", name="a", input={}),
        FakeBlock(type="tool_use", name="b", input={}),
    ]
    fake = FakeClient(FakeMessages(FakeResponse("tool_use", blocks)))
    with pytest.raises(LLMCallError, match="exactly one"):
        await call_tool(as_client(fake), **CALL_KWARGS)


async def test_zero_tool_blocks_raise() -> None:
    fake = FakeClient(FakeMessages(FakeResponse("tool_use", [])))
    with pytest.raises(LLMCallError):
        await call_tool(as_client(fake), **CALL_KWARGS)


async def test_unexpected_stop_reason_raises() -> None:
    fake = FakeClient(FakeMessages(FakeResponse("max_tokens", [])))
    with pytest.raises(LLMCallError, match="max_tokens"):
        await call_tool(as_client(fake), **CALL_KWARGS)


async def test_sdk_error_wraps_into_llm_call_error() -> None:
    class FakeSDKError(anthropic.AnthropicError):
        pass

    fake = FakeClient(FakeMessages(error=FakeSDKError("boom")))
    with pytest.raises(LLMCallError, match="FakeSDKError"):
        await call_tool(as_client(fake), **CALL_KWARGS)


async def test_call_tool_returns_single_block() -> None:
    block = FakeBlock(type="tool_use", name="answer", input={"answer": "yes"})
    fake = FakeClient(FakeMessages(FakeResponse("tool_use", [block])))
    result = await call_tool(as_client(fake), **CALL_KWARGS)
    assert result == ToolCall(name="answer", input={"answer": "yes"})


async def test_call_turn_collects_text_and_tools() -> None:
    blocks = [
        FakeBlock(type="text", text="thinking about it. "),
        FakeBlock(type="tool_use", name="list_sessions", input={}),
    ]
    fake = FakeClient(FakeMessages(FakeResponse("tool_use", blocks)))
    result = await call_turn(as_client(fake), **CALL_KWARGS)
    assert result.text == "thinking about it. "
    assert result.tool_calls == [ToolCall(name="list_sessions", input={})]
