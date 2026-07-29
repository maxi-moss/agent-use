"""The classifier's own tool surface, client discipline, and isolation.

The strict-tool derivation here duplicates the session broker's on purpose, so
the schema pin is what stops the two drifting apart unnoticed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import anthropic
import pytest
from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)

from broker.config import ClassifierConfig
from broker.permission.llm import (
    PERMISSION_TOOLS,
    PermissionCallError,
    ToolCall,
    build_classifier_client,
    call_tool,
    classify,
    render_suggestions,
)
from broker.permission.schemas import AllowCall, EscalateCall
from broker.protocol.schemas import AddDirectoriesSuggestion

PACKAGE = Path(__file__).parent.parent.parent / "src" / "broker" / "permission"

CFG = ClassifierConfig()

CALL_KWARGS: dict[str, Any] = {
    "model": "claude-haiku-4-5",
    "max_tokens": 1024,
    "system": [],
    "messages": [],
    "tools": [],
    "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
}


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

    async def create(self, **kwargs: Any) -> FakeResponse:
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class FakeClient:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages


def as_client(fake: FakeClient) -> AsyncAnthropic:
    return cast(AsyncAnthropic, fake)


class FakeLLM:
    def __init__(self, result: ToolCall) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        return self.result


def _walk_objects(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        typed = cast(dict[str, Any], node)
        if typed.get("type") == "object" or "properties" in typed:
            found.append((path, typed))
        for key, value in typed.items():
            found.extend(_walk_objects(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, item in enumerate(cast(list[Any], node)):
            found.extend(_walk_objects(item, f"{path}[{i}]"))
    return found


def test_both_tools_are_strict_and_fully_required() -> None:
    assert [t["name"] for t in PERMISSION_TOOLS] == ["allow", "escalate"]
    for tool in PERMISSION_TOOLS:
        assert tool.get("strict") is True, f"{tool['name']} not strict"
        schema = cast(dict[str, Any], tool["input_schema"])
        objects = _walk_objects(schema)
        assert objects, f"{tool['name']} has no object schema"
        for path, obj in objects:
            assert obj.get("additionalProperties") is False, (
                f"{tool['name']} {path}: additionalProperties not false"
            )
            props = cast(dict[str, Any], obj.get("properties", {}))
            assert sorted(cast(list[str], obj.get("required", []))) == sorted(
                props.keys()
            ), f"{tool['name']} {path}: not every property is required"
        assert list(cast(dict[str, Any], schema["properties"])) == ["reasoning"]


def test_client_has_zero_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert build_classifier_client(CFG).max_retries == 0


async def test_refusal_raises_before_content_indexing() -> None:
    fake = FakeClient(FakeMessages(FakeResponse("refusal", [])))
    with pytest.raises(PermissionCallError, match="refus"):
        await call_tool(as_client(fake), **CALL_KWARGS)


async def test_unexpected_stop_reason_raises() -> None:
    fake = FakeClient(FakeMessages(FakeResponse("max_tokens", [])))
    with pytest.raises(PermissionCallError, match="max_tokens"):
        await call_tool(as_client(fake), **CALL_KWARGS)


async def test_multiple_tool_blocks_raise() -> None:
    blocks = [
        FakeBlock(type="tool_use", name="allow", input={"reasoning": "a"}),
        FakeBlock(type="tool_use", name="escalate", input={"reasoning": "b"}),
    ]
    fake = FakeClient(FakeMessages(FakeResponse("tool_use", blocks)))
    with pytest.raises(PermissionCallError, match="exactly one"):
        await call_tool(as_client(fake), **CALL_KWARGS)


async def test_sdk_error_wraps_into_permission_call_error() -> None:
    class FakeSDKError(anthropic.AnthropicError):
        pass

    fake = FakeClient(FakeMessages(error=FakeSDKError("boom")))
    with pytest.raises(PermissionCallError, match="FakeSDKError"):
        await call_tool(as_client(fake), **CALL_KWARGS)


async def _run(fake: FakeLLM) -> Any:
    return await classify(
        fake,
        CFG,
        intent="add a login page",
        tool_name="Bash",
        tool_input={"command": "git push"},
        suggestions="(none)",
    )


@pytest.mark.parametrize(
    "name,expected_type",
    [("allow", AllowCall), ("escalate", EscalateCall)],
)
async def test_each_tool_maps_to_its_model(
    name: str, expected_type: type[Any]
) -> None:
    fake = FakeLLM(ToolCall(name=name, input={"reasoning": "r"}))
    assert isinstance(await _run(fake), expected_type)


async def test_unknown_tool_name_raises() -> None:
    fake = FakeLLM(ToolCall(name="deny", input={"reasoning": "r"}))
    with pytest.raises(PermissionCallError):
        await _run(fake)


async def test_invalid_tool_input_raises() -> None:
    fake = FakeLLM(ToolCall(name="allow", input={}))
    with pytest.raises(PermissionCallError):
        await _run(fake)


async def test_the_classifier_model_is_forced_and_pinned() -> None:
    fake = FakeLLM(ToolCall(name="allow", input={"reasoning": "r"}))
    await _run(fake)
    assert fake.calls[0]["model"] == "claude-haiku-4-5"
    assert fake.calls[0]["tool_choice"] == {
        "type": "any",
        "disable_parallel_tool_use": True,
    }
    sent = "".join(
        cast(str, block["text"])
        for block in cast(list[dict[str, Any]], fake.calls[0]["messages"][0]["content"])
    )
    assert "add a login page" in sent
    assert "git push" in sent


def test_unknown_suggestion_arm_renders_as_itself() -> None:
    rendered = render_suggestions(
        [
            AddDirectoriesSuggestion(type="addDirectories", directories=["/repo"]),
            {"type": "somethingNew", "payload": 1},
        ]
    )
    assert "/repo" in rendered
    assert "somethingNew" in rendered
    assert render_suggestions([]) == "(none)"


def test_package_never_imports_the_shared_llm_or_broker_internals() -> None:
    """The classifier's model must not be re-pinnable through a shared helper."""
    forbidden = ("broker.llm", "broker.session", "broker.master", "broker.transcript")
    for path in sorted(PACKAGE.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path.name} references {name}"
