"""triage() + ground_intent() against a fake LLM caller."""

from pathlib import Path
from typing import Any, cast

import pytest
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)

from broker import prompts
from broker.config import BrokerConfig
from broker.llm import LLMCallError, ToolCall
from broker.session.triage import (
    AnswerCall,
    CompleteCall,
    EscalateCall,
    NoActionCall,
    ProposePromptCall,
    assemble_context,
    ground_intent,
    triage,
)
from broker.transcript.schemas import AssistantText, UserPrompt
from broker.transcript.adapter import render

CFG = BrokerConfig(broker_home=Path("/private/tmp/unused"))

EVENTS = [
    UserPrompt(kind="user_prompt", text="add a login page"),
    AssistantText(kind="assistant_text", text="Which auth provider?"),
]

ESCALATE_INPUT: dict[str, Any] = {
    "reasoning": "irreversible",
    "situation": "wants to drop a column",
    "what_was_asked": "drop users.email?",
    "what_is_at_stake": "real data",
    "alternatives": [{"option": "keep", "pros": "safe", "cons": "cruft"}],
    "recommendation": "keep it",
    "uncertainty": "unsure if column is used",
    "what_would_change_my_mind": "proof it is unused",
}


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


async def run_triage(fake: FakeLLM) -> Any:
    return await triage(
        fake,
        CFG,
        intent="add a login page using the existing session store",
        events=list(EVENTS),
        event_name="Stop",
        last_assistant_message="Which auth provider should I use?",
    )


@pytest.mark.parametrize(
    "name,tool_input,expected_type",
    [
        ("answer", {"reasoning": "r", "answer": "use oauth"}, AnswerCall),
        ("escalate", ESCALATE_INPUT, EscalateCall),
        ("complete", {"reasoning": "r", "summary": "done"}, CompleteCall),
        ("no_action", {"reasoning": "r"}, NoActionCall),
    ],
)
async def test_each_tool_maps_to_its_model(
    name: str, tool_input: dict[str, Any], expected_type: type[Any]
) -> None:
    fake = FakeLLM(ToolCall(name=name, input=tool_input))
    result = await run_triage(fake)
    assert isinstance(result, expected_type)


async def test_unknown_tool_name_raises() -> None:
    fake = FakeLLM(ToolCall(name="surprise", input={}))
    with pytest.raises(LLMCallError):
        await run_triage(fake)


async def test_invalid_tool_input_raises() -> None:
    fake = FakeLLM(ToolCall(name="escalate", input={"reasoning": "thin"}))
    with pytest.raises(LLMCallError):
        await run_triage(fake)


async def test_forced_single_tool_choice() -> None:
    fake = FakeLLM(ToolCall(name="no_action", input={"reasoning": "r"}))
    await run_triage(fake)
    assert fake.calls[0]["tool_choice"] == {
        "type": "any",
        "disable_parallel_tool_use": True,
    }
    assert fake.calls[0]["model"] == "claude-sonnet-5"


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


async def test_ground_intent_passes_codebase_facts(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("# Rules\nUse uv.\n")
    fake = FakeLLM(
        ToolCall(
            name="propose_prompt",
            input={"reasoning": "r", "prompt": "Add the page."},
        )
    )
    result = await ground_intent(
        fake, CFG, intent="add a page THE-RAW-INTENT", cwd=tmp_path
    )
    assert isinstance(result, ProposePromptCall)
    assert result.prompt == "Add the page."
    sent = cast(str, fake.calls[0]["messages"][0]["content"])
    assert "THE-RAW-INTENT" in sent  # intent verbatim
    assert "Use uv." in sent  # CLAUDE.md folded in


async def test_ground_intent_wrong_tool_raises(tmp_path: Path) -> None:
    fake = FakeLLM(ToolCall(name="answer", input={"reasoning": "r", "answer": "a"}))
    with pytest.raises(LLMCallError):
        await ground_intent(fake, CFG, intent="x", cwd=tmp_path)
