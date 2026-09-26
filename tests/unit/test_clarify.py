"""clarify() against a fake LLM caller."""

from typing import Any, cast

import pytest
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)

from broker.config import SessionModelConfig
from broker.llm import LLMCallError, ToolCall
from broker.protocol.schemas import EscalationPayload
from broker.session.clarify import ClarifyCall, clarify, render_disclosure
from broker.session.triage import FORCED_ONE
from broker.transcript.schemas import AssistantText, UserPrompt

MODEL_CFG = SessionModelConfig(model_id="test-model", max_tokens=8192)

EVENTS = [
    UserPrompt(kind="user_prompt", text="add a login page"),
    AssistantText(kind="assistant_text", text="I tried the session store first."),
]

ESCALATION = EscalationPayload.model_validate(
    {
        "escalation_id": "e1",
        "session_id": "s1",
        "task_context": "add a login page",
        "disclosure": {
            "escalation_title": "TITLE-TEXT",
            "situation": "SITUATION-TEXT",
            "what_was_asked": "ASKED-TEXT",
            "what_is_at_stake": "STAKE-TEXT",
            "alternatives": [
                {"option": "OPTION-A", "pros": "PROS-A", "cons": "CONS-A"},
                {"option": "OPTION-B", "pros": "PROS-B", "cons": "CONS-B"},
            ],
            "recommendation": "RECOMMENDATION-TEXT",
            "uncertainty": "UNCERTAINTY-TEXT",
            "what_would_change_my_mind": "CHANGE-MIND-TEXT",
        },
    }
)

QUESTION = "what did it already try?"


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


async def run_clarify(fake: FakeLLM) -> ClarifyCall:
    return await clarify(
        fake,
        MODEL_CFG,
        intent="add a login page using the existing session store",
        escalation=ESCALATION,
        question=QUESTION,
        events=list(EVENTS),
    )


async def test_answer_maps_to_model() -> None:
    fake = FakeLLM(
        ToolCall(
            name="answer_clarification",
            input={"reasoning": "r", "answer": "it tried X"},
        )
    )
    result = await run_clarify(fake)
    assert result.answer == "it tried X"
    assert result.reasoning == "r"


async def test_unknown_tool_raises() -> None:
    fake = FakeLLM(ToolCall(name="surprise", input={}))
    with pytest.raises(LLMCallError):
        await run_clarify(fake)


async def test_invalid_input_raises() -> None:
    fake = FakeLLM(ToolCall(name="answer_clarification", input={"answer": "x"}))
    with pytest.raises(LLMCallError):
        await run_clarify(fake)


async def test_forced_single_tool_and_model() -> None:
    fake = FakeLLM(
        ToolCall(name="answer_clarification", input={"reasoning": "r", "answer": "a"})
    )
    await run_clarify(fake)
    call = fake.calls[0]
    assert call["tool_choice"] == FORCED_ONE
    assert call["model"] == "test-model"
    assert [t["name"] for t in call["tools"]] == ["answer_clarification"]


async def test_context_places_disclosure_and_question_in_working() -> None:
    fake = FakeLLM(
        ToolCall(name="answer_clarification", input={"reasoning": "r", "answer": "a"})
    )
    await run_clarify(fake)
    content = cast(list[dict[str, Any]], fake.calls[0]["messages"][0]["content"])
    assert len(content) == 3
    working = content[2]["text"]
    assert render_disclosure(ESCALATION.disclosure) in working
    assert QUESTION in working
    assert "cache_control" not in content[2]
    # The transcript block keeps its breakpoint: repeat questions on the same
    # still-pending escalation hit the cache.
    assert "cache_control" in content[1]
    assert "I tried the session store first." in content[1]["text"]


def test_render_disclosure_includes_analysis() -> None:
    text = render_disclosure(ESCALATION.disclosure)
    for expected in (
        "SITUATION-TEXT",
        "ASKED-TEXT",
        "STAKE-TEXT",
        "OPTION-A",
        "CONS-B",
        "RECOMMENDATION-TEXT",
        "UNCERTAINTY-TEXT",
        "CHANGE-MIND-TEXT",
    ):
        assert expected in text
    assert "e1" not in text
