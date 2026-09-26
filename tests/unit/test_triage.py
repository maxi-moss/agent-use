"""triage() against a fake LLM caller."""

from typing import Any

import pytest
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)

from broker.config import SessionModelConfig
from broker.llm import LLMCallError, ToolCall
from broker.session.llm_stack import EscalateCall
from broker.session.triage import (
    AnswerCall,
    CompleteCall,
    NoActionCall,
    triage,
)
from broker.transcript.schemas import AssistantText, UserPrompt

MODEL_CFG = SessionModelConfig(model_id="test-model", max_tokens=8192)

EVENTS = [
    UserPrompt(kind="user_prompt", text="add a login page"),
    AssistantText(kind="assistant_text", text="Which auth provider?"),
]

ESCALATE_INPUT: dict[str, Any] = {
    "reasoning": "irreversible",
    "task_summary": "Asked before an irreversible step",
    "escalation_title": "Column drop",
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
        MODEL_CFG,
        intent="add a login page using the existing session store",
        events=list(EVENTS),
        event_name="Stop",
        last_assistant_message="Which auth provider should I use?",
    )


@pytest.mark.parametrize(
    "name,tool_input,expected_type",
    [
        (
            "answer",
            {
                "reasoning": "r",
                "answer": "use oauth",
                "task_activity": "wiring up oauth",
                "task_summary": "Chose oauth",
            },
            AnswerCall,
        ),
        ("escalate", ESCALATE_INPUT, EscalateCall),
        (
            "complete",
            {
                "reasoning": "r",
                "headline": "done",
                "supporting": "tests pass",
                "task_activity": "wrapping up",
                "task_summary": "Wrapped up",
            },
            CompleteCall,
        ),
        (
            "no_action",
            {"reasoning": "r", "task_activity": "watching for input"},
            NoActionCall,
        ),
    ],
)
async def test_each_tool_maps_to_its_model(
    name: str, tool_input: dict[str, Any], expected_type: type[Any]
) -> None:
    fake = FakeLLM(ToolCall(name=name, input=tool_input))
    result = await run_triage(fake)
    assert isinstance(result, expected_type)
    if "task_activity" in tool_input:
        assert result.task_activity == tool_input["task_activity"]


async def test_unknown_tool_name_raises() -> None:
    fake = FakeLLM(ToolCall(name="surprise", input={}))
    with pytest.raises(LLMCallError):
        await run_triage(fake)


async def test_invalid_tool_input_raises() -> None:
    fake = FakeLLM(ToolCall(name="escalate", input={"reasoning": "thin"}))
    with pytest.raises(LLMCallError):
        await run_triage(fake)


async def test_forced_single_tool_choice() -> None:
    fake = FakeLLM(
        ToolCall(name="no_action", input={"reasoning": "r", "task_activity": "idle"})
    )
    await run_triage(fake)
    assert fake.calls[0]["tool_choice"] == {
        "type": "any",
        "disable_parallel_tool_use": True,
    }
    assert fake.calls[0]["model"] == "test-model"
