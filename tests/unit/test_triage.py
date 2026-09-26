"""triage() + ground_intent() against a fake LLM caller."""

import hashlib
import json
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
from broker.config import SessionModelConfig
from broker.index.retrieval import RetrievalError
from broker.index.schemas import ContextSymbol, GroundingContext, SymbolKind
from broker.llm import LLMCallError, ToolCall
from broker.session.triage import (
    AnswerCall,
    CompleteCall,
    EscalateCall,
    Grounding,
    NoActionCall,
    assemble_context,
    ground_intent,
    triage,
)
from broker.transcript.schemas import AssistantText, UserPrompt
from broker.transcript.adapter import render

MODEL_CFG = SessionModelConfig(model_id="test-model", max_tokens=8192)

CONTEXT = GroundingContext(
    symbols=[
        ContextSymbol(
            qualified_name="src/x.py::do_it",
            path="src/x.py",
            kind=SymbolKind.FUNCTION,
            start_line=1,
            end_line=3,
            signature="def do_it() -> None:",
            fields=[],
            methods=[],
            score=0.7,
            rank=0.7,
        )
    ],
    edges=[],
    imports={},
)


class FakeRetriever:
    def __init__(self, context: GroundingContext = CONTEXT) -> None:
        self.context = context
        self.calls: list[tuple[str, Path]] = []

    async def __call__(self, intent: str, cwd: Path) -> GroundingContext:
        self.calls.append((intent, cwd))
        return self.context


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


async def test_ground_intent_orders_intent_claude_md_relevant_code(
    tmp_path: Path,
) -> None:
    (tmp_path / "CLAUDE.md").write_text("# Rules\nUse uv.\n")
    fake = FakeLLM(
        ToolCall(
            name="propose_prompt", input={"reasoning": "r", "prompt": "Add the page."}
        )
    )
    retriever = FakeRetriever()
    result = await ground_intent(
        fake,
        MODEL_CFG,
        retrieve=retriever,
        intent="add a page THE-RAW-INTENT",
        cwd=tmp_path,
    )
    assert isinstance(result, Grounding)
    assert result.proposal.prompt == "Add the page."
    assert result.context == CONTEXT
    assert retriever.calls == [("add a page THE-RAW-INTENT", tmp_path)]
    sent = cast(str, fake.calls[0]["messages"][0]["content"])
    intent_at = sent.index("# Developer intent (verbatim)\nadd a page THE-RAW-INTENT")
    claude_at = sent.index("# The codebase's CLAUDE.md\n# Rules\nUse uv.")
    code_at = sent.index(
        "# Relevant code\n\n## src/x.py\n### do_it (function, seed, lines 1-3)"
    )
    assert intent_at < claude_at < code_at
    assert "# Tracked files" not in sent


async def test_ground_intent_without_claude_md_still_sends_relevant_code(
    tmp_path: Path,
) -> None:
    fake = FakeLLM(
        ToolCall(name="propose_prompt", input={"reasoning": "r", "prompt": "p"})
    )
    await ground_intent(
        fake, MODEL_CFG, retrieve=FakeRetriever(), intent="x", cwd=tmp_path
    )
    sent = cast(str, fake.calls[0]["messages"][0]["content"])
    assert "# The codebase's CLAUDE.md" not in sent
    assert "def do_it() -> None:" in sent


async def test_ground_intent_retrieval_failure_propagates(tmp_path: Path) -> None:
    class Failing:
        async def __call__(self, intent: str, cwd: Path) -> GroundingContext:
            raise RetrievalError("no code index")

    fake = FakeLLM(
        ToolCall(name="propose_prompt", input={"reasoning": "r", "prompt": "p"})
    )
    with pytest.raises(RetrievalError):
        await ground_intent(
            fake, MODEL_CFG, retrieve=Failing(), intent="x", cwd=tmp_path
        )
    assert fake.calls == []  # the LLM is never called without retrieval


async def test_ground_intent_wrong_tool_raises(tmp_path: Path) -> None:
    fake = FakeLLM(ToolCall(name="answer", input={"reasoning": "r", "answer": "a"}))
    with pytest.raises(LLMCallError):
        await ground_intent(
            fake, MODEL_CFG, retrieve=FakeRetriever(), intent="x", cwd=tmp_path
        )
