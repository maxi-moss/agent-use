"""ground_intent() against a fake LLM caller and a fake retriever."""

from pathlib import Path
from typing import Any, cast

import pytest
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)

from broker.config import SessionModelConfig
from broker.index.retrieval import RetrievalError
from broker.index.schemas import ContextSymbol, GroundingContext, SymbolKind
from broker.llm import LLMCallError, ToolCall
from broker.session.grounding import Grounding, ground_intent

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
