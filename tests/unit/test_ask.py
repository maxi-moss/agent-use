"""ask decision stack: parsing, rendering, validation, and the retry loop."""

import asyncio
from typing import Any, cast

import pytest

from broker.config import SessionModelConfig
from broker.llm import LLMCallError, ToolCall
from broker.session.ask import (
    AnswerQuestionsCall,
    AnswerValidationError,
    AskInputError,
    EscalateCall,
    ask_once,
    decide_questions,
    parse_questions,
    render_questions,
    validate_answers,
)
from broker.transcript.schemas import Option, Question

MODEL_CFG = SessionModelConfig(model_id="test-model", max_tokens=8192)

ESCALATE_INPUT: dict[str, Any] = {
    "reasoning": "irreversible",
    "task_summary": "Asked before an irreversible step",
    "escalation_title": "Production database choice",
    "situation": "the menu picks a production database",
    "what_was_asked": "which database",
    "what_is_at_stake": "real data",
    "alternatives": [{"option": "a", "pros": "p", "cons": "c"}],
    "recommendation": "ask the developer",
    "uncertainty": "data migration cost",
    "what_would_change_my_mind": "a staging-only scope",
}


class FakeLLM:
    """Results are fed through a queue: an empty queue models a pending call."""

    def __init__(self) -> None:
        self.results: asyncio.Queue[ToolCall] = asyncio.Queue()
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> ToolCall:
        self.calls.append(kwargs)
        return await self.results.get()


def make_question(
    text: str, labels: list[str], *, multi: bool = False, header: str = "Header"
) -> Question:
    return Question(
        question=text,
        header=header,
        options=[Option(label=lab, description=f"{lab} desc") for lab in labels],
        multiSelect=multi,
    )


SINGLE = make_question("Which database?", ["PostgreSQL", "MySQL"])
MULTI = make_question(
    "Which features?", ["auth", "billing", "search"], multi=True
)


def entry(
    question: str, selected: list[str] | None = None, free_text: str = ""
) -> dict[str, Any]:
    return {
        "question": question,
        "selected": selected or [],
        "free_text": free_text,
    }


def answers_call(*entries: dict[str, Any]) -> AnswerQuestionsCall:
    return AnswerQuestionsCall.model_validate(
        {"reasoning": "grounded", "answers": list(entries)}
    )


def answers_tool_call(*entries: dict[str, Any]) -> ToolCall:
    return ToolCall(
        name="answer_questions",
        input={"reasoning": "grounded", "answers": list(entries)},
    )


def working_text(call_kwargs: dict[str, Any]) -> str:
    content = cast(list[dict[str, Any]], call_kwargs["messages"][0]["content"])
    return cast(str, content[-1]["text"])


# ── parse_questions ──────────────────────────────────────────────────────────


def test_parse_valid_single_and_multi() -> None:
    tool_input = {
        "questions": [
            {
                "question": "Which database?",
                "header": "Database",
                "options": [
                    {"label": "PostgreSQL", "description": "pg"},
                    {"label": "MySQL", "description": "my"},
                ],
            },
            {
                "question": "Which features?",
                "header": "Features",
                "multiSelect": True,
                "options": [
                    {"label": "auth", "description": "a"},
                    {"label": "billing", "description": "b"},
                ],
            },
        ]
    }
    questions = parse_questions(tool_input)
    assert [q.question for q in questions] == [
        "Which database?",
        "Which features?",
    ]
    assert questions[0].multiSelect is False
    assert questions[1].multiSelect is True


def test_parse_empty_questions_list_raises() -> None:
    with pytest.raises(AskInputError):
        parse_questions({"questions": []})


def test_parse_missing_questions_key_raises() -> None:
    with pytest.raises(AskInputError):
        parse_questions({})


def test_parse_malformed_entry_raises() -> None:
    with pytest.raises(AskInputError):
        parse_questions({"questions": [{"header": "no question text"}]})


def test_parse_zero_options_raises() -> None:
    with pytest.raises(AskInputError):
        parse_questions(
            {"questions": [{"question": "q", "header": "H", "options": []}]}
        )


# ── render_questions ─────────────────────────────────────────────────────────


def test_render_headings_and_options() -> None:
    rendered = render_questions([SINGLE, MULTI])
    assert "## Question 1 (single-select): Which database? [Header]" in rendered
    assert "## Question 2 (multi-select): Which features? [Header]" in rendered
    assert "- PostgreSQL: PostgreSQL desc" in rendered
    assert "- billing: billing desc" in rendered


# ── validate_answers ─────────────────────────────────────────────────────────


def test_validate_single_select_label() -> None:
    call = answers_call(entry("Which database?", ["PostgreSQL"]))
    assert validate_answers(call, [SINGLE]) == {
        "Which database?": "PostgreSQL"
    }


def test_validate_multiselect_list() -> None:
    call = answers_call(entry("Which features?", ["auth", "search"]))
    assert validate_answers(call, [MULTI]) == {
        "Which features?": ["auth", "search"]
    }


def test_validate_free_text_stripped() -> None:
    call = answers_call(entry("Which database?", free_text="  use SQLite  "))
    assert validate_answers(call, [SINGLE]) == {
        "Which database?": "use SQLite"
    }


def test_validate_missing_question_raises() -> None:
    call = answers_call(entry("Which database?", ["PostgreSQL"]))
    with pytest.raises(AnswerValidationError):
        validate_answers(call, [SINGLE, MULTI])


def test_validate_extra_question_raises() -> None:
    call = answers_call(
        entry("Which database?", ["PostgreSQL"]),
        entry("Which cache?", ["redis"]),
    )
    with pytest.raises(AnswerValidationError):
        validate_answers(call, [SINGLE])


def test_validate_duplicate_question_raises() -> None:
    call = answers_call(
        entry("Which database?", ["PostgreSQL"]),
        entry("Which database?", ["MySQL"]),
    )
    with pytest.raises(AnswerValidationError):
        validate_answers(call, [SINGLE])


def test_validate_both_fields_set_raises() -> None:
    call = answers_call(
        entry("Which database?", ["PostgreSQL"], free_text="also this")
    )
    with pytest.raises(AnswerValidationError) as exc_info:
        validate_answers(call, [SINGLE])
    assert "Which database?" in str(exc_info.value)


def test_validate_neither_field_set_raises() -> None:
    call = answers_call(entry("Which database?"))
    with pytest.raises(AnswerValidationError) as exc_info:
        validate_answers(call, [SINGLE])
    assert "Which database?" in str(exc_info.value)


def test_validate_unknown_label_raises() -> None:
    call = answers_call(entry("Which database?", ["SQLite"]))
    with pytest.raises(AnswerValidationError) as exc_info:
        validate_answers(call, [SINGLE])
    assert "Which database?" in str(exc_info.value)
    assert "SQLite" in str(exc_info.value)


def test_validate_duplicate_labels_raise() -> None:
    call = answers_call(entry("Which features?", ["auth", "auth"]))
    with pytest.raises(AnswerValidationError) as exc_info:
        validate_answers(call, [MULTI])
    assert "Which features?" in str(exc_info.value)


def test_validate_two_selections_on_single_select_raises() -> None:
    call = answers_call(entry("Which database?", ["PostgreSQL", "MySQL"]))
    with pytest.raises(AnswerValidationError) as exc_info:
        validate_answers(call, [SINGLE])
    assert "Which database?" in str(exc_info.value)


# ── ask_once ─────────────────────────────────────────────────────────────────


async def test_ask_once_unknown_tool_raises() -> None:
    fake = FakeLLM()
    await fake.results.put(ToolCall(name="surprise", input={}))
    with pytest.raises(LLMCallError):
        await ask_once(
            fake, MODEL_CFG, intent="i", events=[], questions=[SINGLE],
            prior_error=None,
        )


async def test_ask_once_invalid_tool_input_raises() -> None:
    fake = FakeLLM()
    await fake.results.put(
        ToolCall(name="answer_questions", input={"reasoning": "thin"})
    )
    with pytest.raises(LLMCallError):
        await ask_once(
            fake, MODEL_CFG, intent="i", events=[], questions=[SINGLE],
            prior_error=None,
        )


async def test_ask_once_escalate_passes_through() -> None:
    fake = FakeLLM()
    await fake.results.put(ToolCall(name="escalate", input=ESCALATE_INPUT))
    result = await ask_once(
        fake, MODEL_CFG, intent="i", events=[], questions=[SINGLE], prior_error=None
    )
    assert isinstance(result, EscalateCall)
    assert result.situation == "the menu picks a production database"


async def test_ask_once_prior_error_in_working_block() -> None:
    fake = FakeLLM()
    await fake.results.put(ToolCall(name="escalate", input=ESCALATE_INPUT))
    await ask_once(
        fake, MODEL_CFG, intent="i", events=[], questions=[SINGLE],
        prior_error="unknown label 'SQLite'",
    )
    working = working_text(fake.calls[0])
    assert "unknown label 'SQLite'" in working
    assert "Which database?" in working


# ── decide_questions ─────────────────────────────────────────────────────────


async def test_decide_valid_first_try() -> None:
    fake = FakeLLM()
    await fake.results.put(
        answers_tool_call(entry("Which database?", ["PostgreSQL"]))
    )
    result = await decide_questions(
        fake, MODEL_CFG, intent="i", events=[], questions=[SINGLE]
    )
    assert not isinstance(result, EscalateCall)
    call, answers = result
    assert call.reasoning == "grounded"
    assert answers == {"Which database?": "PostgreSQL"}
    assert len(fake.calls) == 1


async def test_decide_invalid_then_valid_retries_once() -> None:
    fake = FakeLLM()
    await fake.results.put(
        answers_tool_call(entry("Which database?", ["SQLite"]))
    )
    await fake.results.put(
        answers_tool_call(entry("Which database?", ["PostgreSQL"]))
    )
    result = await decide_questions(
        fake, MODEL_CFG, intent="i", events=[], questions=[SINGLE]
    )
    assert not isinstance(result, EscalateCall)
    _, answers = result
    assert answers == {"Which database?": "PostgreSQL"}
    assert len(fake.calls) == 2
    assert "SQLite" in working_text(fake.calls[1])


async def test_decide_invalid_twice_raises() -> None:
    fake = FakeLLM()
    await fake.results.put(
        answers_tool_call(entry("Which database?", ["SQLite"]))
    )
    await fake.results.put(
        answers_tool_call(entry("Which database?", ["SQLite"]))
    )
    with pytest.raises(AnswerValidationError):
        await decide_questions(
            fake, MODEL_CFG, intent="i", events=[], questions=[SINGLE]
        )
    assert len(fake.calls) == 2


async def test_decide_escalate_on_retry_returns_escalation() -> None:
    fake = FakeLLM()
    await fake.results.put(
        answers_tool_call(entry("Which database?", ["SQLite"]))
    )
    await fake.results.put(ToolCall(name="escalate", input=ESCALATE_INPUT))
    result = await decide_questions(
        fake, MODEL_CFG, intent="i", events=[], questions=[SINGLE]
    )
    assert isinstance(result, EscalateCall)
    assert len(fake.calls) == 2
