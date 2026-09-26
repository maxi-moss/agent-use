"""AskUserQuestion decisions: answer the agent's menu, or escalate it.

The session stack's second call site, beside triage. It shares triage's
escalate model from ``llm_stack`` on purpose — both raise the same wire
payload — and stays inside the session stack: nothing here touches the
permission or master stacks. Answers are option labels copied verbatim
(never indices), validated against the questions before anything leaves the
broker.

Class names, field names, `Field` descriptions and docstrings of these models
are sent to the model.
"""

from typing import Any, cast

from anthropic.types import ToolParam
from pydantic import BaseModel, ConfigDict, ValidationError

from broker import llm_timing
from broker import prompts
from broker.config import SessionModelConfig
from broker.llm import LLMCaller, LLMCallError, ToolCall, strict_tool
from broker.session.llm_stack import (
    FORCED_ONE,
    EscalateCall,
    assemble_context,
)
from broker.transcript.schemas import Question, TranscriptEvent


class QuestionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    selected: list[str]
    free_text: str


class AnswerQuestionsCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str
    answers: list[QuestionAnswer]


AskResult = AnswerQuestionsCall | EscalateCall

# The wire value type: label str (single-select), list of labels
# (multiSelect), or free text str. Matches updatedInput.answers exactly.
AnswerValue = str | list[str]


class AskInputError(Exception):
    """The AskUserQuestion tool_input itself is unusable. Escalate, never guess."""


class AnswerValidationError(Exception):
    """The model's answers do not fit the questions asked."""


ASK_TOOLS: list[ToolParam] = [
    strict_tool(
        "answer_questions",
        "Answer the coding agent's menu yourself. Answer every question:"
        " copy the question text exactly, pick listed option labels verbatim"
        " in `selected`, or write a custom reply in `free_text` when no"
        " listed option is right — never both on one question. Use when"
        " well-supported choices follow from the stated intent and the"
        " conversation.",
        AnswerQuestionsCall,
    ),
    strict_tool(
        "escalate",
        "Show the menu to the developer instead. Use when a choice is"
        " irreversible or high blast-radius, architecturally significant, or"
        " you cannot ground one. Every field must carry real analysis.",
        EscalateCall,
    ),
]

_TOOL_MODELS: dict[str, type[AskResult]] = {
    "answer_questions": AnswerQuestionsCall,
    "escalate": EscalateCall,
}

_ASK_PROMPT = prompts.load("ask")


def parse_questions(tool_input: dict[str, Any]) -> list[Question]:
    """Validate AskUserQuestion tool input into questions, strictly.

    Args:
        tool_input: Raw tool input from the hook payload.

    Returns:
        The validated questions, in menu order.

    Raises:
        AskInputError: The input has no questions or any entry is malformed.
            Answering demands a fully-understood menu, so this rejects any
            malformed entry rather than skipping it as display-only rendering
            may.
    """
    raw_questions = tool_input.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise AskInputError("tool_input carries no questions")
    questions: list[Question] = []
    for raw_question in cast(list[Any], raw_questions):
        try:
            question = Question.model_validate(raw_question)
        except ValidationError as exc:
            raise AskInputError(f"malformed question entry: {exc}") from exc
        if not question.options:
            raise AskInputError(f"question {question.question!r} has no options")
        questions.append(question)
    return questions


def render_questions(questions: list[Question]) -> str:
    """Render the menu for the working block of the ask call."""
    lines: list[str] = []
    for number, question in enumerate(questions, start=1):
        mode = "multi-select" if question.multiSelect else "single-select"
        lines.append(
            f"## Question {number} ({mode}): {question.question} "
            f"[{question.header}]"
        )
        for option in question.options:
            lines.append(f"- {option.label}: {option.description}")
    return "\n".join(lines)


def validate_answers(
    call: AnswerQuestionsCall, questions: list[Question]
) -> dict[str, AnswerValue]:
    """Check the model's answers against the questions and build the wire dict.

    Args:
        call: The model's ``answer_questions`` call.
        questions: The validated questions being answered.

    Returns:
        The ``updatedInput.answers`` mapping: label for single-select, list of
        labels for multiSelect, free text verbatim.

    Raises:
        AnswerValidationError: Coverage, exclusivity, label membership, or
            cardinality is wrong. Never clamped — a wrong answer silently
            delivered is the one unrecoverable failure.
    """
    by_text = {question.question: question for question in questions}
    given = [answer.question for answer in call.answers]
    if sorted(given) != sorted(by_text):
        raise AnswerValidationError(
            f"answers must cover every question exactly once; asked "
            f"{sorted(by_text)}, got {sorted(given)}"
        )
    out: dict[str, AnswerValue] = {}
    for answer in call.answers:
        question = by_text[answer.question]
        free_text = answer.free_text.strip()
        if bool(answer.selected) == bool(free_text):
            raise AnswerValidationError(
                f"question {answer.question!r}: exactly one of `selected` or "
                "`free_text` must be used"
            )
        if free_text:
            out[answer.question] = free_text
            continue
        labels = [option.label for option in question.options]
        unknown = [label for label in answer.selected if label not in labels]
        if unknown:
            raise AnswerValidationError(
                f"question {answer.question!r}: {unknown!r} not among the "
                f"listed options {labels!r}"
            )
        if len(set(answer.selected)) != len(answer.selected):
            raise AnswerValidationError(
                f"question {answer.question!r}: duplicate selections"
            )
        if question.multiSelect:
            out[answer.question] = list(answer.selected)
        else:
            if len(answer.selected) != 1:
                raise AnswerValidationError(
                    f"question {answer.question!r} is single-select; got "
                    f"{len(answer.selected)} selections"
                )
            out[answer.question] = answer.selected[0]
    return out


@llm_timing.timed("ask")
async def ask_once(
    llm_call: LLMCaller[ToolCall],
    model_cfg: SessionModelConfig,
    *,
    intent: str,
    events: list[TranscriptEvent],
    questions: list[Question],
    prior_error: str | None,
) -> AskResult:
    """Make one answer-or-escalate call over a pending menu.

    Args:
        llm_call: The injected tool-calling seam.
        model_cfg: Supplies the model id and the token cap.
        intent: Authoritative task intent, taken from the registry.
        events: Cleaned transcript events, as surrounding context.
        questions: The menu being decided.
        prior_error: Validation error from a previous attempt, fed back for
            the one retry; ``None`` on the first attempt.

    Returns:
        The validated call model for the tool the LLM chose.

    Raises:
        LLMCallError: The LLM called an unknown tool, or the tool input
            failed validation.
    """
    working = (
        "# The coding agent is asking questions via AskUserQuestion "
        "(decide THIS: answer or escalate)\n" + render_questions(questions)
    )
    if prior_error is not None:
        working += (
            "\n\n# Your previous answer_questions call was invalid — "
            "correct this\n" + prior_error
        )
    system, messages = assemble_context(_ASK_PROMPT, intent, events, working)
    call: ToolCall = await llm_call(
        model=model_cfg.model_id,
        max_tokens=model_cfg.max_tokens,
        system=system,
        messages=messages,
        tools=ASK_TOOLS,
        tool_choice=FORCED_ONE,
    )
    model = _TOOL_MODELS.get(call.name)
    if model is None:
        raise LLMCallError(f"unknown ask tool {call.name!r}")
    try:
        return model.model_validate(call.input)
    except ValidationError as exc:
        raise LLMCallError(f"invalid {call.name} input: {exc}") from exc


async def decide_questions(
    llm_call: LLMCaller[ToolCall],
    model_cfg: SessionModelConfig,
    *,
    intent: str,
    events: list[TranscriptEvent],
    questions: list[Question],
) -> tuple[AnswerQuestionsCall, dict[str, AnswerValue]] | EscalateCall:
    """Decide a pending menu, retrying a failed validation exactly once.

    Args:
        llm_call: The injected tool-calling seam.
        model_cfg: Supplies the model id and the token cap.
        intent: Authoritative task intent, taken from the registry.
        events: Cleaned transcript events, as surrounding context.
        questions: The menu being decided.

    Returns:
        The answer call with its validated wire answers, or the escalation.

    Raises:
        LLMCallError: An underlying call failed.
        AnswerValidationError: Both attempts produced invalid answers.
    """
    result = await ask_once(
        llm_call, model_cfg, intent=intent, events=events, questions=questions,
        prior_error=None,
    )
    if isinstance(result, EscalateCall):
        return result
    try:
        return result, validate_answers(result, questions)
    except AnswerValidationError as exc:
        retry = await ask_once(
            llm_call, model_cfg, intent=intent, events=events, questions=questions,
            prior_error=str(exc),
        )
        if isinstance(retry, EscalateCall):
            return retry
        return retry, validate_answers(retry, questions)
