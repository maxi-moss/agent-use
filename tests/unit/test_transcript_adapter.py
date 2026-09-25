"""Adapter determinism + real-fixture behaviour."""

import json
from pathlib import Path
from typing import Any

from broker.transcript.adapter import (
    ReadReport,
    read_cleaned,
    render,
)
from broker.transcript.schemas import (
    TRANSCRIPT_VALIDATED_AGAINST,
    AskUserAnswer,
    AskUserQuestionUse,
    ExitPlanModeUse,
    ExitPlanResult,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "transcripts"

MINIMAL_ASK = FIXTURES / "8753fe50-2884-4cb6-9728-ba9c1101b617.jsonl"
ASK_VARIANTS = FIXTURES / "71a46971-27ec-40fc-8379-ff5351eddf90.jsonl"
REJECTED_ASK = FIXTURES / "d4032982-9753-4cce-ac1a-589ee8fe7e19.jsonl"
EXIT_PLAN = FIXTURES / "4f98b564-4f25-4a62-bee0-7808a82cd868.jsonl"
MULTISELECT_ABSENT = FIXTURES / "3ec7ee94-4dd8-4e26-a8fa-6b047e3382e2.jsonl"
APPROVED_PLAN_SYNTHETIC = FIXTURES / "approved-exit-plan.jsonl"


def test_minimal_ask_user_question_answered() -> None:
    events, _ = read_cleaned(MINIMAL_ASK)
    questions = [e for e in events if isinstance(e, AskUserQuestionUse)]
    answers = [e for e in events if isinstance(e, AskUserAnswer)]
    assert len(questions) == 2
    assert len(answers) == 2
    by_id = {a.id: a for a in answers}
    answered = by_id[questions[0].id]
    assert answered.rejected is False
    assert 'answered: "Which database should this project use?"="PostgreSQL"' in (
        answered.raw
    )
    # single-select question shape
    assert questions[0].questions[0].multiSelect is False
    assert [o.label for o in questions[0].questions[0].options] == [
        "PostgreSQL",
        "MySQL",
    ]


def test_rejected_ask_user_question_is_error_path() -> None:
    events, _ = read_cleaned(REJECTED_ASK)
    answers = [e for e in events if isinstance(e, AskUserAnswer)]
    assert len(answers) == 1
    assert answers[0].rejected is True


def test_ask_prose_variants_pass_through_verbatim() -> None:
    """Multi-question batch and multiSelect comma-join answers arrive as raw prose."""
    events, _ = read_cleaned(ASK_VARIANTS)
    questions = [e for e in events if isinstance(e, AskUserQuestionUse)]
    answers = {a.id: a for a in events if isinstance(a, AskUserAnswer)}
    assert len(questions) == 3
    assert set(answers) == {q.id for q in questions}
    raws = [answers[q.id].raw for q in questions]
    # 2-question batch: two "q"="label" pairs joined by ", "
    assert raws[0].count('"=') == 2
    # multiSelect: several labels comma-joined inside one answer value
    assert "Bundle + Claude Code skills (Recommended), Auto-capture" in raws[1]
    # multiSelect 2-question batch
    assert raws[2].count('"=') == 2
    for raw in raws:
        assert raw.startswith("Your questions have been answered: ")


def test_structured_answers_on_answered_and_rejected() -> None:
    events, _ = read_cleaned(MINIMAL_ASK)
    questions = [e for e in events if isinstance(e, AskUserQuestionUse)]
    answers = {a.id: a for a in events if isinstance(a, AskUserAnswer)}
    answered = answers[questions[0].id]
    assert answered.answers == {
        "Which database should this project use?": "PostgreSQL"
    }
    rejected = answers[questions[1].id]
    assert rejected.rejected is True
    assert rejected.answers is None


def test_native_multiselect_answer_is_joined_str() -> None:
    """A native-UI multiSelect answer arrives as one comma-joined str, not a list."""
    events, _ = read_cleaned(ASK_VARIANTS)
    questions = [e for e in events if isinstance(e, AskUserQuestionUse)]
    answers = {a.id: a for a in events if isinstance(a, AskUserAnswer)}
    multi = answers[questions[1].id]
    assert multi.answers is not None
    (value,) = multi.answers.values()
    assert isinstance(value, str)
    assert value == (
        "Bundle + Claude Code skills (Recommended), Auto-capture from "
        "Gmail/Slack/Jira"
    )


def test_rejected_ask_answers_is_none() -> None:
    events, _ = read_cleaned(REJECTED_ASK)
    answers = [e for e in events if isinstance(e, AskUserAnswer)]
    assert len(answers) == 1
    assert answers[0].rejected is True
    assert answers[0].answers is None


def test_ill_typed_answer_entries_dropped_entry_wise(tmp_path: Path) -> None:
    question_record: dict[str, Any] = {
        "type": "assistant",
        "version": TRANSCRIPT_VALIDATED_AGAINST,
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": "AskUserQuestion",
                    "id": "toolu_mixed",
                    "input": {
                        "questions": [
                            {"question": "a", "header": "A", "options": []}
                        ]
                    },
                }
            ]
        },
    }
    answer_record: dict[str, Any] = {
        "type": "user",
        "version": TRANSCRIPT_VALIDATED_AGAINST,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_mixed",
                    "content": "answered",
                }
            ],
        },
        "toolUseResult": {
            "answers": {"a": "x", "b": ["y", "z"], "c": 7}
        },
    }
    p = tmp_path / "mixed-answers.jsonl"
    p.write_text(
        json.dumps(question_record) + "\n" + json.dumps(answer_record) + "\n"
    )
    events, _ = read_cleaned(p)
    answers = [e for e in events if isinstance(e, AskUserAnswer)]
    assert len(answers) == 1
    assert answers[0].answers == {"a": "x", "b": ["y", "z"]}


def test_multiselect_absent_defaults_false() -> None:
    events, _ = read_cleaned(MULTISELECT_ABSENT)
    questions = [e for e in events if isinstance(e, AskUserQuestionUse)]
    flags = [q.multiSelect for e in questions for q in e.questions]
    assert False in flags  # at least one record lacked the key -> default


def test_exit_plan_mode_inputs_and_rejected_results() -> None:
    events, _ = read_cleaned(EXIT_PLAN)
    plans = [e for e in events if isinstance(e, ExitPlanModeUse)]
    results = {r.id: r for r in events if isinstance(r, ExitPlanResult)}
    assert len(plans) == 3
    for plan in plans:
        assert plan.plan  # non-empty plan text
        assert plan.plan_file_path is not None
        assert plan.plan_file_path.endswith(".md")
        assert results[plan.id].rejected is True  # only rejected samples exist


def test_approved_exit_plan_synthetic_fixture() -> None:
    """SYNTHETIC shape — replaced by the real developer capture."""
    events, _ = read_cleaned(APPROVED_PLAN_SYNTHETIC)
    results = [e for e in events if isinstance(e, ExitPlanResult)]
    assert len(results) == 1
    assert results[0].rejected is False


def test_render_sections() -> None:
    events, _ = read_cleaned(MINIMAL_ASK)
    rendered = render(events)
    assert "## user\n" in rendered
    assert "## assistant\n" in rendered
    assert "## question (id=" in rendered
    assert "## answer (id=" in rendered
    assert "- PostgreSQL: " in rendered


def test_version_collected_and_warning_shape() -> None:
    _, report = read_cleaned(MINIMAL_ASK)
    assert isinstance(report, ReadReport)
    assert TRANSCRIPT_VALIDATED_AGAINST in report.versions
    # this fixture is pure 2.1.220 -> no version warning
    assert report.warnings == []


def test_unknown_types_counted_not_fatal() -> None:
    _, report = read_cleaned(MINIMAL_ASK)
    assert report.unknown_types["ai-title"] > 0
    assert "assistant" not in report.unknown_types
    assert "user" not in report.unknown_types
