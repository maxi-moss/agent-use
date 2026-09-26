#!/usr/bin/env python
"""Offline calibration runner — RUN MANUALLY, makes real API calls:

    uv run python scripts/calibrate.py                 # both files
    uv run python scripts/calibrate.py calibration-cases/questions.json
    uv run python scripts/calibrate.py calibration-cases/permissions.json

Feeds each calibration case through the real production triage / permission
seams against the pinned models and prints, per case, the model's decision and
its reasoning beside the recorded judgment, plus the aggregate escalation rate.

This is a tuning aid, not a gate. The "right" decision on a case is a human
judgment, so nothing here asserts the model matches the recorded label; a
MISMATCH is a prompt to look, not a failure.
"""

import asyncio
import functools
import os
import sys
import textwrap
from pathlib import Path

from broker.calibration.schemas import (
    PermissionCase,
    PermissionSet,
    QuestionCase,
    QuestionSet,
)
from broker.config import BrokerConfig, ClassifierConfig, SessionModelConfig
from broker.llm import LLMCallError, build_client, call_tool
from broker.permission.classifier import (
    AllowCall,
    PermissionCallError,
    PermissionEscalateCall,
    bind,
    build_classifier_client,
    classify,
    render_suggestions,
)
from broker.session.llm_stack import EscalateCall
from broker.session.triage import AnswerCall, CompleteCall, NoActionCall, triage

CALIBRATION_DIR = Path(__file__).parent.parent / "calibration-cases"

_QUESTION_DECISION = {
    AnswerCall: "answer",
    EscalateCall: "escalate",
    CompleteCall: "complete",
    NoActionCall: "no_action",
}
_PERMISSION_DECISION = {
    AllowCall: "allow",
    PermissionEscalateCall: "escalate",
}


def _print_row(case_id: str, expected: str, decision: str, reasoning: str) -> None:
    """Print one case's decision beside its recorded judgment."""
    match = "MATCH   " if decision == expected else "MISMATCH"
    print(f"  [{match}] {case_id}  expected={expected}  decision={decision}")
    if reasoning:
        wrapped = textwrap.fill(
            reasoning, width=88, initial_indent=" " * 12, subsequent_indent=" " * 12
        )
        print(wrapped)


async def run_questions(cases: list[QuestionCase]) -> None:
    """Run the triage leg against the pinned session model."""
    cfg = BrokerConfig()
    model_cfg = SessionModelConfig(model_id=cfg.model_id, max_tokens=cfg.max_tokens)
    caller = functools.partial(call_tool, build_client())
    print(f"questions ({model_cfg.model_id}): {len(cases)} cases")
    escalations = 0
    for case in cases:
        try:
            result = await triage(
                caller,
                model_cfg,
                intent=case.intent,
                events=[],
                last_assistant_message=case.last_message,
            )
            decision = _QUESTION_DECISION.get(type(result), type(result).__name__)
            reasoning = result.reasoning
        except LLMCallError as exc:
            decision, reasoning = f"ERROR: {exc}", ""
        escalations += decision == "escalate"
        _print_row(case.id, case.expected, decision, reasoning)
    print(f"  escalation rate: {escalations}/{len(cases)}\n")


async def run_permissions(cases: list[PermissionCase]) -> None:
    """Run the permission leg against the pinned classifier model."""
    cfg = ClassifierConfig()
    caller = bind(build_classifier_client())
    print(f"permissions ({cfg.model_id}): {len(cases)} cases")
    escalations = 0
    for case in cases:
        try:
            result = await classify(
                caller,
                cfg,
                intent=case.intent,
                tool_name=case.tool_name,
                tool_input=case.tool_input,
                suggestions=render_suggestions([]),
            )
            decision = _PERMISSION_DECISION.get(type(result), type(result).__name__)
            reasoning = result.reasoning
        except PermissionCallError as exc:
            decision, reasoning = f"ERROR: {exc}", ""
        escalations += decision == "escalate"
        _print_row(case.id, case.expected, decision, reasoning)
    print(f"  escalation rate: {escalations}/{len(cases)}\n")


async def run_file(path: Path) -> None:
    """Dispatch one calibration file to the leg its cases belong to."""
    text = path.read_text(encoding="utf-8")
    if "permission" in path.name:
        await run_permissions(PermissionSet.model_validate_json(text).cases)
    else:
        await run_questions(QuestionSet.model_validate_json(text).cases)


async def main(argv: list[str]) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 1
    if argv:
        await run_file(Path(argv[0]))
    else:
        await run_file(CALIBRATION_DIR / "questions.json")
        await run_file(CALIBRATION_DIR / "permissions.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1:])))
