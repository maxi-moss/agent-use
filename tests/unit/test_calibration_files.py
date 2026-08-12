"""Offline schema guard for the calibration files: no live calls.

The runner reports each case by its id, so a duplicate id would make its output
ambiguous; that is worth catching here rather than discovering it live.
"""

from pathlib import Path

import pytest

from broker.calibration.schemas import PermissionSet, QuestionSet

pytestmark = pytest.mark.unit

CALIBRATION_DIR = Path(__file__).parent.parent.parent / "calibration-cases"


def test_question_file_validates() -> None:
    text = (CALIBRATION_DIR / "questions.json").read_text(encoding="utf-8")
    qset = QuestionSet.model_validate_json(text)
    assert qset.cases


def test_permission_file_validates() -> None:
    text = (CALIBRATION_DIR / "permissions.json").read_text(encoding="utf-8")
    pset = PermissionSet.model_validate_json(text)
    assert pset.cases


def test_case_ids_unique() -> None:
    qset = QuestionSet.model_validate_json(
        (CALIBRATION_DIR / "questions.json").read_text(encoding="utf-8")
    )
    pset = PermissionSet.model_validate_json(
        (CALIBRATION_DIR / "permissions.json").read_text(encoding="utf-8")
    )
    q_ids = [c.id for c in qset.cases]
    p_ids = [c.id for c in pset.cases]
    assert len(q_ids) == len(set(q_ids))
    assert len(p_ids) == len(set(p_ids))
