"""Calibration case models: one triage case, one permission case, two sets.

Every model forbids extras, so a stray key in a hand-authored JSON file is a
typo that fails loud rather than a field that silently does nothing. Each set
carries a ``note`` because the recorded ``expected`` labels are the developer's
to confirm or flip — the seeded values are a starting point, not authority.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class QuestionCase(BaseModel):
    """One triage calibration case: a question and the recorded judgment."""

    model_config = ConfigDict(extra="forbid")

    id: str
    intent: str
    last_message: str
    event_name: str = "Stop"
    expected: Literal["answer", "escalate"]
    rationale: str


class PermissionCase(BaseModel):
    """One permission calibration case: a tool call and the recorded judgment."""

    model_config = ConfigDict(extra="forbid")

    id: str
    intent: str
    tool_name: str
    tool_input: dict[str, Any]
    expected: Literal["allow", "escalate"]
    rationale: str


class QuestionSet(BaseModel):
    """The triage calibration file: a note plus its cases."""

    model_config = ConfigDict(extra="forbid")

    note: str
    cases: list[QuestionCase]


class PermissionSet(BaseModel):
    """The permission calibration file: a note plus its cases."""

    model_config = ConfigDict(extra="forbid")

    note: str
    cases: list[PermissionCase]
