"""Public transcript event models. No JSONL field names here — raw.py owns those.

These rely on pydantic's default extra="ignore" — never set extra="forbid": an
unknown key is a newer Claude Code writing the transcript, not a typo.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

VALIDATED_AGAINST = "2.1.220"


class Option(BaseModel):
    label: str
    description: str = ""


class Question(BaseModel):
    question: str
    header: str
    options: list[Option] = Field(default_factory=list[Option])
    # 3/200 real 2.1.220 samples lack the key — the default is load-bearing
    multiSelect: bool = False


class UserPrompt(BaseModel):
    kind: Literal["user_prompt"]
    text: str


class AssistantText(BaseModel):
    kind: Literal["assistant_text"]
    text: str


class AskUserQuestion(BaseModel):
    kind: Literal["ask_user_question"]
    id: str
    questions: list[Question]


class AskUserAnswer(BaseModel):
    kind: Literal["ask_user_answer"]
    id: str
    raw: str
    rejected: bool = False
    # Structured answers from the transcript record (2.1.234): label str for
    # single-select, list[str] when injected via updatedInput multiSelect,
    # comma-joined str when answered in the native UI. None when rejected or
    # on records that predate the field. Comparisons are structural — the
    # prose in `raw` has two templates and is never parsed.
    answers: dict[str, str | list[str]] | None = None


class ExitPlanMode(BaseModel):
    kind: Literal["exit_plan_mode"]
    id: str
    plan: str
    plan_file_path: str | None = None


class ExitPlanResult(BaseModel):
    # Shape PROVISIONAL for the approved path — only rejected samples exist
    # on this machine as of 2026-07-27.
    kind: Literal["exit_plan_result"]
    id: str
    raw: str
    rejected: bool = False


class CompactionBoundary(BaseModel):
    # Shape PROVISIONAL pending a real captured /compact record.
    kind: Literal["compaction_boundary"]


TranscriptEvent = Annotated[
    UserPrompt
    | AssistantText
    | AskUserQuestion
    | AskUserAnswer
    | ExitPlanMode
    | ExitPlanResult
    | CompactionBoundary,
    Field(discriminator="kind"),
]

# Module-level by requirement: per-call construction rebuilds the core schema.
# The alias is a type, not a model — it has no .model_validate.
_EVENT: TypeAdapter[TranscriptEvent] = TypeAdapter(TranscriptEvent)

# Public name for consumers (adapter, tests).
EVENT_ADAPTER = _EVENT
