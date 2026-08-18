"""Prompt loading + the static-prefix rule (cache determinism)."""

import re
from pathlib import Path

import pytest

from broker import prompts

PROMPT_DIR = Path(__file__).parent.parent.parent / "src" / "broker" / "prompts"
NAMES = ["triage", "master", "grounding", "permission", "ask"]


@pytest.mark.parametrize("name", NAMES)
def test_prompts_are_static_prefixes(name: str) -> None:
    """No templating slots: prompts must be byte-stable across calls
    (any volatile content belongs after the cache breakpoint)."""
    text = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    assert not re.search(r"\{[a-z_]+\}", text), "format-style slot in prompt"
    assert "{{" not in text
    assert prompts.load(name) == prompts.load(name)


def test_triage_prompt_names_all_four_tools() -> None:
    text = prompts.load("triage")
    for tool in ("answer", "escalate", "complete", "no_action"):
        assert f"`{tool}`" in text


def test_ask_prompt_names_both_tools() -> None:
    text = prompts.load("ask")
    for tool in ("answer_questions", "escalate"):
        assert f"`{tool}`" in text
