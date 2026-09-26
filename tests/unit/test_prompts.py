"""Prompt loading + the static-prefix rule (cache determinism)."""

import re
from pathlib import Path

import pytest
from anthropic.types import ToolParam

from broker import prompts
from broker.master.llm import MASTER_TOOLS
from broker.permission.classifier import PERMISSION_TOOLS
from broker.session.ask import ASK_TOOLS
from broker.session.clarify import CLARIFY_TOOLS
from broker.session.grounding import GROUNDING_TOOLS
from broker.session.triage import TRIAGE_TOOLS

PROMPT_DIR = Path(__file__).parent.parent.parent / "src" / "broker" / "prompts"
NAMES = ["triage", "master", "grounding", "permission", "ask", "clarify"]

TOOL_REGISTRIES: list[tuple[str, list[ToolParam]]] = [
    ("triage", TRIAGE_TOOLS),
    ("ask", ASK_TOOLS),
    ("clarify", CLARIFY_TOOLS),
    ("grounding", GROUNDING_TOOLS),
    ("permission", PERMISSION_TOOLS),
    ("master", MASTER_TOOLS),
]


@pytest.mark.parametrize("name", NAMES)
def test_prompts_are_static_prefixes(name: str) -> None:
    """No templating slots: prompts must be byte-stable across calls
    (any volatile content belongs after the cache breakpoint)."""
    text = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    assert not re.search(r"\{[a-z_]+\}", text), "format-style slot in prompt"
    assert "{{" not in text
    assert prompts.load(name) == prompts.load(name)


@pytest.mark.parametrize(
    "prompt_key, registry", TOOL_REGISTRIES, ids=[key for key, _ in TOOL_REGISTRIES]
)
def test_prompt_names_every_registry_tool(
    prompt_key: str, registry: list[ToolParam]
) -> None:
    """Every tool name in a stack's registry must appear backticked in that
    stack's own prompt, so a renamed or added tool cannot go undocumented."""
    text = prompts.load(prompt_key)
    for tool in registry:
        name = tool["name"]
        assert re.search(rf"`{re.escape(name)}[`(]", text), name
