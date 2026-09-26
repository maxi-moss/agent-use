"""Strict-tool schema derivation pins (drift pin vs EscalationDisclosure)."""

import hashlib
import json
from typing import Any, cast

import pytest

from broker.protocol.schemas import EscalationDisclosure
from broker.session.ask import ASK_TOOLS
from broker.session.clarify import CLARIFY_TOOLS
from broker.session.grounding import GROUNDING_TOOLS
from broker.session.llm_stack import EscalateCall
from broker.session.triage import TRIAGE_TOOLS

ALL_TOOLS = TRIAGE_TOOLS + GROUNDING_TOOLS + ASK_TOOLS


def _walk_objects(node: Any, path: str = "$") -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        typed = cast(dict[str, Any], node)
        if typed.get("type") == "object" or "properties" in typed:
            found.append((path, typed))
        for key, value in typed.items():
            found.extend(_walk_objects(value, f"{path}.{key}"))
    elif isinstance(node, list):
        for i, item in enumerate(cast(list[Any], node)):
            found.extend(_walk_objects(item, f"{path}[{i}]"))
    return found


def test_all_tools_are_strict_and_fully_required() -> None:
    assert len(ALL_TOOLS) == 7
    for tool in ALL_TOOLS:
        assert tool.get("strict") is True, f"{tool['name']} not strict"
        schema = cast(dict[str, Any], tool["input_schema"])
        objects = _walk_objects(schema)
        assert objects, f"{tool['name']} has no object schema"
        for path, obj in objects:
            assert obj.get("additionalProperties") is False, (
                f"{tool['name']} {path}: additionalProperties not false"
            )
            props = cast(dict[str, Any], obj.get("properties", {}))
            assert sorted(cast(list[str], obj.get("required", []))) == sorted(
                props.keys()
            ), f"{tool['name']} {path}: not every property is required"


def test_no_ref_cycles() -> None:
    """Strict tools reject recursive schemas — no $def may reference itself."""
    for tool in ALL_TOOLS:
        schema = cast(dict[str, Any], tool["input_schema"])
        defs = cast(dict[str, Any], schema.get("$defs", {}))
        for name, definition in defs.items():
            assert f"#/$defs/{name}" not in json.dumps(definition), (
                f"{tool['name']}: recursive $def {name}"
            )


def test_escalate_tool_fields_match_escalation_disclosure() -> None:
    """The tool input minus broker-only fields == the wire disclosure. Drift pin."""
    tool_fields = set(EscalateCall.model_fields) - {"reasoning", "task_summary"}
    assert tool_fields == set(EscalationDisclosure.model_fields)


def test_task_activity_on_non_escalate_triage_tools_only() -> None:
    for tool in TRIAGE_TOOLS:
        schema = cast(dict[str, Any], tool["input_schema"])
        props = cast(dict[str, Any], schema.get("properties", {}))
        if tool["name"] == "escalate":
            assert "task_activity" not in props
        else:
            assert "task_activity" in props


def test_task_summary_on_answer_escalate_complete_not_no_action() -> None:
    for tool in TRIAGE_TOOLS:
        schema = cast(dict[str, Any], tool["input_schema"])
        props = cast(dict[str, Any], schema.get("properties", {}))
        if tool["name"] == "no_action":
            assert "task_summary" not in props
        else:
            assert "task_summary" in props


def test_ask_and_triage_escalate_schemas_identical() -> None:
    """Both call sites derive escalate from the same EscalateCall. Drift pin."""
    triage_escalate = next(t for t in TRIAGE_TOOLS if t["name"] == "escalate")
    ask_escalate = next(t for t in ASK_TOOLS if t["name"] == "escalate")
    assert ask_escalate["input_schema"] == triage_escalate["input_schema"]


@pytest.mark.parametrize(
    "name,tools,expected_hash",
    [
        (
            "TRIAGE_TOOLS",
            TRIAGE_TOOLS,
            "fdc3d7d14fbc4f05c589d52447411519a0e8dadbcb10f70790fd26e98de89c00",
        ),
        (
            "GROUNDING_TOOLS",
            GROUNDING_TOOLS,
            "dfa68f06bf238ed5b9ca4efd439195e0188c547de316aaa18fc4bcc68c5f74d8",
        ),
        (
            "ASK_TOOLS",
            ASK_TOOLS,
            "4064d42c4dbbd34f056deee7b169c3ee059c716b969be9897e652a6328c470ca",
        ),
        (
            "CLARIFY_TOOLS",
            CLARIFY_TOOLS,
            "7922e2977f47e925989419ff12aab1c7de308116ba3c8b9a110940f63d6e8d42",
        ),
    ],
)
def test_tool_schema_pin(
    name: str, tools: list[dict[str, Any]], expected_hash: str
) -> None:
    digest = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()
    assert digest == expected_hash, (
        f"LLM-visible tool schema changed ({name}); review the dumped schema "
        "and update the pin"
    )
