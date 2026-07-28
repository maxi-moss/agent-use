"""Strict-tool schema derivation pins (drift pin vs EscalationPayload)."""

import json
from typing import Any, cast

from broker.protocol.schemas import EscalationPayload
from broker.session.triage import (
    GROUNDING_TOOLS,
    TRIAGE_TOOLS,
    EscalateCall,
)

ALL_TOOLS = TRIAGE_TOOLS + GROUNDING_TOOLS


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
    assert len(ALL_TOOLS) == 5
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


def test_escalate_tool_fields_match_escalation_payload() -> None:
    """The tool input + broker-filled fields == the wire payload. Drift pin."""
    tool_fields = set(EscalateCall.model_fields) - {"reasoning"}
    payload_fields = set(EscalationPayload.model_fields) - {
        "escalation_id",
        "session_id",
        "task_context",
    }
    assert tool_fields == payload_fields
