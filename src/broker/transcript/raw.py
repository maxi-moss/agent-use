"""ALL JSONL field-name knowledge lives here.

Importable only by broker.transcript.adapter (import-linter contract) and the
only module allowed to contain raw JSONL key literals (scripts/check_jsonl_literals.sh).

Stateless: maps one parsed record to zero or more (kind, mapped dict) pairs.
Pairing tool_results back to the AskUserQuestion / ExitPlanMode calls they answer
is the adapter's job — this module never holds cross-record state.
"""

from typing import Any, cast

# Keep-known rule: only these top-level record types are recognised.
# Everything else — including future types — is discarded, counted, never fatal.
KNOWN_RECORD_TYPES = frozenset({"assistant", "user"})

# Internal (non-public) kind emitted for user tool_result blocks; the adapter
# pairs it to a previously seen AskUserQuestion / ExitPlanMode id or strips it.
KIND_TOOL_RESULT = "tool_result"

# The AskUserQuestion answer arrives as this prose (verified against 2.1.220):
#   Your questions have been answered: "<q>"="<label>"[, ...]. You can now
#   continue with these answers in mind.
# It is exposed verbatim as `raw`; consumers never string-match it. The only
# parsing the public surface needs is answer-vs-denial, below.
_REJECTED_TOOL_USE_RESULT = "User rejected tool use"
_REJECTED_DENIAL_KIND = "user-rejected"


def record_type(obj: dict[str, Any]) -> str | None:
    """Return the record's top-level type, or ``None`` when absent or non-string."""
    t = obj.get("type")
    return t if isinstance(t, str) else None


def record_version(obj: dict[str, Any]) -> str | None:
    """Return the Claude Code version stamped on the record, if it carries one."""
    v = obj.get("version")
    return v if isinstance(v, str) else None


def map_line(obj: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Map one parsed JSONL record to (kind, mapped dict) pairs.

    Args:
        obj: One parsed JSONL record.

    Returns:
        Zero or more (kind, mapped dict) pairs; ``[]`` for records to discard.
    """
    rtype = record_type(obj)
    if rtype == "assistant":
        return _map_assistant(obj)
    if rtype == "user":
        return _map_user(obj)
    return []


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Return ``value`` as a dict, or ``None`` when it is not one."""
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return None


def _as_list(value: Any) -> list[Any] | None:
    """Return ``value`` as a list, or ``None`` when it is not one."""
    if isinstance(value, list):
        return cast(list[Any], value)
    return None


def _message_content(obj: dict[str, Any]) -> Any:
    """Return the record's message content, or ``None`` when it has no message."""
    message = _as_dict(obj.get("message"))
    if message is None:
        return None
    return message.get("content")


def _map_assistant(obj: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Map an assistant record's content blocks to (kind, mapped dict) pairs.

    Args:
        obj: One parsed assistant record.

    Returns:
        One pair per block that survived the filter; ``[]`` if none did.
    """
    content = _as_list(_message_content(obj))
    if content is None:
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for raw_block in content:
        block = _as_dict(raw_block)
        if block is None:
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text")
            if isinstance(text, str):
                out.append(("assistant_text", {"kind": "assistant_text", "text": text}))
        elif btype == "tool_use":
            mapped = _map_tool_use(block)
            if mapped is not None:
                out.append(mapped)
        # thinking blocks (empty text + opaque signature) and every other
        # block type are implementation detail — stripped.
    return out


def _map_tool_use(block: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Map one assistant tool_use block to a public event, if it is one we keep.

    Args:
        block: One ``tool_use`` content block from an assistant record.

    Returns:
        The (kind, mapped dict) pair, or ``None`` if not kept, or malformed.
    """
    name = block.get("name")
    block_id = block.get("id")
    tool_input = _as_dict(block.get("input"))
    if not isinstance(block_id, str) or tool_input is None:
        return None
    if name == "AskUserQuestion":
        questions = _as_list(tool_input.get("questions"))
        if questions is None:
            return None
        return (
            "ask_user_question",
            {"kind": "ask_user_question", "id": block_id, "questions": questions},
        )
    if name == "ExitPlanMode":
        plan = tool_input.get("plan")
        if not isinstance(plan, str):
            return None
        plan_file_path = tool_input.get("planFilePath")
        return (
            "exit_plan_mode",
            {
                "kind": "exit_plan_mode",
                "id": block_id,
                "plan": plan,
                "plan_file_path": plan_file_path
                if isinstance(plan_file_path, str)
                else None,
            },
        )
    # Read, Edit, Bash, Grep, ... — implementation tool calls are stripped.
    return None


def _map_user(obj: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Map a user record to a prompt event or to internal tool_result pairs.

    Args:
        obj: One parsed user record.

    Returns:
        Zero or more (kind, mapped dict) pairs; ``[]`` if neither.
    """
    content = _message_content(obj)
    if isinstance(content, str):
        # Typed developer prompts carry origin.kind == "human" (and a
        # promptSource); string-content records without it are local-command
        # wrappers and other noise — not prompts.
        origin = _as_dict(obj.get("origin"))
        if origin is not None and origin.get("kind") == "human":
            return [("user_prompt", {"kind": "user_prompt", "text": content})]
        return []
    blocks = _as_list(content)
    if blocks is None:
        return []
    rejected_record = (
        obj.get("toolDenialKind") == _REJECTED_DENIAL_KIND
        or obj.get("toolUseResult") == _REJECTED_TOOL_USE_RESULT
    )
    out: list[tuple[str, dict[str, Any]]] = []
    for raw_block in blocks:
        block = _as_dict(raw_block)
        if block is None or block.get("type") != "tool_result":
            continue
        tool_use_id = block.get("tool_use_id")
        if not isinstance(tool_use_id, str):
            continue
        out.append(
            (
                KIND_TOOL_RESULT,
                {
                    "tool_use_id": tool_use_id,
                    "raw": _content_text(block.get("content")),
                    "rejected": bool(block.get("is_error")) or rejected_record,
                },
            )
        )
    return out


def _content_text(content: Any) -> str:
    """Flatten tool_result content: a plain string, or a list of text blocks."""
    if isinstance(content, str):
        return content
    blocks = _as_list(content)
    if blocks is not None:
        parts: list[str] = []
        for raw_block in blocks:
            block = _as_dict(raw_block)
            if block is not None and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""
