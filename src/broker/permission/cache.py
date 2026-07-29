"""Reuse cache for permission judgements within one session.

Claude Code re-fires the same permission request whenever a tool call is
retried, so one logical operation arrives several times. The cache is what
turns that flood into a single inference. The module is per-session, so the
session identity is structural and never part of a key.
"""

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CacheEntry:
    """A judgement, kept so an identical request can replay it."""

    decision: str
    reason: str
    model_id: str | None


def cache_key(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Build the canonical reuse key for one tool call.

    Args:
        tool_name: Name of the tool the session is asking to run.
        tool_input: Arguments the session passed to it.

    Returns:
        A key stable across argument orderings.
    """
    return tool_name + "\x00" + json.dumps(tool_input, sort_keys=True)


class ReuseCache:
    """Judgements keyed by canonical tool call, for one session."""

    def __init__(self) -> None:
        self._entries: dict[str, CacheEntry] = {}

    def get(self, key: str) -> CacheEntry | None:
        """Return the judgement stored for ``key``, or ``None``."""
        return self._entries.get(key)

    def put(self, key: str, entry: CacheEntry) -> None:
        """Store ``entry`` as the judgement for ``key``."""
        self._entries[key] = entry

    def clear(self) -> None:
        """Drop every judgement."""
        self._entries.clear()
