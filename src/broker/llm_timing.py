"""How long each LLM call takes, as an append-only NDJSON sidecar.

``timed`` is a decorator: put it on the coroutine that makes one LLM call and
every invocation appends one line — ``stage``, wall-clock ``ts``, and
``duration_ms`` — to a shared file the master and every session broker write to.
An unconfigured process, or a failed write, is swallowed: timing is degraded,
never load-bearing. These timestamps live only in the file, never in LLM
context.
"""

import functools
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_path: Path | None = None
_process: str = "?"


def configure(path: Path, process: str) -> None:
    """Point LLM-call timing at its NDJSON file and name the writing process.

    Args:
        path: Shared timing log; created, with parents, on first write.
        process: Label for this process, e.g. ``"master"`` or
            ``"session:s3"``, recorded on every line it writes.
    """
    global _path, _process
    path.parent.mkdir(parents=True, exist_ok=True)
    _path = path
    _process = process


def _write(stage: str, duration_ms: float) -> None:
    """Append one timing line, or do nothing if timing is unconfigured."""
    if _path is None:
        return
    entry: dict[str, Any] = {
        "ts": time.time(),
        "process": _process,
        "pid": os.getpid(),
        "stage": stage,
        "duration_ms": round(duration_ms, 3),
    }
    try:
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        with _path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    except OSError as exc:
        logger.warning("llm timing write failed: %r", exc)


def timed[C: Callable[..., Any]](stage: str) -> Callable[[C], C]:
    """Decorate a coroutine so each call records its duration under ``stage``.

    The wrapped coroutine keeps its exact signature and behaviour; only a
    timing line is written as it returns.

    Args:
        stage: Stable name for this LLM call site, e.g. ``"grounding"``.

    Returns:
        The decorator.
    """

    def decorator(fn: C) -> C:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                _write(stage, (time.perf_counter() - start) * 1000.0)

        return cast(C, wrapper)

    return decorator
