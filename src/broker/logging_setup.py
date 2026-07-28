"""File-based logging setup for the broker's long-lived processes.

Records go to a file and never to stderr: the master hands its terminal to a
full-screen TUI and session brokers inherit that terminal, so a stream handler
writes into a display that immediately repaints over it.
"""

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure(path: Path, *, level: int = logging.INFO) -> None:
    """Route root-logger records to ``path``, replacing any existing handlers.

    Args:
        path: Log file; created, with its parent directories, if absent.
        level: Minimum level to record.

    Raises:
        OSError: The log file could not be opened for appending.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()
    root.addHandler(handler)
    root.setLevel(level)
