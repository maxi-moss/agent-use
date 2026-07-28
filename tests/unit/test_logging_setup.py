"""Diagnostic logging lands in a file, never on the TUI's terminal."""

import logging
from pathlib import Path

from broker import logging_setup


def _reset_root() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_no_stream_handler_survives(tmp_path: Path) -> None:
    logging.getLogger().addHandler(logging.StreamHandler())
    try:
        logging_setup.configure(tmp_path / "broker.log")
        handlers = logging.getLogger().handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.FileHandler)
    finally:
        _reset_root()
