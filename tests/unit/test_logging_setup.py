"""Diagnostic logging lands in a file, never on the TUI's terminal."""

import logging
import warnings
from pathlib import Path

from broker import logging_setup


def _reset_root() -> None:
    logging.captureWarnings(False)
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


def test_warnings_land_in_the_log_file(tmp_path: Path) -> None:
    path = tmp_path / "broker.log"
    # Another module's configure may have left capture on, which would make
    # this configure's capture a no-op under pytest's own warning recorder.
    _reset_root()
    try:
        logging_setup.configure(path)
        warnings.warn("a deprecated call", DeprecationWarning, stacklevel=1)
        assert "a deprecated call" in path.read_text(encoding="utf-8")
    finally:
        _reset_root()
