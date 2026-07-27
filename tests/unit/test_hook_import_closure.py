"""The hook's import closure is asserted, not left to discipline (plan §1.3)."""

import subprocess
import sys


def test_no_pydantic_in_hook_closure() -> None:
    code = (
        "import sys, broker.hook.__main__; "
        "assert 'pydantic' not in sys.modules, 'pydantic in hook closure'; "
        "assert 'broker.protocol.schemas' not in sys.modules, "
        "'protocol.schemas in hook closure'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


def test_hook_closure_is_stdlib_plus_constants_only() -> None:
    """No broker module beyond protocol.constants may load."""
    code = (
        "import sys, broker.hook.__main__; "
        "loaded = sorted(m for m in sys.modules if m.startswith('broker')); "
        "allowed = {'broker', 'broker.hook', 'broker.hook.__main__', "
        "'broker.protocol', 'broker.protocol.constants'}; "
        "extra = set(loaded) - allowed; "
        "assert not extra, f'unexpected broker imports: {extra}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
