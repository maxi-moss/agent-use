"""The hook's import closure is asserted, not left to discipline."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_HOOK_MAIN_SRC = (
    Path(__file__).resolve().parents[2] / "src" / "broker" / "hook" / "__main__.py"
)


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


def test_broken_constants_import_degrades_and_records() -> None:
    """A broken broker.protocol.constants exits 0 with silent stdout, loudly logged."""
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        tmp = Path(td)
        (tmp / "broker").mkdir()
        (tmp / "broker" / "__init__.py").write_text("")
        (tmp / "broker" / "hook").mkdir()
        (tmp / "broker" / "hook" / "__init__.py").write_text("")
        shutil.copyfile(_HOOK_MAIN_SRC, tmp / "broker" / "hook" / "__main__.py")
        (tmp / "broker" / "protocol").mkdir()
        (tmp / "broker" / "protocol" / "__init__.py").write_text("")
        (tmp / "broker" / "protocol" / "constants.py").write_text(
            'raise RuntimeError("constants boom")\n'
        )

        log_path = tmp / "hook.log"
        run_code = (
            "import runpy, sys; "
            f"sys.path.insert(0, {str(tmp)!r}); "
            "runpy.run_module('broker.hook', run_name='__main__')"
        )
        env = dict(os.environ)
        env["BROKER_HOOK_LOG"] = str(log_path)
        proc = subprocess.run(
            [sys.executable, "-I", "-c", run_code],
            capture_output=True,
            text=True,
            env=env,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout == ""
        assert "RuntimeError: constants boom" in log_path.read_text()

        import_code = (
            "import sys; "
            f"sys.path.insert(0, {str(tmp)!r}); "
            "import broker.hook.__main__"
        )
        proc = subprocess.run(
            [sys.executable, "-I", "-c", import_code],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
