"""Every herdr CLI call, one module (plan §7). Thin subprocess wrappers.

Rules encoded here, spike-verified against herdr 0.7.5:
- `agent prompt` types but does NOT submit; every submit is prompt +
  `pane send-keys <pane_id> enter`. submit_prompt() is the sanctioned two-step.
- Every wait carries an explicit timeout (constraint 13) — no defaults on waits.
- No public API accepts or emits `--current` (constraint 14).
- Exit 1 + stderr JSON -> HerdrError(code, message); exit 2 -> RuntimeError (our bug).
- Trust `herdr status --json` .server.compatible, never `herdr --version`.
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any, cast

from broker.herdr.schemas import AgentStartResult, HerdrStatus, PaneInfo

HERDR = "herdr"

# spec §5 correction 1: names must match this and be unique among live agents
AGENT_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")

_DIRECTIONS = frozenset({"right", "down"})


class HerdrError(Exception):
    """herdr reported a server/timeout error (exit 1, JSON on stderr)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _check_identifier(value: str, what: str) -> str:
    """Targets, pane ids and anchors must never be flags — kills --current
    (constraint 14) and every other flag injection in one place."""
    if not value or value.startswith("-"):
        raise ValueError(f"invalid {what}: {value!r}")
    return value


def _check_agent_name(name: str) -> str:
    if not AGENT_NAME_RE.fullmatch(name):
        raise ValueError(
            f"agent name {name!r} must match {AGENT_NAME_RE.pattern}"
        )
    return name


def _run(argv: list[str], timeout_s: float) -> str:
    proc = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout_s
    )
    if proc.returncode == 0:
        return proc.stdout
    if proc.returncode == 1:
        raise _error_from_stderr(proc.stderr)
    # exit 2 = CLI usage error -> a bug in this module, fail loud
    raise RuntimeError(
        f"herdr usage error (exit {proc.returncode}): {argv!r}: {proc.stderr!r}"
    )


def _error_from_stderr(stderr: str) -> HerdrError:
    try:
        parsed: Any = json.loads(stderr)
    except json.JSONDecodeError:
        return HerdrError("unparseable_error", stderr.strip())
    if isinstance(parsed, dict):
        error = cast(dict[str, Any], parsed).get("error")
        if isinstance(error, dict):
            typed_error = cast(dict[str, Any], error)
            return HerdrError(
                str(typed_error.get("code", "unknown")),
                str(typed_error.get("message", "")),
            )
    return HerdrError("unknown", stderr.strip())


def _parse_json(stdout: str) -> Any:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise HerdrError("unparseable_result", stdout.strip()[:200]) from exc


def _unwrap_result(parsed: Any) -> Any:
    """Some herdr results nest under {"result": ...}; accept both."""
    if isinstance(parsed, dict) and "result" in cast(dict[str, Any], parsed):
        return cast(dict[str, Any], parsed)["result"]
    return cast(Any, parsed)


def status(*, timeout_s: float = 10.0) -> HerdrStatus:
    stdout = _run([HERDR, "status", "--json"], timeout_s)
    return HerdrStatus.model_validate(_parse_json(stdout))


def pane_split(
    anchor: str,
    *,
    direction: str,
    cwd: Path,
    env: dict[str, str],
    focus: bool,
    timeout_s: float,
) -> PaneInfo:
    _check_identifier(anchor, "anchor pane id")
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}")
    argv = [
        HERDR, "pane", "split",
        "--pane", anchor,
        "--direction", direction,
        "--cwd", str(cwd),
    ]
    for key, value in env.items():
        argv += ["--env", f"{key}={value}"]
    if not focus:
        argv.append("--no-focus")
    stdout = _run(argv, timeout_s)
    return PaneInfo.model_validate(_parse_json(stdout))


def agent_start(
    name: str,
    *,
    kind: str,
    pane_id: str,
    timeout_ms: int,
) -> AgentStartResult:
    _check_agent_name(name)
    _check_identifier(pane_id, "pane id")
    _check_identifier(kind, "agent kind")
    argv = [
        HERDR, "agent", "start", name,
        "--kind", kind,
        "--pane", pane_id,
        "--timeout", str(timeout_ms),
    ]
    stdout = _run(argv, timeout_s=timeout_ms / 1000 + 10.0)
    return AgentStartResult.model_validate(_unwrap_result(_parse_json(stdout)))


def agent_prompt(target: str, text: str, *, timeout_s: float) -> None:
    """Types into the input box only — does NOT submit (spike-verified twice)."""
    _check_identifier(target, "agent target")
    _run([HERDR, "agent", "prompt", target, text], timeout_s)


def pane_send_keys(pane_id: str, *keys: str, timeout_s: float) -> None:
    _check_identifier(pane_id, "pane id")
    for key in keys:
        _check_identifier(key, "key")
    _run([HERDR, "pane", "send-keys", pane_id, *keys], timeout_s)


def submit_prompt(
    target: str, pane_id: str, text: str, *, timeout_s: float
) -> None:
    """The sanctioned two-step: type, then submit with an explicit Enter."""
    agent_prompt(target, text, timeout_s=timeout_s)
    pane_send_keys(pane_id, "enter", timeout_s=timeout_s)


def agent_wait(
    target: str, *, until: list[str], timeout_ms: int
) -> dict[str, Any]:
    """--timeout is REQUIRED — herdr waits block forever without it."""
    _check_identifier(target, "agent target")
    if not until:
        raise ValueError("agent_wait requires at least one --until state")
    argv = [HERDR, "agent", "wait", target]
    for state in until:
        _check_identifier(state, "wait state")
        argv += ["--until", state]
    argv += ["--timeout", str(timeout_ms)]
    stdout = _run(argv, timeout_s=timeout_ms / 1000 + 10.0)
    parsed: Any = _parse_json(stdout) if stdout.strip() else {}
    if isinstance(parsed, dict):
        return cast(dict[str, Any], parsed)
    return {}


def pane_read(pane_id: str, *, timeout_s: float) -> str:
    """Returns PLAIN TEXT, not JSON (spike-verified)."""
    _check_identifier(pane_id, "pane id")
    return _run(
        [HERDR, "pane", "read", pane_id, "--source", "visible", "--format", "text"],
        timeout_s,
    )


def pane_close(pane_id: str, *, timeout_s: float) -> None:
    _check_identifier(pane_id, "pane id")
    _run([HERDR, "pane", "close", pane_id], timeout_s)


def notification_show(
    title: str, *, body: str, sound: str, timeout_s: float
) -> None:
    if sound not in {"none", "done", "request"}:
        raise ValueError(f"invalid sound: {sound!r}")
    _run(
        [HERDR, "notification", "show", title, "--body", body, "--sound", sound],
        timeout_s,
    )
