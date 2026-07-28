"""Every herdr CLI call, one module. Thin subprocess wrappers.

Rules encoded here, verified against herdr 0.7.5:
- `agent prompt` types but does NOT submit; every submit is prompt +
  `pane send-keys <pane_id> enter`. submit_prompt() is the sanctioned two-step.
- Every wait carries an explicit timeout — no defaults on waits.
- No public API accepts or emits `--current`.
- Exit 1 + stderr JSON -> HerdrError(code, message); exit 2 -> RuntimeError (our bug).
- Trust `herdr status --json` .server.compatible, never `herdr --version`.
"""

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, cast

from broker.herdr.schemas import AgentStartResult, HerdrStatus, PaneInfo

HERDR = "herdr"

# Names must match this and be unique among live agents
AGENT_NAME_RE = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")

_DIRECTIONS = frozenset({"right", "down"})

# A pane freshly returned by `pane split` has not necessarily finished
# initializing its shell yet. `agent start`'s own --timeout claims to "wait
# for interactive readiness" but does not: called immediately after a split,
# it fails instantly with agent_pane_busy roughly half the time (verified
# live, repeatedly, against herdr 0.7.5). This fixed delay is the workaround.
_PANE_READY_DELAY_S = 1.0


class HerdrError(Exception):
    """herdr reported a server/timeout error (exit 1, JSON on stderr)."""

    def __init__(self, code: str, message: str) -> None:
        """Store herdr's error ``code`` and ``message``."""
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def _check_identifier(value: str, what: str) -> str:
    """Reject an identifier that could be read as a CLI flag.

    Args:
        value: Candidate identifier destined for a herdr argv.
        what: Noun used in the error message, e.g. ``"pane id"``.

    Returns:
        ``value`` unchanged.

    Raises:
        ValueError: ``value`` is empty or starts with ``-``.
    """
    if not value or value.startswith("-"):
        raise ValueError(f"invalid {what}: {value!r}")
    return value


def _check_agent_name(name: str) -> str:
    """Validate an agent name against ``AGENT_NAME_RE``.

    Args:
        name: Proposed agent name.

    Returns:
        ``name`` unchanged.

    Raises:
        ValueError: The name does not match ``AGENT_NAME_RE``.
    """
    if not AGENT_NAME_RE.fullmatch(name):
        raise ValueError(
            f"agent name {name!r} must match {AGENT_NAME_RE.pattern}"
        )
    return name


def _run(argv: list[str], timeout_s: float) -> str:
    """Run one herdr command and return its stdout.

    Args:
        argv: Full command line, starting with the herdr binary.
        timeout_s: Wall-clock limit handed to ``subprocess.run``.

    Returns:
        The command's stdout.

    Raises:
        HerdrError: herdr exited 1; the code and message come from stderr.
        RuntimeError: herdr exited with any other non-zero code.
    """
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
    """Build a ``HerdrError`` from herdr's JSON error output.

    Args:
        stderr: Raw stderr text from an exit-1 herdr run.

    Returns:
        An error built from ``error.code``/``error.message`` when present,
        otherwise one coded ``unparseable_error`` or ``unknown``.
    """
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
    """Decode herdr's stdout as JSON.

    Args:
        stdout: Raw stdout from a successful herdr run.

    Returns:
        The decoded JSON value.

    Raises:
        HerdrError: The output is not valid JSON.
    """
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise HerdrError("unparseable_result", stdout.strip()[:200]) from exc


def _unwrap_result(parsed: Any) -> Any:
    """Strip herdr's optional ``result`` envelope, accepting both shapes.

    Args:
        parsed: Decoded JSON value from a herdr command.

    Returns:
        The value under ``result`` if present, otherwise ``parsed`` unchanged.
    """
    if isinstance(parsed, dict) and "result" in cast(dict[str, Any], parsed):
        return cast(dict[str, Any], parsed)["result"]
    return cast(Any, parsed)


def status(*, timeout_s: float = 10.0) -> HerdrStatus:
    """Read the herdr client/server status.

    Args:
        timeout_s: Wall-clock limit for the subprocess.

    Returns:
        The parsed status. Trust ``.server.compatible``, never ``--version``.
    """
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
    """Split an existing pane and return the pane that was created.

    Args:
        anchor: Pane id to split from.
        direction: Where the new pane goes; one of ``"right"`` or ``"down"``.
        cwd: Working directory for the new pane.
        env: Environment variables, each passed as its own ``--env KEY=VALUE``.
        focus: When false, leaves the developer's focus where it is.
        timeout_s: Wall-clock limit for the subprocess.

    Returns:
        The new pane.

    Raises:
        ValueError: ``anchor`` looks like a flag, or ``direction`` is invalid.
    """
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
    """Start an agent in an existing pane.

    Args:
        name: Agent name; must match ``AGENT_NAME_RE``.
        kind: Agent kind, passed through to ``--kind``.
        pane_id: Pane the agent takes over.
        timeout_ms: herdr's own start timeout, in milliseconds.

    Returns:
        The parsed start result, unwrapped from any ``result`` envelope.

    Raises:
        ValueError: The name, kind or pane id fails validation.
    """
    _check_agent_name(name)
    _check_identifier(pane_id, "pane id")
    _check_identifier(kind, "agent kind")
    time.sleep(_PANE_READY_DELAY_S)
    argv = [
        HERDR, "agent", "start", name,
        "--kind", kind,
        "--pane", pane_id,
        "--timeout", str(timeout_ms),
    ]
    stdout = _run(argv, timeout_s=timeout_ms / 1000 + 10.0)
    return AgentStartResult.model_validate(_unwrap_result(_parse_json(stdout)))


def agent_get(target: str, *, timeout_s: float) -> dict[str, Any]:
    """Read ``herdr agent get <target>``.

    Args:
        target: Agent name or id.
        timeout_s: Wall-clock limit for the subprocess.

    Returns:
        The unwrapped result dict.

    Raises:
        HerdrError: The result unwrapped to something other than an object.
    """
    _check_identifier(target, "agent target")
    stdout = _run([HERDR, "agent", "get", target], timeout_s)
    unwrapped = _unwrap_result(_parse_json(stdout))
    if isinstance(unwrapped, dict):
        return cast(dict[str, Any], unwrapped)
    raise HerdrError("unexpected_result", f"agent get returned {type(unwrapped).__name__}")


def agent_status(info: dict[str, Any]) -> str:
    """Extract ``agent_status`` from an unwrapped ``agent get`` result.

    Unreadable or missing values degrade to ``"unknown"``.

    Args:
        info: Unwrapped result dict, as returned by :func:`agent_get`.

    Returns:
        The status string, or ``"unknown"`` when it is absent or not a string.
    """
    agent = info.get("agent")
    if isinstance(agent, dict):
        status = cast(dict[str, Any], agent).get("agent_status")
        if isinstance(status, str):
            return status
    return "unknown"


def agent_prompt(target: str, text: str, *, timeout_s: float) -> None:
    """Type ``text`` into an agent's input box.

    Does not submit; a submit keystroke must follow. See :func:`submit_prompt`.

    Args:
        target: Agent name or id.
        text: Prompt text to type.
        timeout_s: Wall-clock limit for the subprocess.

    Raises:
        ValueError: ``target`` looks like a flag.
    """
    _check_identifier(target, "agent target")
    _run([HERDR, "agent", "prompt", target, text], timeout_s)


def pane_send_keys(pane_id: str, *keys: str, timeout_s: float) -> None:
    """Send literal keystrokes to a pane.

    Args:
        pane_id: Pane to send to.
        *keys: Key names, e.g. ``"enter"``; each is flag-checked in turn.
        timeout_s: Wall-clock limit for the subprocess.

    Raises:
        ValueError: The pane id or any key looks like a flag.
    """
    _check_identifier(pane_id, "pane id")
    for key in keys:
        _check_identifier(key, "key")
    _run([HERDR, "pane", "send-keys", pane_id, *keys], timeout_s)


def submit_prompt(
    target: str, pane_id: str, text: str, *, timeout_s: float
) -> None:
    """Type a prompt and submit it: herdr never submits on its own.

    Args:
        target: Agent name or id to type into.
        pane_id: Pane that receives the Enter keystroke.
        text: Prompt text.
        timeout_s: Wall-clock limit applied to each of the two calls.
    """
    agent_prompt(target, text, timeout_s=timeout_s)
    pane_send_keys(pane_id, "enter", timeout_s=timeout_s)


def agent_wait(
    target: str, *, until: list[str], timeout_ms: int
) -> dict[str, Any]:
    """Block until an agent reaches one of the ``until`` states.

    Requires an explicit timeout; herdr waits block forever without one.

    Args:
        target: Agent name or id.
        until: Agent states to wait for; each becomes its own ``--until`` flag.
        timeout_ms: herdr's wait timeout, in milliseconds.

    Returns:
        The parsed result object, or ``{}`` when herdr printed nothing usable.

    Raises:
        ValueError: ``until`` is empty, or the target or a state looks like a
            flag.
    """
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
    """Read the visible contents of a pane as plain text, not JSON.

    Args:
        pane_id: Pane to read.
        timeout_s: Wall-clock limit for the subprocess.

    Returns:
        The visible text of the pane.

    Raises:
        ValueError: ``pane_id`` looks like a flag.
    """
    _check_identifier(pane_id, "pane id")
    return _run(
        [HERDR, "pane", "read", pane_id, "--source", "visible", "--format", "text"],
        timeout_s,
    )


def pane_close(pane_id: str, *, timeout_s: float) -> None:
    """Close the pane with the given id."""
    _check_identifier(pane_id, "pane id")
    _run([HERDR, "pane", "close", pane_id], timeout_s)


def notification_show(
    title: str, *, body: str, sound: str, timeout_s: float
) -> None:
    """Raise a desktop notification through herdr.

    Args:
        title: Notification title.
        body: Notification body text.
        sound: One of ``"none"``, ``"done"`` or ``"request"``.
        timeout_s: Wall-clock limit for the subprocess.

    Raises:
        ValueError: ``sound`` is not one of the three accepted values.
    """
    if sound not in {"none", "done", "request"}:
        raise ValueError(f"invalid sound: {sound!r}")
    _run(
        [HERDR, "notification", "show", title, "--body", body, "--sound", sound],
        timeout_s,
    )
