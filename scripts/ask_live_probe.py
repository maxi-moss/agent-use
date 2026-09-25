"""Live regression probe for PreToolUse updatedInput answer injection.

Manual-run only, never collected by pytest. Verifies, against the INSTALLED
Claude Code and Herdr binaries, that a PreToolUse hook returning
permissionDecision "allow" plus updatedInput answers an AskUserQuestion menu
without the native picker rendering. The mechanism is changelog-documented
only, so run this after every Claude Code or Herdr upgrade.

What it does:
- builds a throwaway project under /private/tmp with a project-scoped
  .claude/settings.json whose PreToolUse hook answers every question with its
  SECOND option's label (a list of labels for multiSelect, and a fixed free
  text for the question whose header is "Free"), and whose PostToolUse hook
  logs its echo;
- seeds folder trust for that one throwaway path (same undocumented key the
  master seeds, written read -> backup -> temp-write -> rename);
- splits a fresh Herdr pane off $HERDR_PANE_ID, starts a throwaway
  `claude --model sonnet` in it, and submits a prompt requesting one
  single-select, one multiSelect, and one free-text-targeted question;
- polls the session transcript for the three tool results and checks the
  structured answers, the PostToolUse echoes, and duration_ms == 0.

Safety rails: refuses to run when BROKER_SOCKET is set; only ever addresses
the pane it created; never touches ~/.claude/settings.json; nothing is killed
by name. Exit 0 = mechanism intact; non-zero names the failed assertion.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, cast

HERDR = "herdr"
FREE_TEXT_ANSWER = "probe free text answer"
PANE_READY_DELAY_S = 4.0  # agent start right after a split races pane init
AGENT_START_TIMEOUT_MS = 30000
TRANSCRIPT_WAIT_S = 300.0  # three real LLM turns with tool calls
POLL_INTERVAL_S = 2.0
HERDR_CMD_TIMEOUT_S = 20.0

PROMPT = (
    "Use the AskUserQuestion tool exactly three times, one question per "
    "call, in this order. Do not read or write any files.\n"
    "1. header 'Pick', single-select: 'Which fruit should I buy?' with "
    "options Apple, Banana, Cherry.\n"
    "2. header 'Multi', multiSelect true: 'Which toppings should I add?' "
    "with options Ham, Olives, Mushrooms.\n"
    "3. header 'Free', single-select: 'What should the project be named?' "
    "with options Alpha, Beta.\n"
    "After the third answer arrives, reply with exactly the word DONE."
)

PRE_HOOK = '''
import json, sys
payload = json.load(sys.stdin)
if payload.get("tool_name") != "AskUserQuestion":
    sys.exit(0)
tool_input = payload.get("tool_input") or {}
answers = {}
for q in tool_input.get("questions") or []:
    labels = [o.get("label") for o in q.get("options") or []]
    if q.get("header") == "Free":
        answers[q["question"]] = "%FREE_TEXT%"
    elif q.get("multiSelect"):
        answers[q["question"]] = labels[1:3]
    else:
        answers[q["question"]] = labels[1]
with open("%LOG_DIR%/injected.jsonl", "a") as f:
    f.write(json.dumps(
        {"tool_use_id": payload.get("tool_use_id"), "answers": answers}
    ) + "\\n")
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
        "permissionDecisionReason": "probe answered",
        "updatedInput": dict(tool_input, answers=answers),
    }
}))
'''

POST_HOOK = '''
import json, sys
payload = json.load(sys.stdin)
if payload.get("tool_name") != "AskUserQuestion":
    sys.exit(0)
with open("%LOG_DIR%/posttool.jsonl", "a") as f:
    f.write(json.dumps(payload) + "\\n")
'''


class ProbeFailure(Exception):
    """One named assertion failed; the message is the diagnosis."""


def run_herdr(argv: list[str], timeout_s: float) -> str:
    """Run one herdr command and return stdout, failing loud on any error."""
    proc = subprocess.run(
        [HERDR, *argv], capture_output=True, text=True, timeout=timeout_s
    )
    if proc.returncode != 0:
        raise ProbeFailure(
            f"herdr {' '.join(argv[:2])} failed "
            f"(exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def _as_dict(value: Any) -> dict[str, Any]:
    """Narrow a JSON value to an object, treating anything else as empty."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _content_blocks(message: dict[str, Any]) -> list[Any]:
    """Return a transcript message's content blocks, or an empty list."""
    content = message.get("content")
    return cast(list[Any], content) if isinstance(content, list) else []


def unwrap(stdout: str) -> Any:
    """Decode herdr JSON output, stripping the optional result envelope."""
    parsed: Any = json.loads(stdout)
    envelope = _as_dict(parsed)
    if "result" in envelope:
        return envelope["result"]
    return parsed


def seed_trust(project_path: Path, backup_dir: Path) -> None:
    """Set the trust flag for one project in ~/.claude.json, atomically.

    Args:
        project_path: The throwaway project to trust.
        backup_dir: Where the pre-write backup of ~/.claude.json goes.
    """
    target = Path.home() / ".claude.json"
    data: dict[str, Any] = {}
    if target.exists():
        raw: Any = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ProbeFailure(f"{target} is not a JSON object; refusing")
        data = cast(dict[str, Any], raw)
        shutil.copy2(target, backup_dir / "claude.json.probe-backup")
    raw_projects: Any = data.setdefault("projects", {})
    if not isinstance(raw_projects, dict):
        raise ProbeFailure(f"{target}: 'projects' is not an object; refusing")
    projects = cast(dict[str, Any], raw_projects)
    raw_entry: Any = projects.setdefault(str(project_path), {})
    if not isinstance(raw_entry, dict):
        raise ProbeFailure(
            f"{target}: projects[{str(project_path)!r}] is not an object; "
            "refusing"
        )
    entry = cast(dict[str, Any], raw_entry)
    entry["hasTrustDialogAccepted"] = True
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".claude.json.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2))
    os.replace(tmp_name, target)


def write_project(root: Path) -> Path:
    """Create the throwaway project with its hook scripts and settings.

    Args:
        root: The probe's temp directory.

    Returns:
        The project directory the pane will run in.
    """
    proj = root / "proj"
    proj.mkdir()
    (proj / ".claude").mkdir()
    pre_path = root / "pre_hook.py"
    post_path = root / "post_hook.py"
    pre_path.write_text(
        PRE_HOOK.replace("%FREE_TEXT%", FREE_TEXT_ANSWER).replace(
            "%LOG_DIR%", str(root)
        )
    )
    post_path.write_text(POST_HOOK.replace("%LOG_DIR%", str(root)))
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "AskUserQuestion",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{sys.executable} {pre_path}",
                            "timeout": 30,
                        }
                    ],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "AskUserQuestion",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"{sys.executable} {post_path}",
                            "timeout": 30,
                        }
                    ],
                }
            ],
        }
    }
    (proj / ".claude" / "settings.json").write_text(
        json.dumps(settings, indent=2)
    )
    return proj


def transcript_dir_for(cwd: Path) -> Path:
    """Return Claude Code's transcript directory for ``cwd``."""
    munged = re.sub(r"[^A-Za-z0-9]", "-", str(cwd))
    config = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
    return config / "projects" / munged


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into its record dicts, skipping non-objects."""
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed: Any = json.loads(line)
        if isinstance(parsed, dict):
            out.append(cast(dict[str, Any], parsed))
    return out


def transcript_answers(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Map tool_use_id -> structured answers for every ask result present."""
    ask_ids: set[str] = set()
    for rec in records:
        for raw_block in _content_blocks(_as_dict(rec.get("message"))):
            if not isinstance(raw_block, dict):
                continue
            block = cast(dict[str, Any], raw_block)
            if (
                block.get("type") == "tool_use"
                and block.get("name") == "AskUserQuestion"
            ):
                ask_ids.add(block["id"])
    answers: dict[str, Any] = {}
    for rec in records:
        for raw_block in _content_blocks(_as_dict(rec.get("message"))):
            if not isinstance(raw_block, dict):
                continue
            block = cast(dict[str, Any], raw_block)
            if (
                block.get("type") != "tool_result"
                or block.get("tool_use_id") not in ask_ids
            ):
                continue
            result = rec.get("toolUseResult")
            if isinstance(result, dict):
                answers[block["tool_use_id"]] = cast(
                    dict[str, Any], result
                ).get("answers")
    return answers


def find_duration_ms(node: Any) -> list[Any]:
    """Collect every duration_ms value anywhere in a JSON structure."""
    found: list[Any] = []
    if isinstance(node, dict):
        for key, value in cast(dict[str, Any], node).items():
            if key == "duration_ms":
                found.append(value)
            found.extend(find_duration_ms(value))
    elif isinstance(node, list):
        for item in cast(list[Any], node):
            found.extend(find_duration_ms(item))
    return found


def wait_for_answers(
    tdir: Path, before: set[Path], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Poll for a new transcript holding three injected answers.

    Args:
        tdir: Transcript directory for the project cwd.
        before: Transcript files that existed before the session started.
        root: The probe temp dir holding injected.jsonl.

    Returns:
        The transcript's answers by tool_use_id, and the injected log records.

    Raises:
        ProbeFailure: The deadline passed without three recorded answers.
    """
    deadline = time.monotonic() + TRANSCRIPT_WAIT_S
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_S)
        injected_path = root / "injected.jsonl"
        if not injected_path.exists():
            continue
        injected = read_jsonl(injected_path)
        candidates = (
            [p for p in tdir.glob("*.jsonl") if p not in before]
            if tdir.exists()
            else []
        )
        for path in candidates:
            try:
                answers = transcript_answers(read_jsonl(path))
            except json.JSONDecodeError:
                continue  # mid-write; next poll gets a full line
            if len(answers) >= 3 and len(injected) >= 3:
                return answers, injected
    raise ProbeFailure(
        f"no transcript with 3 recorded answers within "
        f"{TRANSCRIPT_WAIT_S:.0f} s — the injection likely stalled; "
        "check the pane before it closes"
    )


def check(answers: dict[str, Any], injected: list[dict[str, Any]], root: Path) -> None:
    """Run every assertion; each raises ProbeFailure with a diagnosis.

    Args:
        answers: tool_use_id -> structured transcript answers.
        injected: The pre-hook's log of what it injected, per tool_use_id.
        root: The probe temp dir holding posttool.jsonl.
    """
    by_id = {rec["tool_use_id"]: rec["answers"] for rec in injected}
    for tool_use_id, expected in by_id.items():
        got = answers.get(tool_use_id)
        if got != expected:
            raise ProbeFailure(
                f"structural mismatch for {tool_use_id}: transcript recorded "
                f"{got!r}, hook injected {expected!r}"
            )
    shapes = {
        type(value).__name__
        for rec in injected
        for value in rec["answers"].values()
    }
    if "list" not in shapes:
        raise ProbeFailure(
            "no multiSelect list was injected — the session never asked a "
            "multiSelect question; the probe covered less than it must"
        )
    free_values = [
        value
        for rec in injected
        for value in rec["answers"].values()
        if value == FREE_TEXT_ANSWER
    ]
    if not free_values:
        raise ProbeFailure(
            "the free-text answer was never injected — no question carried "
            "the 'Free' header"
        )
    post_path = root / "posttool.jsonl"
    if not post_path.exists():
        raise ProbeFailure("no PostToolUse echo was logged")
    posts = read_jsonl(post_path)
    if len(posts) < 3:
        raise ProbeFailure(f"expected 3 PostToolUse echoes, got {len(posts)}")
    for post in posts:
        response = post.get("tool_response")
        echoed = (
            cast(dict[str, Any], response).get("answers")
            if isinstance(response, dict)
            else None
        )
        expected = by_id.get(str(post.get("tool_use_id")))
        if echoed != expected:
            raise ProbeFailure(
                f"PostToolUse echo mismatch for {post.get('tool_use_id')}: "
                f"echoed {echoed!r}, injected {expected!r}"
            )
        durations = find_duration_ms(post)
        if not durations:
            raise ProbeFailure(
                "no duration_ms in the PostToolUse echo — the field moved; "
                "re-verify picker suppression by hand"
            )
        if any(d != 0 for d in durations):
            raise ProbeFailure(
                f"duration_ms {durations!r} != 0 — the picker rendered"
            )


def main() -> int:
    if os.environ.get("BROKER_SOCKET"):
        print(
            "refusing to run: BROKER_SOCKET is set, so this shell belongs "
            "to a broker-driven session. Run the probe from a plain pane.",
            file=sys.stderr,
        )
        return 2
    anchor = os.environ.get("HERDR_PANE_ID")
    if not anchor:
        print(
            "refusing to run: HERDR_PANE_ID is not set — run from inside a "
            "Herdr pane.",
            file=sys.stderr,
        )
        return 2

    root = Path(
        tempfile.mkdtemp(prefix=f"ask-probe-{os.getpid()}-", dir="/private/tmp")
    )
    proj = write_project(root)
    seed_trust(proj, root)
    tdir = transcript_dir_for(proj)
    before: set[Path] = set(tdir.glob("*.jsonl")) if tdir.exists() else set()

    pane_id: str | None = None
    agent = f"ask-probe-{os.getpid()}"
    try:
        pane = unwrap(
            run_herdr(
                [
                    "pane", "split",
                    "--pane", anchor,
                    "--direction", "right",
                    "--cwd", str(proj),
                    "--no-focus",
                ],
                HERDR_CMD_TIMEOUT_S,
            )
        )
        pane_id = str(pane["pane"]["pane_id"])
        time.sleep(PANE_READY_DELAY_S)
        run_herdr(
            [
                "agent", "start", agent,
                "--kind", "claude",
                "--pane", pane_id,
                "--timeout", str(AGENT_START_TIMEOUT_MS),
                "--", "--model", "sonnet",
            ],
            AGENT_START_TIMEOUT_MS / 1000 + 10.0,
        )
        run_herdr(["agent", "prompt", agent, PROMPT], HERDR_CMD_TIMEOUT_S)
        answers, injected = wait_for_answers(tdir, before, root)
        check(answers, injected, root)
    except ProbeFailure as exc:
        print(f"PROBE FAILED: {exc}", file=sys.stderr)
        print(f"artifacts kept for inspection: {root}", file=sys.stderr)
        if pane_id is not None:
            run_herdr(["pane", "close", pane_id], HERDR_CMD_TIMEOUT_S)
        return 1
    run_herdr(["pane", "close", pane_id], HERDR_CMD_TIMEOUT_S)
    print("PROBE OK: updatedInput answered all three menus; picker never rendered")
    print(f"artifacts: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
