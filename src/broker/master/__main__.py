"""broker-master entrypoint. Startup sequence — ORDER IS BINDING:
environment assertions → hook registration (user level) + verify-and-repair →
registry load with unmanaged marking → runtime/app built → App.run() (the
socket server starts inside on_mount).

The registry file is READ before hook registration only to supply candidate
shadow paths (known cwds are needed here); the unmanaged marking and save
happen in their bound position, after verify-and-repair.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import NoReturn

from broker import config as broker_config
from broker import logging_setup
from broker.claude.settings import register_hooks, verify_and_repair
from broker.herdr import driver
from broker.paths import BrokerPaths
from broker.llm import build_client
from broker.master.llm import bind_call_turn
from broker.master.queue import EscalationQueue, QueueError
from broker.master.registry import Registry
from broker.master.tui.app import BrokerMasterApp
from broker.protocol.constants import SessionState

# Core hook events plus observability extras
EVENTS = [
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "Stop",
    "StopFailure",
    "Notification",
    "PreToolUse",
    "PostToolUse",
    "PreCompact",
    "PostCompact",
    "PermissionRequest",
]


def _fail(reason: str) -> NoReturn:
    print(f"broker-master: {reason}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="broker-master")
    parser.add_argument(
        "--anchor", help="anchor pane id (overrides HERDR_PANE_ID)"
    )
    args = parser.parse_args()

    # 1. Environment assertions — FAIL LOUD on any of them.
    if os.environ.get("CLAUDE_CODE_SKIP_PROMPT_HISTORY"):
        _fail("CLAUDE_CODE_SKIP_PROMPT_HISTORY is set; unset it first")
    if os.environ.get("CLAUDE_CODE_CHILD_SESSION"):
        _fail(
            "CLAUDE_CODE_CHILD_SESSION is set — it silently disables "
            "transcripts; unset it first"
        )
    if shutil.which("claude") is None:
        _fail("`claude` is not on PATH")
    if shutil.which("herdr") is None:
        _fail("`herdr` is not on PATH")
    try:
        if not driver.status().compatible:
            _fail("herdr client/server report incompatible")
    except Exception as exc:
        _fail(f"`herdr status` failed: {exc}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _fail("ANTHROPIC_API_KEY is not set")
    anchor = args.anchor or os.environ.get("HERDR_PANE_ID")
    if not anchor:
        _fail("HERDR_PANE_ID is not set and --anchor was not given")

    cfg = broker_config.load()
    paths = BrokerPaths(cfg.broker_home)
    logging_setup.configure(paths.master_log)
    registry = Registry.load(paths.registry)
    try:
        queue = EscalationQueue.load(paths.escalation_queue)
    except QueueError as exc:
        _fail(str(exc))

    # 2. Hook registration at USER level (never project-level),
    #    then verify-and-repair with candidate shadow paths.
    command = (
        f'[ -n "$BROKER_SOCKET" ] || exit 0; '
        f"exec {sys.executable} -m broker.hook  # broker-hook"
    )
    register_hooks(EVENTS, command)
    candidates = [
        Path(record.cwd) / ".claude" / "settings.json"
        for record in registry.records.values()
    ]
    report = verify_and_repair(EVENTS, command, shadow_candidates=candidates)
    warnings = list(report.warnings)

    # 3. Sessions found at startup are unmanaged.
    for record in registry.records.values():
        if record.state != SessionState.UNMANAGED:
            record.state = SessionState.UNMANAGED
            warnings.append(
                f"session {record.name} found in registry — marked unmanaged"
            )
    if registry.records:
        registry.save()

    # 4. Runtime and app built before run; runtime.serve() starts in on_mount.
    llm_call = bind_call_turn(build_client(cfg))
    app = BrokerMasterApp(
        cfg,
        registry,
        queue,
        llm_call,
        anchor_pane=anchor,
        startup_warnings=warnings,
    )
    app.run()


if __name__ == "__main__":
    main()
