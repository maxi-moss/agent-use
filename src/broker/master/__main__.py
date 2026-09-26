"""broker.master entrypoint. Startup sequence — ORDER IS BINDING:
environment assertions → hook verify-and-repair (user level) →
registry reconciliation (probe, classify, retract — never spawn) →
runtime/app built → App.run() (the socket server starts inside on_mount).

The registry file is READ before hook verify-and-repair only to supply candidate
shadow paths (known cwds are needed here); reconciliation and its save happen
in their bound position, after verify-and-repair and before the app is built,
so a dead session's retracted escalation and open pane escalations are gone
from disk before the runtime re-announces the persisted head and open pane
escalations.
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import NoReturn

from broker import config as broker_config
from broker import llm_timing
from broker import logging_setup
from broker.claude.paths import claude_json_path, settings_path
from broker.claude.settings import broker_hook_command, verify_and_repair
from broker.herdr import driver
from broker.paths import BrokerPaths
from broker.llm import build_client
from broker.protocol.constants import HookEventName
from broker.master.llm import MasterLLM, bind_call_turn
from broker.master.pane_escalations import PaneEscalations, PaneStoreError
from broker.master.queue import EscalationQueue, QueueError
from broker.master.registry import Registry, RegistryError
from broker.master.runtime import MasterRuntime, reconcile_registry
from broker.master.testmode import InjectCommand, test_mode_llm_call
from broker.master.tui.app import BrokerMasterApp
from broker.master.viewmodel import Notice, ViewEvent, ViewEventRelay

TEST_MODE_ANCHOR = "%test-mode"
TEST_MODE_WARNING = "TEST MODE — synthetic traffic only; LLM disabled"
STATUS_TIMEOUT_S = 10.0

logger = logging.getLogger(__name__)


def log_notice(event: ViewEvent) -> None:
    """Write every ``Notice`` view event to the configured log file at WARNING."""
    if isinstance(event, Notice):
        logger.warning(event.text)


def _fail(reason: str) -> NoReturn:
    print(f"broker.master: {reason}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(prog="broker.master")
    parser.add_argument(
        "--anchor", help="anchor pane id (overrides HERDR_PANE_ID)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="synthetic escalation mode: no LLM, no hooks, no live sessions",
    )
    args = parser.parse_args()

    # 1. Environment assertions — FAIL LOUD on any of them. Test mode has no
    #    hooks, no LLM and no live sessions, so none of them apply.
    if not args.test_mode:
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
            if not driver.status(timeout_s=STATUS_TIMEOUT_S).compatible:
                _fail("herdr client/server report incompatible")
        except Exception as exc:
            _fail(f"`herdr status` failed: {exc}")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            _fail("ANTHROPIC_API_KEY is not set")
        if not os.environ.get("OPENAI_API_KEY"):
            _fail("OPENAI_API_KEY is not set")
        anchor = args.anchor or os.environ.get("HERDR_PANE_ID")
        if not anchor:
            _fail("HERDR_PANE_ID is not set and --anchor was not given")
    else:
        anchor = args.anchor or os.environ.get("HERDR_PANE_ID") or TEST_MODE_ANCHOR

    try:
        cfg = broker_config.load()
    except broker_config.ConfigError as exc:
        _fail(str(exc))
    paths = BrokerPaths(cfg.broker_home)
    logging_setup.configure(paths.master_log)
    llm_timing.configure(paths.llm_timings, "master")
    settings = settings_path()
    claude_json = claude_json_path()
    relay = ViewEventRelay()
    relay.connect(log_notice)
    try:
        registry = Registry.load(paths.registry)
    except RegistryError as exc:
        _fail(str(exc))
    try:
        queue = EscalationQueue.load(paths.escalation_queue)
    except QueueError as exc:
        _fail(str(exc))
    try:
        panes = PaneEscalations.load(paths.pane_escalations)
    except PaneStoreError as exc:
        _fail(str(exc))

    if args.test_mode:
        warnings = [TEST_MODE_WARNING]
        llm_call = test_mode_llm_call
    else:
        # 2. Hook verify-and-repair at USER level (never project-level),
        #    with candidate shadow paths.
        command = broker_hook_command(sys.executable)
        candidates = [
            Path(record.cwd) / ".claude" / "settings.json"
            for record in registry.records.values()
        ]
        report = verify_and_repair(
            list(HookEventName), command, settings, shadow_candidates=candidates
        )
        warnings = list(report.warnings)

        # 3. Reconcile the registry: probe, classify, retract — never spawn.
        warnings.extend(asyncio.run(reconcile_registry(registry, queue, panes)))

        llm_call = bind_call_turn(build_client())

    # 4. Runtime and app built before run; the runtime starts in on_mount.
    runtime = MasterRuntime(
        relay, registry, queue, panes, cfg, anchor_pane=anchor, claude_json=claude_json
    )
    master_llm = MasterLLM(llm_call, runtime, cfg)
    inject: InjectCommand | None = None
    if args.test_mode:
        posts: list[ViewEvent] = []
        relay.connect(posts.append)
        inject = InjectCommand(runtime, posts)
    app = BrokerMasterApp(
        runtime,
        master_llm,
        relay,
        startup_warnings=warnings,
        inject=inject,
    )
    app.run()


if __name__ == "__main__":
    main()
