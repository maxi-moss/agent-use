"""Startup registry reconciliation: classify every persisted session and
drop the ones no broker can ever drive again.
"""

import asyncio
from pathlib import Path

from broker.herdr import driver
from broker.master.broker_link import adoption_fields, broker_is_listening
from broker.master.pane_escalations import PaneEscalations
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry
from broker.protocol.constants import SessionState

AGENT_PROBE_TIMEOUT_S = 5.0


def _drop_from_fleet(
    registry: Registry,
    queue: EscalationQueue,
    panes: PaneEscalations,
    name: str,
) -> list[str]:
    """Remove a session no broker can drive again, retracting its escalations.

    Args:
        registry: Session registry; the removal persists on its own.
        queue: Persisted escalation queue.
        panes: Persisted pane escalations.
        name: Session to drop.

    Returns:
        One line per retracted escalation.
    """
    lines: list[str] = []
    queued = queue.retract_for_session(name)
    if queued is not None:
        lines.append(
            f"session {name}: queued escalation {queued.payload.escalation_id} "
            "retracted — no decision can reach it"
        )
    for prompt in panes.retract_for_session(name):
        lines.append(
            f"session {name}: {prompt.kind} escalation "
            f"{prompt.escalation_id} retracted — no broker will see it closed"
        )
    registry.remove(name)
    return lines


async def reconcile_registry(
    registry: Registry,
    queue: EscalationQueue,
    panes: PaneEscalations,
) -> list[str]:
    """Classify every registry session at startup, dropping finished ones.

    Probes are client-side connects and herdr reads only — nothing is spawned
    and nothing binds a socket. A session with no answering broker is dropped
    once its Claude Code no longer runs, or when the registry lacks what a
    replacement broker needs to adopt it. An inconclusive probe never drops a
    session — a wrong ``unmanaged`` costs the developer a glance, a wrong
    removal throws away queued decisions.

    Args:
        registry: Loaded session registry; a dropped session is removed, every
            other classification updates its state in place and is saved once.
        queue: Persisted escalation queue; a dropped session's queued
            escalation is retracted from it before the TUI re-announces
            the head.
        panes: Persisted pane escalations; a dropped session's open ones are
            retracted before the TUI re-announces what remains.

    Returns:
        One classification line per session, plus one line per retraction.
    """
    warnings: list[str] = []
    for name in registry.names_in_order():
        record = registry.records[name]
        if await broker_is_listening(Path(record.socket_path)):
            warnings.append(
                f"session {name}: broker still answering — left as-is"
            )
            continue
        try:
            running = await asyncio.to_thread(
                driver.agent_running, name, timeout_s=AGENT_PROBE_TIMEOUT_S
            )
        except Exception as exc:
            record.state = SessionState.UNMANAGED
            warnings.append(
                f"session {name}: agent probe inconclusive ({exc!r}) — "
                "marked unmanaged"
            )
            continue
        if not running:
            warnings.append(
                f"session {name}: Claude Code no longer runs — it exited or "
                "its pane closed; removed"
            )
            warnings.extend(_drop_from_fleet(registry, queue, panes, name))
            continue
        try:
            adoption_fields(record)
        except ValueError as exc:
            warnings.append(
                f"session {name}: Claude Code still runs but no broker can "
                f"take it over ({exc}) — removed; its pane is left untouched"
            )
            warnings.extend(_drop_from_fleet(registry, queue, panes, name))
            continue
        record.state = SessionState.UNMANAGED
        line = (
            f"session {name}: Claude Code alive with nothing driving it — "
            "marked unmanaged; recover it with attach_session"
        )
        if record.approved_prompt is None:
            line += (
                " (no approved prompt persisted — reassign_session with a "
                "new task is the route instead)"
            )
        warnings.append(line)
    if registry.records:
        registry.save()
    return warnings
