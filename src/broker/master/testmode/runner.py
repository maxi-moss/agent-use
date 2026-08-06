"""Scenario loader and executor.

``run_scenario`` drives synthetic escalations through the real master socket,
runs dispatches and reassignments through the real runtime, and evaluates each
step's expectation once. Assertions poll the observable state with a bounded
loop and fail loud on timeout; nothing waits on a bare sleep. Any op naming a
session that was never seeded, a socket that was never started, or an
unresolvable expectation raises ``ScenarioError``.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from broker.master.messages import (
    EscalationArrived,
    PermissionEscalationArrived,
    QueueDepthChanged,
)
from broker.master.registry import SessionRecord
from broker.master.runtime import MasterRuntime
from broker.master.testmode.fake_broker import FakeBrokerClient, FakeSessionSocket
from broker.master.testmode.schemas import (
    AssertDepth,
    AssertIsolated,
    AssertNeverSurfaced,
    AssertNoDispatchWrite,
    AssertSurfaced,
    AssertUnreachable,
    Attach,
    Dispatch,
    Escalate,
    PermissionEscalate,
    Retract,
    Scenario,
    ScenarioError,
    ScenarioReport,
    SeedSession,
    StartFakeSocket,
    StepResult,
    StopFakeSocket,
)
from broker.paths import BrokerPaths
from broker.protocol.constants import T_DISPATCH_DECISION
from broker.protocol.schemas import (
    Alternative,
    EscalationPayload,
    PermissionEscalationPayload,
    RaiserIdentity,
    Response,
)

_POLL_INTERVAL_S = 0.01


def load_scenario(path: Path) -> Scenario:
    """Load and validate a scenario file, failing loud on any problem.

    Args:
        path: Scenario JSON file.

    Returns:
        The validated scenario.

    Raises:
        ScenarioError: The file is missing, unreadable, or not a valid
            scenario.
    """
    if not path.exists():
        raise ScenarioError(f"scenario file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ScenarioError(f"cannot read scenario {path}: {exc}") from exc
    try:
        return Scenario.model_validate_json(text)
    except ValidationError as exc:
        raise ScenarioError(f"invalid scenario {path}: {exc}") from exc


def _escalation_payload(step: Escalate) -> EscalationPayload:
    """Build a fully-populated escalation from a step's ids alone."""
    eid = step.escalation_id
    return EscalationPayload(
        escalation_id=eid,
        session_id=step.session,
        task_context=f"task context for {eid}",
        situation=f"situation for {eid}",
        what_was_asked=f"what was asked in {eid}",
        what_is_at_stake=f"what is at stake in {eid}",
        alternatives=[
            Alternative(
                option=f"option a for {eid}",
                pros=f"pros for {eid}",
                cons=f"cons for {eid}",
            )
        ],
        recommendation=f"recommendation for {eid}",
        uncertainty=f"uncertainty for {eid}",
        what_would_change_my_mind=f"what would change my mind for {eid}",
        raiser=RaiserIdentity(component="broker", session_id=step.session),
    )


def _permission_payload(step: PermissionEscalate) -> PermissionEscalationPayload:
    """Build a fully-populated permission escalation from a step."""
    eid = step.escalation_id
    return PermissionEscalationPayload(
        escalation_id=eid,
        session_id=step.session,
        tool_name=step.tool_name,
        tool_input=step.tool_input,
        task_intent=f"task intent for {eid}",
        reason=f"reason for {eid}",
        raised_at=step.raised_at,
        raiser=RaiserIdentity(component="permission", session_id=step.session),
    )


def _surfaced_ids(posts: list[Any]) -> set[str]:
    """Return the ids of every escalation surfaced so far."""
    return {
        m.escalation_id
        for m in posts
        if isinstance(m, (EscalationArrived, PermissionEscalationArrived))
    }


def _latest[T](posts: list[Any], cls: type[T]) -> T | None:
    """Return the most recent post of ``cls``, or ``None``."""
    for m in reversed(posts):
        if isinstance(m, cls):
            return m
    return None


async def _poll_until(
    predicate: Callable[[], bool], timeout_s: float
) -> bool:
    """Poll ``predicate`` until it holds or the deadline passes.

    Args:
        predicate: Observable condition to wait for.
        timeout_s: Bound on how long to wait.

    Returns:
        The final value of the predicate, checked once more after the
        deadline so a slow-but-true condition is not lost.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(_POLL_INTERVAL_S)
    return predicate()


def _check_expect(index: int, op: str, expect: str, resp: Response) -> StepResult:
    """Evaluate an ``ack``/``nack:<code>`` expectation against a reply.

    Raises:
        ScenarioError: ``expect`` is neither ``ack`` nor ``nack:<code>``.
    """
    if expect == "ack":
        passed = resp.ok
        detail = "ACKed" if passed else f"expected ACK, got NACK {resp.payload}"
    elif expect.startswith("nack:"):
        code = expect[len("nack:"):]
        passed = (not resp.ok) and resp.payload.get("reason_code") == code
        detail = (
            f"NACKed {code}"
            if passed
            else f"expected NACK {code}, got ok={resp.ok} {resp.payload}"
        )
    else:
        raise ScenarioError(f"step {index}: invalid expect {expect!r}")
    return StepResult(index=index, op=op, passed=passed, detail=detail)


class _Context:
    """Mutable state threaded through a scenario's steps."""

    def __init__(
        self, runtime: MasterRuntime, posts: list[Any], paths: BrokerPaths
    ) -> None:
        self.runtime = runtime
        self.posts = posts
        self.paths = paths
        self.sockets: dict[str, FakeSessionSocket] = {}
        self.seeded: set[str] = set()

    def require_seeded(self, index: int, session: str) -> None:
        """Raise if ``session`` was never seeded into the registry."""
        if session not in self.seeded:
            raise ScenarioError(
                f"step {index}: session {session!r} was never seeded"
            )


async def run_scenario(
    runtime: MasterRuntime,
    posts: list[Any],
    scenario: Scenario,
    *,
    paths: BrokerPaths,
    step_timeout_s: float = 5.0,
) -> ScenarioReport:
    """Execute a scenario against a live runtime and report each step.

    Args:
        runtime: The runtime under test; its master socket must be bound.
        posts: The capture the runtime's post callable appends to; cleared at
            the start so assertions read only this scenario's traffic.
        scenario: The loaded scenario.
        paths: On-disk locations, used to place fake session sockets.
        step_timeout_s: Bound on every polling assertion.

    Returns:
        The per-step report; ``report.passed`` is the scenario verdict.
    """
    posts.clear()
    ctx = _Context(runtime, posts, paths)
    results: list[StepResult] = []
    try:
        for index, step in enumerate(scenario.steps):
            results.append(await _run_step(index, step, ctx, step_timeout_s))
    finally:
        for sock in ctx.sockets.values():
            await sock.stop()
    return ScenarioReport(name=scenario.name, results=results)


async def _run_step(
    index: int, step: Any, ctx: _Context, timeout_s: float
) -> StepResult:
    """Execute one step and return its result.

    Raises:
        ScenarioError: The step names something that does not exist.
    """
    op = step.op
    runtime = ctx.runtime
    posts = ctx.posts

    if isinstance(step, SeedSession):
        record = SessionRecord(
            name=step.name,
            socket_path=str(ctx.paths.session_socket(step.name)),
            cwd=str(ctx.paths.home),
            anchor_pane=runtime.anchor_pane,
            state=step.state,
            pane_id=step.pane_id,
            claude_session_id=step.claude_session_id,
            transcript_path=step.transcript_path,
            approved_prompt=step.approved_prompt,
        )
        runtime.registry.upsert(record)
        ctx.seeded.add(step.name)
        return StepResult(
            index=index, op=op, passed=True, detail=f"seeded {step.name}"
        )

    if isinstance(step, StartFakeSocket):
        ctx.require_seeded(index, step.session)
        if step.session in ctx.sockets:
            raise ScenarioError(
                f"step {index}: socket for {step.session!r} already started"
            )
        sock = FakeSessionSocket()
        await sock.start(ctx.paths.session_socket(step.session))
        ctx.sockets[step.session] = sock
        return StepResult(
            index=index, op=op, passed=True, detail=f"socket up for {step.session}"
        )

    if isinstance(step, StopFakeSocket):
        ctx.require_seeded(index, step.session)
        sock = ctx.sockets.pop(step.session, None)
        if sock is None:
            raise ScenarioError(
                f"step {index}: no socket started for {step.session!r}"
            )
        await sock.stop()
        return StepResult(
            index=index, op=op, passed=True, detail=f"socket down for {step.session}"
        )

    if isinstance(step, Escalate):
        ctx.require_seeded(index, step.session)
        broker = FakeBrokerClient(
            runtime.master_socket_path, step.session, timeout_s=timeout_s
        )
        resp = await broker.escalate(_escalation_payload(step))
        return _check_expect(index, op, step.expect, resp)

    if isinstance(step, PermissionEscalate):
        ctx.require_seeded(index, step.session)
        broker = FakeBrokerClient(
            runtime.master_socket_path, step.session, timeout_s=timeout_s
        )
        resp = await broker.permission_escalate(_permission_payload(step))
        return _check_expect(index, op, step.expect, resp)

    if isinstance(step, Retract):
        ctx.require_seeded(index, step.session)
        broker = FakeBrokerClient(
            runtime.master_socket_path, step.session, timeout_s=timeout_s
        )
        resp = await broker.retract(step.escalation_id, step.reason)
        passed = resp.ok
        detail = "retracted" if passed else f"retract NACKed: {resp.payload}"
        return StepResult(index=index, op=op, passed=passed, detail=detail)

    if isinstance(step, Dispatch):
        outcome = await runtime.dispatch(step.escalation_id, step.decision)
        if step.expect == "dispatched":
            passed = outcome.startswith("decision dispatched")
        else:
            passed = outcome.startswith("decision NOT dispatched")
        return StepResult(index=index, op=op, passed=passed, detail=outcome)

    if isinstance(step, Attach):
        ctx.require_seeded(index, step.session)
        try:
            detail = await runtime.attach_session(step.session)
            outcome = "attached"
        except (ValueError, RuntimeError) as exc:
            detail = f"refused: {exc}"
            outcome = "refused"
        return StepResult(
            index=index,
            op=op,
            passed=outcome == step.expect,
            detail=detail,
        )

    if isinstance(step, AssertSurfaced):
        eid = step.escalation_id
        passed = await _poll_until(
            lambda: eid in _surfaced_ids(posts), timeout_s
        )
        detail = "surfaced" if passed else f"escalation {eid} never surfaced"
        return StepResult(index=index, op=op, passed=passed, detail=detail)

    if isinstance(step, AssertNeverSurfaced):
        eid = step.escalation_id
        passed = eid not in _surfaced_ids(posts)
        detail = (
            "never surfaced"
            if passed
            else f"escalation {eid} surfaced but should not have"
        )
        return StepResult(index=index, op=op, passed=passed, detail=detail)

    if isinstance(step, AssertDepth):
        want = step.waiting

        def depth_matches() -> bool:
            latest = _latest(posts, QueueDepthChanged)
            return (
                latest is not None
                and latest.depth == step.depth
                and list(latest.waiting) == want
            )

        passed = await _poll_until(depth_matches, timeout_s)
        latest = _latest(posts, QueueDepthChanged)
        seen = (
            f"depth={latest.depth} waiting={list(latest.waiting)}"
            if latest is not None
            else "no queue depth posted"
        )
        detail = (
            seen
            if passed
            else f"expected depth={step.depth} waiting={want}, saw {seen}"
        )
        return StepResult(index=index, op=op, passed=passed, detail=detail)

    if isinstance(step, AssertIsolated):
        violations: list[str] = []
        for m in posts:
            if not isinstance(
                m, (EscalationArrived, PermissionEscalationArrived)
            ):
                continue
            if m.session_id not in m.rendered:
                violations.append(
                    f"{m.escalation_id} omits its own session {m.session_id}"
                )
            for other in ctx.seeded:
                if other != m.session_id and other in m.rendered:
                    violations.append(
                        f"{m.escalation_id} for {m.session_id} names {other}"
                    )
        passed = not violations
        detail = (
            "each surfaced block names only its own session"
            if passed
            else "; ".join(violations)
        )
        return StepResult(index=index, op=op, passed=passed, detail=detail)

    if isinstance(step, AssertNoDispatchWrite):
        ctx.require_seeded(index, step.session)
        sock = ctx.sockets.get(step.session)
        if sock is None:
            raise ScenarioError(
                f"step {index}: no socket started for {step.session!r}"
            )
        wrote = any(e.type == T_DISPATCH_DECISION for e in sock.received)
        passed = not wrote
        detail = (
            "no dispatch decision reached the session socket"
            if passed
            else "a dispatch decision reached the session socket"
        )
        return StepResult(index=index, op=op, passed=passed, detail=detail)

    if isinstance(step, AssertUnreachable):
        ctx.require_seeded(index, step.session)
        rendered = await runtime.render_sessions_with_permission_prompts()
        passed = f"- {step.session}: unreachable" in rendered
        others = [
            other
            for other in ctx.seeded
            if other != step.session
            and f"- {other}: unreachable" in rendered
        ]
        if others:
            passed = False
        detail = (
            f"{step.session} reads unreachable, others reachable"
            if passed
            else f"unreachable check failed; others unreachable: {others}"
        )
        return StepResult(index=index, op=op, passed=passed, detail=detail)

    raise ScenarioError(f"step {index}: unknown op {op!r}")
