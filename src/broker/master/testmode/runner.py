"""Scenario loader and executor.

``run_scenario`` drives synthetic escalations through the real master socket,
runs dispatches and reassignments through the real runtime, and evaluates each
step's expectation once. Assertions poll the observable state with a bounded
loop and fail loud on timeout; nothing waits on a bare sleep. Any op naming a
session that was never seeded or a socket that was never started raises
``ScenarioError``.
"""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, assert_never

from pydantic import ValidationError

from broker.master.viewmodel import (
    EscalationArrived,
    FleetUpdated,
    FleetView,
    PaneEscalationArrived,
)
from broker.master.registry import SessionRecord
from broker.master.runtime import MasterRuntime
from broker.master.testmode.fake_broker import FakeBrokerClient, FakeSessionSocket
from broker.master.testmode.schemas import (
    AssertDepthStep,
    AssertIsolatedStep,
    AssertNeverSurfacedStep,
    AssertNoDispatchWriteStep,
    AssertOpenPanesStep,
    AssertSurfacedStep,
    AssertUnreachableStep,
    AttachStep,
    DeliverStep,
    DispatchStep,
    EscalateStep,
    EscalationRetractStep,
    NackExpectation,
    PaneRetractStep,
    PermissionEscalateStep,
    QuestionEscalateStep,
    Scenario,
    ScenarioError,
    ScenarioReport,
    SeedSessionStep,
    StartSocketStep,
    StepModel,
    StepResult,
    StopSocketStep,
)
from broker.paths import BrokerPaths
from broker.protocol.constants import SessionState, T_DISPATCH_DECISION
from broker.protocol.schemas import (
    Alternative,
    EscalationDisclosure,
    EscalationPayload,
    PermissionEscalationPayload,
    QuestionEscalationPayload,
    Response,
    parse_nack,
)

_POLL_INTERVAL_S = 0.01

SCENARIO_DIR = Path(__file__).parent / "scenarios"


def scenario_names(directory: Path = SCENARIO_DIR) -> list[str]:
    """Return the sorted scenario names available in a directory.

    Args:
        directory: Directory to scan for ``*.json`` scenario files.

    Returns:
        The scenario names (file stems), sorted.

    Raises:
        ScenarioError: The directory does not exist or holds no
            ``*.json`` files.
    """
    if not directory.is_dir():
        raise ScenarioError(f"scenario directory not found: {directory}")
    names = sorted(p.stem for p in directory.glob("*.json"))
    if not names:
        raise ScenarioError(f"no scenario files in {directory}")
    return names


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


def _escalation_payload(step: EscalateStep) -> EscalationPayload:
    """Build a fully-populated escalation from a step's ids alone."""
    eid = step.escalation_id
    return EscalationPayload(
        escalation_id=eid,
        session_id=step.session,
        task_context=f"task context for {eid}",
        disclosure=EscalationDisclosure(
            escalation_title=f"title for {eid}",
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
        ),
    )


def _permission_payload(step: PermissionEscalateStep) -> PermissionEscalationPayload:
    """Build a fully-populated permission escalation from a step."""
    eid = step.escalation_id
    return PermissionEscalationPayload(
        escalation_id=eid,
        session_id=step.session,
        tool_name=step.tool_name,
        tool_input=step.tool_input,
        task_intent=f"task intent for {eid}",
        reason=f"reason for {eid}",
    )


def _question_payload(step: QuestionEscalateStep) -> QuestionEscalationPayload:
    """Build a fully-populated question escalation from a step."""
    eid = step.escalation_id
    return QuestionEscalationPayload(
        escalation_id=eid,
        session_id=step.session,
        task_context=f"task context for {eid}",
        menu=(
            f"## Question 1 (single-select): {step.question} [header for {eid}]\n"
            f"- option a for {eid}: "
        ),
        first_question=step.question,
        reason=f"reason for {eid}",
    )


def _surfaced_ids(posts: list[Any]) -> set[str]:
    """Return the ids of every escalation surfaced so far."""
    return {
        m.escalation_id
        for m in posts
        if isinstance(m, (EscalationArrived, PaneEscalationArrived))
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


def _check_expect(
    index: int, op: str, expect: Literal["ack"] | NackExpectation, resp: Response
) -> StepResult:
    """Evaluate an ``ack``/``nack:<code>`` expectation against a reply."""
    if expect == "ack":
        passed = resp.ok
        detail = "ACKed" if passed else f"expected ACK, got NACK {resp.payload}"
    else:
        code = expect[len("nack:"):]
        passed = (not resp.ok) and parse_nack(resp).reason_code == code
        detail = (
            f"NACKed {code}"
            if passed
            else f"expected NACK {code}, got ok={resp.ok} {resp.payload}"
        )
    return StepResult(index=index, op=op, passed=passed, detail=detail)


def _push_refused(index: int, op: str, resp: Response) -> StepResult:
    """Fail a step whose preceding status push the master NACKed."""
    return StepResult(
        index=index,
        op=op,
        passed=False,
        detail=f"status push NACKed: {resp.payload}",
    )


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

    def broker(self, index: int, session: str, timeout_s: float) -> FakeBrokerClient:
        """Return a fake broker client for a seeded session.

        Raises:
            ScenarioError: ``session`` was never seeded.
        """
        self.require_seeded(index, session)
        return FakeBrokerClient(
            self.runtime.master_socket_path, session, timeout_s=timeout_s
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


async def _run_seed_session(
    index: int, step: SeedSessionStep, ctx: _Context
) -> StepResult:
    """Upsert a session record into the registry."""
    record = SessionRecord(
        name=step.name,
        socket_path=str(ctx.paths.session_socket(step.name)),
        cwd=str(ctx.paths.home),
        anchor_pane=ctx.runtime.anchor_pane,
        state=step.state,
        pane_id=step.pane_id,
        claude_session_id=step.claude_session_id,
        transcript_path=step.transcript_path,
        approved_prompt=step.approved_prompt,
    )
    ctx.runtime.registry.upsert(record)
    ctx.seeded.add(step.name)
    return StepResult(
        index=index, op=step.op, passed=True, detail=f"seeded {step.name}"
    )


async def _run_start_socket(
    index: int, step: StartSocketStep, ctx: _Context
) -> StepResult:
    """Start a fake session socket for a seeded session.

    Raises:
        ScenarioError: A socket for the session is already running.
    """
    ctx.require_seeded(index, step.session)
    if step.session in ctx.sockets:
        raise ScenarioError(
            f"step {index}: socket for {step.session!r} already started"
        )
    sock = FakeSessionSocket()
    await sock.start(ctx.paths.session_socket(step.session))
    ctx.sockets[step.session] = sock
    return StepResult(
        index=index, op=step.op, passed=True, detail=f"socket up for {step.session}"
    )


async def _run_stop_socket(
    index: int, step: StopSocketStep, ctx: _Context
) -> StepResult:
    """Stop the fake session socket for a seeded session.

    Raises:
        ScenarioError: No socket is running for the session.
    """
    ctx.require_seeded(index, step.session)
    sock = ctx.sockets.pop(step.session, None)
    if sock is None:
        raise ScenarioError(
            f"step {index}: no socket started for {step.session!r}"
        )
    await sock.stop()
    return StepResult(
        index=index, op=step.op, passed=True, detail=f"socket down for {step.session}"
    )


async def _run_escalate(
    index: int, step: EscalateStep, ctx: _Context, timeout_s: float
) -> StepResult:
    """Raise a developer escalation and check the expected reply."""
    broker = ctx.broker(index, step.session, timeout_s)
    pushed = await broker.push_status(SessionState.ESCALATED)
    if not pushed.ok:
        return _push_refused(index, step.op, pushed)
    resp = await broker.escalate(_escalation_payload(step))
    return _check_expect(index, step.op, step.expect, resp)


async def _run_permission_escalate(
    index: int, step: PermissionEscalateStep, ctx: _Context, timeout_s: float
) -> StepResult:
    """Raise a permission pane escalation and check the expected reply."""
    broker = ctx.broker(index, step.session, timeout_s)
    resp = await broker.pane_escalate(_permission_payload(step))
    return _check_expect(index, step.op, step.expect, resp)


async def _run_question_escalate(
    index: int, step: QuestionEscalateStep, ctx: _Context, timeout_s: float
) -> StepResult:
    """Raise a question pane escalation and check the expected reply."""
    broker = ctx.broker(index, step.session, timeout_s)
    resp = await broker.pane_escalate(_question_payload(step))
    return _check_expect(index, step.op, step.expect, resp)


async def _run_escalation_retract(
    index: int, step: EscalationRetractStep, ctx: _Context, timeout_s: float
) -> StepResult:
    """Retract a developer escalation."""
    broker = ctx.broker(index, step.session, timeout_s)
    pushed = await broker.push_status(SessionState.DRIVING)
    if not pushed.ok:
        return _push_refused(index, step.op, pushed)
    resp = await broker.escalation_retract(step.escalation_id, step.reason)
    passed = resp.ok
    detail = "retracted" if passed else f"retract NACKed: {resp.payload}"
    return StepResult(index=index, op=step.op, passed=passed, detail=detail)


async def _run_pane_retract(
    index: int, step: PaneRetractStep, ctx: _Context, timeout_s: float
) -> StepResult:
    """Retract a pane escalation."""
    broker = ctx.broker(index, step.session, timeout_s)
    resp = await broker.pane_retract(step.escalation_id, step.reason)
    passed = resp.ok
    detail = "retracted" if passed else f"retract NACKed: {resp.payload}"
    return StepResult(index=index, op=step.op, passed=passed, detail=detail)


async def _run_dispatch(index: int, step: DispatchStep, ctx: _Context) -> StepResult:
    """Dispatch a decision and check whether it was expected to land."""
    outcome = await ctx.runtime.dispatch(step.escalation_id, step.decision)
    if step.expect == "dispatched":
        passed = outcome.startswith("decision dispatched")
    else:
        passed = outcome.startswith("decision NOT dispatched")
    return StepResult(index=index, op=step.op, passed=passed, detail=outcome)


async def _run_deliver(
    index: int, step: DeliverStep, ctx: _Context, timeout_s: float
) -> StepResult:
    """Deliver an escalation to its session."""
    broker = ctx.broker(index, step.session, timeout_s)
    resp = await broker.deliver(step.escalation_id)
    passed = resp.ok
    detail = "delivered" if passed else f"delivery NACKed: {resp.payload}"
    return StepResult(index=index, op=step.op, passed=passed, detail=detail)


async def _run_attach(index: int, step: AttachStep, ctx: _Context) -> StepResult:
    """Attempt to attach to a session and check whether it was refused."""
    ctx.require_seeded(index, step.session)
    try:
        detail = await ctx.runtime.attach_session(step.session)
        outcome = "attached"
    except (ValueError, RuntimeError) as exc:
        detail = f"refused: {exc}"
        outcome = "refused"
    return StepResult(
        index=index,
        op=step.op,
        passed=outcome == step.expect,
        detail=detail,
    )


async def _run_assert_surfaced(
    index: int, step: AssertSurfacedStep, ctx: _Context, timeout_s: float
) -> StepResult:
    """Assert an escalation was surfaced to the developer."""
    eid = step.escalation_id
    passed = await _poll_until(lambda: eid in _surfaced_ids(ctx.posts), timeout_s)
    detail = "surfaced" if passed else f"escalation {eid} never surfaced"
    return StepResult(index=index, op=step.op, passed=passed, detail=detail)


async def _run_assert_never_surfaced(
    index: int, step: AssertNeverSurfacedStep, ctx: _Context
) -> StepResult:
    """Assert an escalation was never surfaced to the developer."""
    eid = step.escalation_id
    passed = eid not in _surfaced_ids(ctx.posts)
    detail = (
        "never surfaced"
        if passed
        else f"escalation {eid} surfaced but should not have"
    )
    return StepResult(index=index, op=step.op, passed=passed, detail=detail)


async def _run_assert_depth(
    index: int, step: AssertDepthStep, ctx: _Context, timeout_s: float
) -> StepResult:
    """Assert the queue depth and waiting set the fleet view last posted."""
    want = step.waiting

    def depth_matches() -> bool:
        latest = _latest(ctx.posts, FleetUpdated)
        return (
            latest is not None
            and latest.view.queue_depth == step.depth
            and list(latest.view.waiting) == want
        )

    passed = await _poll_until(depth_matches, timeout_s)
    latest = _latest(ctx.posts, FleetUpdated)
    seen = (
        f"depth={latest.view.queue_depth} "
        f"waiting={list(latest.view.waiting)}"
        if latest is not None
        else "no fleet view posted"
    )
    detail = (
        seen
        if passed
        else f"expected depth={step.depth} waiting={want}, saw {seen}"
    )
    return StepResult(index=index, op=step.op, passed=passed, detail=detail)


async def _run_assert_open_panes(
    index: int, step: AssertOpenPanesStep, ctx: _Context, timeout_s: float
) -> StepResult:
    """Assert which sessions (and optionally ids) hold an open pane of a kind."""
    want_sessions = step.session_ids
    want_ids = step.ids

    def open_sessions(view: FleetView) -> list[str]:
        return [p.session_id for p in view.panes if p.kind == step.kind]

    def open_ids(view: FleetView) -> list[str]:
        return [p.escalation_id for p in view.panes if p.kind == step.kind]

    def panes_match() -> bool:
        latest = _latest(ctx.posts, FleetUpdated)
        if latest is None or open_sessions(latest.view) != want_sessions:
            return False
        return not want_ids or open_ids(latest.view) == want_ids

    passed = await _poll_until(panes_match, timeout_s)
    latest = _latest(ctx.posts, FleetUpdated)
    seen = (
        f"open {step.kind}={open_sessions(latest.view)} ids={open_ids(latest.view)}"
        if latest is not None
        else "no fleet view posted"
    )
    detail = (
        seen
        if passed
        else f"expected open {step.kind}={want_sessions} ids={want_ids}, saw {seen}"
    )
    return StepResult(index=index, op=step.op, passed=passed, detail=detail)


async def _run_assert_isolated(
    index: int, step: AssertIsolatedStep, ctx: _Context
) -> StepResult:
    """Assert every surfaced block names only its own session."""
    violations: list[str] = []
    for m in ctx.posts:
        if not isinstance(m, (EscalationArrived, PaneEscalationArrived)):
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
    return StepResult(index=index, op=step.op, passed=passed, detail=detail)


async def _run_assert_no_dispatch_write(
    index: int, step: AssertNoDispatchWriteStep, ctx: _Context
) -> StepResult:
    """Assert no dispatch decision reached the session's fake socket.

    Raises:
        ScenarioError: No socket is running for the session.
    """
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
    return StepResult(index=index, op=step.op, passed=passed, detail=detail)


async def _run_assert_unreachable(
    index: int, step: AssertUnreachableStep, ctx: _Context
) -> StepResult:
    """Assert a session reads unreachable and no other seeded session does."""
    ctx.require_seeded(index, step.session)
    rendered = await ctx.runtime.list_sessions()
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
    return StepResult(index=index, op=step.op, passed=passed, detail=detail)


async def _run_step(
    index: int, step: StepModel, ctx: _Context, timeout_s: float
) -> StepResult:
    """Execute one step and return its result.

    Raises:
        ScenarioError: The step names something that does not exist.
    """
    match step:
        case SeedSessionStep():
            return await _run_seed_session(index, step, ctx)
        case StartSocketStep():
            return await _run_start_socket(index, step, ctx)
        case StopSocketStep():
            return await _run_stop_socket(index, step, ctx)
        case EscalateStep():
            return await _run_escalate(index, step, ctx, timeout_s)
        case PermissionEscalateStep():
            return await _run_permission_escalate(index, step, ctx, timeout_s)
        case QuestionEscalateStep():
            return await _run_question_escalate(index, step, ctx, timeout_s)
        case EscalationRetractStep():
            return await _run_escalation_retract(index, step, ctx, timeout_s)
        case PaneRetractStep():
            return await _run_pane_retract(index, step, ctx, timeout_s)
        case DispatchStep():
            return await _run_dispatch(index, step, ctx)
        case DeliverStep():
            return await _run_deliver(index, step, ctx, timeout_s)
        case AttachStep():
            return await _run_attach(index, step, ctx)
        case AssertSurfacedStep():
            return await _run_assert_surfaced(index, step, ctx, timeout_s)
        case AssertNeverSurfacedStep():
            return await _run_assert_never_surfaced(index, step, ctx)
        case AssertDepthStep():
            return await _run_assert_depth(index, step, ctx, timeout_s)
        case AssertOpenPanesStep():
            return await _run_assert_open_panes(index, step, ctx, timeout_s)
        case AssertIsolatedStep():
            return await _run_assert_isolated(index, step, ctx)
        case AssertNoDispatchWriteStep():
            return await _run_assert_no_dispatch_write(index, step, ctx)
        case AssertUnreachableStep():
            return await _run_assert_unreachable(index, step, ctx)
        case _:
            assert_never(step)
