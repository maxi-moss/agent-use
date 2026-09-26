"""SessionBroker: headless driver of one Claude Code session.

Structure:
- `handle()` is the socket handler and does NO slow work: it replies, then
  enqueues. The exceptions are permission_request, ask_question, and
  clarify_escalation, answered inline because the caller blocks on a single reply
  on that connection — they must never queue behind an unrelated turn triage.
- The serial event queue is the only place pane writes happen.
- Classification input is `last_assistant_message` from the Stop payload,
  never the transcript tail. The watchdog reconciliation is the
  one sanctioned pure-transcript read.
- Every pane write is a single `agent prompt`, which submits on its own,
  via driver.agent_prompt with an explicit timeout.
"""

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never, cast

from pydantic import ValidationError

from broker import decision_log
from broker.decision_log import DecisionLogKind, DecisionLogRow
from broker import llm as llm_module
from broker.index.embedding import EmbeddingError
from broker.index.retrieval import RetrievalError
from broker.llm import LLMCallError, LLMCaller, ToolCall
from broker.config import (
    AdoptedSession,
    ResumedTask,
    SessionBrokerConfig,
)
from broker.paths import BrokerPaths
from broker.herdr import driver
from broker.herdr.schemas import AgentStatus
from broker.claude.paths import transcript_dir_for_cwd
from broker.permission import PermissionModule, render_permission_log
from broker.protocol.constants import (
    ACTIVE_STATES,
    ASK_DECISION_ANSWER,
    ASK_USER_QUESTION,
    DECISION_ALLOW,
    DECISION_ESCALATED,
    ENV_BROKER_HOOK_LOG,
    ENV_BROKER_SOCKET,
    HookEventName,
    NackCode,
    SessionState,
    T_APPROVE_PROMPT,
    T_CLARIFY_ESCALATION,
    T_ASK_QUESTION,
    T_DISPATCH_DECISION,
    T_GET_DECISION_LOG,
    T_GET_PERMISSION_LOG,
    T_HOOK_EVENT,
    T_PERMISSION_REQUEST,
    T_REACTIVATE,
    T_SEND_PROMPT,
    T_SHUTDOWN,
    T_STATUS,
)
from broker.protocol.server import serve_unix
from broker.protocol.schemas import (
    Alternative,
    ApprovePromptPayload,
    BudgetUpdatePayload,
    ClarifyEscalationReplyPayload,
    ClarifyEscalationRequestPayload,
    CompletionPayload,
    AskQuestionDecisionPayload,
    AskQuestionRequestPayload,
    DecisionDeliveredPayload,
    DecisionLogPayload,
    DecisionLogRequestPayload,
    DecisionUndeliveredPayload,
    DispatchDecisionPayload,
    Envelope,
    EscalationDisclosure,
    EscalationPayload,
    EscalationRetractPayload,
    FatalErrorPayload,
    HookEventPayload,
    LiveStatusPayload,
    NackPayload,
    PaneRetractPayload,
    PermissionDecisionPayload,
    PermissionLogPayload,
    PermissionLogRequestPayload,
    PermissionRequestPayload,
    PromptProposalPayload,
    PromptUndeliveredPayload,
    QuestionEscalationPayload,
    ReactivatePayload,
    Response,
    RetrievedSymbol,
    SESSION_SOCKET_PAYLOADS,
    SendPromptPayload,
    SessionEndedPayload,
    ShutdownPayload,
    StatusPayload,
    StatusRequestPayload,
    WireMessage,
    nack_response,
)
from broker.session import ask, clarify
from broker.session.grounding import (
    Retriever,
    bind_index_retriever,
    ground_intent,
)
from broker.session.llm_stack import EscalateCall, bind_call_tool, disclosure_of
from broker.session.master_link import (
    LiveStatusPusher,
    describe_refusal,
    send_to_master,
)
from broker.session.triage import AnswerCall, CompleteCall, NoActionCall, triage
from broker.session.watchdog import Watchdog
from broker.transcript.adapter import ReadReport, read_cleaned
from broker.transcript.schemas import (
    AskUserAnswer,
    AssistantText,
    TranscriptEvent,
    UserPrompt,
)

logger = logging.getLogger(__name__)

SUBMIT_TIMEOUT_S = 15.0
SESSION_BIND_TIMEOUT_S = 60.0

# Both waits are named and bounded (global rule). The decision deadline sits
# inside HOOK_WAIT_SECONDS (30) so the hook never gives up while the broker
# still intends to answer — a reply after hook death would be an answer the
# broker believes in and Claude Code never saw.
ASK_DECISION_TIMEOUT_S = 20.0
# updatedInput delivery is instantaneous when it works (duration_ms: 0
# observed); this only fires when the mechanism broke or a hook was dropped.
ASK_VERIFY_TIMEOUT_S = 30.0
# Sits under the master's CLARIFY_ESCALATION_TIMEOUT_S so the broker's own deadline
# expires first and it fails loud on its own terms.
CLARIFY_TIMEOUT_S = 45.0

# Forwarded to the claude binary at spawn. "auto" classifies each tool call and
# still prompts on the risky ones, so the hook's escalation path survives;
# "bypassPermissions" would silently approve every escalation.
CLAUDE_AGENT_ARGS = ["--model", "opus", "--permission-mode", "auto"]

# Dashboard activity phrases, one per LLM call site.
PHRASE_GROUNDING = "constructing the prompt…"
PHRASE_TRIAGE = "reviewing the latest turn…"
PHRASE_PERMISSION = "reviewing a permission request…"
PHRASE_ASK = "deciding a question…"
PHRASE_CLARIFY = "answering a question about the escalation…"

Job = Callable[[], Awaitable[None]]
_Handler = Callable[[Envelope, Any], Awaitable[Response | None]]


@dataclass(frozen=True, slots=True)
class _TaskIntent:
    """The task this broker drives: the raw intent and its approved prompt."""

    raw: str
    approved: str | None  # None: superseded until the developer approves

    @property
    def authoritative(self) -> str:
        """Return the approved prompt if there is one, else the raw intent."""
        return self.approved or self.raw


@dataclass(frozen=True, slots=True)
class _PendingApproval:
    """The prompt proposal the broker is awaiting the developer's approval on."""

    future: asyncio.Future[ApprovePromptPayload]
    payload: PromptProposalPayload


@dataclass(slots=True)
class DecisionEscalation:
    """The one live decision escalation, from its raise until it ends."""

    payload: EscalationPayload
    user_prompt_baseline: int | None  # None: only a dispatched decision resolves it
    clarify_tasks: set[asyncio.Task[clarify.ClarifyCall]]


@dataclass(slots=True)
class _OpenMenu:
    """An AskUserQuestion picker the broker left open in the pane."""

    tool_use_id: str
    escalation_id: str | None  # None: nothing escalated for it
    injected: dict[str, ask.AnswerValue] | None  # unverified injected answers


@dataclass(frozen=True, slots=True)
class _InjectedAnswers:
    """Answers the broker injected into a menu, awaiting verification."""

    tool_input: dict[str, Any]
    answers: dict[str, ask.AnswerValue]


class PaneOccupiedError(Exception):
    def __init__(self, what: str, pane_id: str | None) -> None:
        """Record which native prompt holds the pane."""
        super().__init__(
            f"{what} is open in pane {pane_id or '?'}; nothing is typed into "
            "the pane while it is"
        )
        self.what = what
        self.pane_id = pane_id


class FatalSessionError(Exception):
    def __init__(self, error_class: str, detail: str) -> None:
        """Record the machine-readable class and human detail of the failure."""
        super().__init__(f"{error_class}: {detail}")
        self.error_class = error_class
        self.detail = detail


class SessionBroker:
    def __init__(
        self,
        cfg: SessionBrokerConfig,
        *,
        llm_call: LLMCaller[ToolCall] | None = None,
        retrieve: Retriever | None = None,
        permission: PermissionModule | None = None,
    ) -> None:
        """Build the broker's state without touching the socket or the pane.

        Args:
            cfg: Session identity, socket paths, cwd, intent and budget limits.
            llm_call: Injected tool-calling backend. When omitted, ``run()``
                builds one from an Anthropic client — tests pass a fake.
            retrieve: Injected code retrieval. When omitted, ``run()`` binds
                the repository index under the broker home — tests pass a fake.
            permission: Injected permission triage module. When omitted, one is
                built from ``cfg`` — tests pass a fake or a spy.
        """
        self.cfg = cfg
        self._llm_call = llm_call
        self._retrieve = retrieve
        self.state: SessionState = SessionState.SPAWNING
        self.pane_id: str | None = None
        self.claude_session_id: str | None = None
        self.transcript_path: str | None = None
        self._logged_drift_warnings: set[str] = set()
        self._logged_drift_versions: set[frozenset[str]] = set()
        self.budget_count = cfg.budget_count
        self._task = _TaskIntent(cfg.intent, None)
        self.task_activity: str = ""

        self.session_bound = asyncio.Event()
        self._pending: _PendingApproval | None = None
        self.jobs: asyncio.Queue[Job] = asyncio.Queue()
        self._run_task: asyncio.Task[None] | None = None

        self._decision_escalation: DecisionEscalation | None = None
        self._open_menu: _OpenMenu | None = None
        self._ask_decisions: dict[str, AskQuestionDecisionPayload] = {}
        self._ask_expected: dict[str, _InjectedAnswers] = {}
        self._ask_verify_tasks: dict[str, asyncio.Task[None]] = {}
        self._permission_prompt_pending = False
        self._last_event_count = -1

        self._status = LiveStatusPusher(
            snapshot=self._live_status, send=self._to_master
        )

        self._paths = BrokerPaths(cfg.broker_home)
        self.decision_log_path = self._paths.session_decisions(cfg.name)
        self.permission_log_path = self._paths.session_permissions(cfg.name)
        self.permission = permission or PermissionModule(
            cfg.classifier,
            session_id=cfg.name,
            master_socket_path=cfg.master_socket_path,
            log_path=self.permission_log_path,
            intent=self._task.authoritative,
        )
        self.watchdog = Watchdog(
            cfg.watchdog_seconds, self._herdr_state, self._reconcile
        )
        self._handlers: dict[str, _Handler] = {
            T_PERMISSION_REQUEST: self._on_permission_request,
            T_ASK_QUESTION: self._on_ask_question,
            T_CLARIFY_ESCALATION: self._on_clarify_escalation,
            T_HOOK_EVENT: self._on_hook_event,
            T_APPROVE_PROMPT: self._on_approve_prompt,
            T_DISPATCH_DECISION: self._on_dispatch_decision,
            T_REACTIVATE: self._on_reactivate,
            T_SEND_PROMPT: self._on_send_prompt,
            T_STATUS: self._on_status,
            T_GET_DECISION_LOG: self._on_get_decision_log,
            T_GET_PERMISSION_LOG: self._on_get_permission_log,
            T_SHUTDOWN: self._on_shutdown,
        }

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Serve the socket and drive the session until it is stopped."""
        # Bind FIRST — hooks may fire before the pane exists.
        server = await serve_unix(Path(self.cfg.socket_path), self.handle)
        if self._llm_call is None:
            self._llm_call = bind_call_tool(llm_module.build_client())
        if self._retrieve is None:
            self._retrieve = bind_index_retriever(self._paths, self.cfg.embedding)
        self.watchdog.start()
        self._status.start()
        drive = self._run_task = asyncio.create_task(self._drive())
        try:
            await drive
        except asyncio.CancelledError:
            # A requested stop cancels only the drive task; cancellation of
            # run() itself must propagate.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        finally:
            await self._status.aclose()
            for task in list(self._ask_verify_tasks.values()):
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await self.permission.aclose()
            await self.watchdog.stop()
            server.close()
            await server.wait_closed()

    async def _drive(self) -> None:
        """Launch the session, then run queued jobs one at a time."""
        try:
            await self._launch()
            await self._event_loop()
        except FatalSessionError as exc:
            await self._fatal(exc.error_class, exc.detail)
        except Exception as exc:  # fail loud to the master, never to stderr
            await self._fatal(type(exc).__name__, str(exc))

    def _request_stop(self) -> None:
        """Stop driving the session once the current reply has been written."""
        assert self._run_task is not None  # set before any handler runs
        asyncio.get_running_loop().call_soon(self._run_task.cancel)

    async def _launch(self) -> None:
        """Take over or start a Claude session, then resume or ground the task.

        Raises:
            FatalSessionError: A fresh start saw no SessionStart hook event
                within ``SESSION_BIND_TIMEOUT_S``.
        """
        if self.cfg.resume is not None:
            assert self.cfg.adopt is not None  # enforced by config validation
            self._adopt(self.cfg.adopt)
            self._resume(self.cfg.resume)
            return
        if self.cfg.adopt is not None:
            self._adopt(self.cfg.adopt)
        else:
            await self._start_session()
        await self._ground_and_submit(self.cfg.intent)

    def _adopt(self, adopt: AdoptedSession) -> None:
        """Take over a Claude session a previous broker was driving.

        The pane, chat and transcript already exist and SessionStart fired for
        the outgoing broker, so nothing is started and nothing is waited for.

        Args:
            adopt: Pane, Claude session id and transcript path to take over.
        """
        self.pane_id = adopt.pane_id
        self.claude_session_id = adopt.claude_session_id
        self.transcript_path = adopt.transcript_path
        self.session_bound.set()
        self._log(
            DecisionLogKind.ADOPTED,
            "reassigned to a session that was already running",
            f"pane={adopt.pane_id} claude_session={adopt.claude_session_id}",
        )

    def _resume(self, resume: ResumedTask) -> None:
        """Pick a persisted task back up where the previous broker left it.

        No grounding, no proposal, no approval wait: the developer already
        approved this prompt, and re-grounding it would propose it a second
        time.

        Args:
            resume: The persisted approved prompt and completed-ness.
        """
        self._set_task(self._task.raw, resume.approved_prompt)
        self._log(
            DecisionLogKind.RESUMED,
            "attached to a session whose previous broker is gone",
            f"completed={resume.completed}",
        )
        self._set_state(
            SessionState.COMPLETED if resume.completed else SessionState.DRIVING
        )

    async def _start_session(self) -> None:
        """Split a pane, start Claude in it, and wait for the session to bind.

        Raises:
            FatalSessionError: No SessionStart hook event arrived within
                ``SESSION_BIND_TIMEOUT_S``.
        """
        cfg = self.cfg
        # Trust was seeded by the MASTER before spawn.
        pane = await asyncio.to_thread(
            driver.pane_split,
            cfg.anchor_pane,
            direction="right",
            cwd=Path(cfg.cwd),
            env={
                ENV_BROKER_SOCKET: cfg.socket_path,
                ENV_BROKER_HOOK_LOG: str(self._paths.session_hook_log(cfg.name)),
            },
            focus=False,
            timeout_s=15.0,
        )
        self.pane_id = pane.pane_id  # the DURABLE handle
        start = await asyncio.to_thread(
            driver.agent_start,
            cfg.name,
            kind="claude",
            pane_id=self.pane_id,
            timeout_ms=30000,
            agent_args=[
                *CLAUDE_AGENT_ARGS,
                "--settings",
                cfg.claude_settings_path,
            ],
        )
        session = start.agent_session
        # Optimistic fill-in only: the SessionStart hook may already have
        # bound the authoritative id by the time this returns, and must
        # never be clobbered by agent_start's guess.
        if (
            session is not None
            and session.kind == "id"
            and self.claude_session_id is None
        ):
            self.claude_session_id = session.value
        try:
            async with asyncio.timeout(SESSION_BIND_TIMEOUT_S):
                # Primary binding is the SessionStart hook event.
                await self.session_bound.wait()
        except TimeoutError:
            raise FatalSessionError(
                "session_start_timeout",
                f"no SessionStart hook event within {SESSION_BIND_TIMEOUT_S:.0f} s",
            ) from None

    async def _ground_and_submit(self, intent: str) -> None:
        """Ground an intent into a prompt, await approval, and submit it.

        Approval is synchronous and blocking, with no timeout — the broker
        never decides the opening prompt of a task for the developer.

        Args:
            intent: Raw task intent, superseding whatever this broker was
                driving before.

        Raises:
            FatalSessionError: Retrieval, embedding, or the grounding call
                failed. The spawn aborts; nothing degraded is proposed.
        """
        self._set_task(intent, None)
        self.task_activity = ""
        self._set_state(SessionState.GROUNDING)
        assert self._llm_call is not None
        assert self._retrieve is not None
        with self._status.activity(PHRASE_GROUNDING):
            try:
                grounding = await ground_intent(
                    self._llm_call,
                    self.cfg.session_model,
                    retrieve=self._retrieve,
                    intent=intent,
                    cwd=Path(self.cfg.cwd),
                )
            except (RetrievalError, EmbeddingError, LLMCallError) as exc:
                raise FatalSessionError(type(exc).__name__, str(exc)) from exc
        loop = asyncio.get_running_loop()
        approval: asyncio.Future[ApprovePromptPayload] = loop.create_future()
        payload = PromptProposalPayload(
            proposal_id=uuid.uuid4().hex,
            proposed_prompt=grounding.proposal.prompt,
            grounding_summary=grounding.proposal.reasoning,
            retrieved=[
                RetrievedSymbol(name=s.qualified_name, score=s.score)
                for s in grounding.context.symbols
            ],
        )
        self._pending = _PendingApproval(approval, payload)
        self._set_state(SessionState.AWAITING_APPROVAL)
        await self._to_master(payload)
        # Approval is synchronous and blocking — no timeout.
        approved = await approval
        self._set_task(intent, approved.prompt)
        await self._submit(approved.prompt)
        self._pending = None
        self._set_state(SessionState.DRIVING)

    def _live_status(self) -> LiveStatusPayload:
        """Return the session's current live status."""
        return LiveStatusPayload(
            state=self.state,
            activity=self._status.current_activity,
            permission_prompt=self._reports_permission_prompt(),
            task_activity=self.task_activity,
            pane_id=self.pane_id,
            claude_session_id=self.claude_session_id,
            transcript_path=self.transcript_path,
        )

    async def _event_loop(self) -> None:
        """Run queued jobs one at a time until the drive task is cancelled."""
        while True:
            job = await self.jobs.get()
            try:
                await job()
            except FatalSessionError as exc:
                await self._fatal(exc.error_class, exc.detail)
            except Exception as exc:  # fail loud, keep serving the socket
                await self._fatal(type(exc).__name__, str(exc))

    # ── socket handler: reply, then enqueue — never slow work ─────────────

    async def handle(self, env: Envelope) -> Response | None:
        """Validate one socket envelope and run its message's handler.

        Args:
            env: Decoded envelope from the master or the hook client.

        Returns:
            The handler's reply, the NACK refusing an unknown type or a
            malformed payload, or ``None`` for fire-and-forget messages.
        """
        validate = SESSION_SOCKET_PAYLOADS.get(env.type)
        if validate is None:
            logger.error("unknown message type %r", env.type)
            return nack_response(
                env, f"unknown message type {env.type!r}", NackCode.MALFORMED
            )
        try:
            payload = validate(env.payload)
        except ValidationError as exc:
            logger.error("invalid %s payload: %s", env.type, exc)
            return nack_response(
                env, f"malformed {env.type}: {exc}", NackCode.MALFORMED
            )
        return await self._handlers[env.type](env, payload)

    async def _on_permission_request(
        self, env: Envelope, payload: PermissionRequestPayload
    ) -> Response:
        """Judge a permission request and reply with the decision.

        The triage module is awaited right here rather than enqueued. Each
        accepted socket connection runs in its own task, so waiting blocks only
        the tool call being judged; putting it on the serial queue would stall
        the whole coding session behind an unrelated turn classification.

        Args:
            env: Envelope the request arrived in.
            payload: The tool call awaiting a permission decision.

        Returns:
            The decision reply for the hook.
        """
        with self._status.activity(PHRASE_PERMISSION):
            decision = await self.permission.decide(
                payload.tool_name,
                payload.tool_input,
                payload.permission_suggestions,
            )
        if decision == DECISION_ALLOW:
            reply = PermissionDecisionPayload(decision=DECISION_ALLOW)
        else:
            reply = PermissionDecisionPayload(decision=DECISION_ESCALATED)
        return Response(id=env.id, ok=True, payload=reply.model_dump())

    async def _on_ask_question(
        self, env: Envelope, payload: AskQuestionRequestPayload
    ) -> Response:
        """Decide a pending AskUserQuestion and reply answer-or-escalated.

        Args:
            env: Envelope the question arrived in.
            payload: The pending question.

        Returns:
            The decision reply for the hook.
        """
        self.watchdog.reset()
        # Cached per tool_use_id: PreToolUse can fire several times per logical
        # operation, and a duplicate must get the same reply without a second
        # LLM call.
        cached = self._ask_decisions.get(payload.tool_use_id)
        if cached is None:
            with self._status.activity(PHRASE_ASK):
                cached = await self._decide_ask(payload)
            self._ask_decisions[payload.tool_use_id] = cached
        return Response(id=env.id, ok=True, payload=cached.model_dump())

    async def _decide_ask(
        self, payload: AskQuestionRequestPayload
    ) -> AskQuestionDecisionPayload:
        """Run the answer-or-escalate decision for one pending menu.

        Every failure arm returns ``escalated`` — the safe default that hands
        the menu to the developer rather than answering it. Every ``escalated``
        reply claims the open picker before returning, so no pane write can
        slip in while it renders.

        Args:
            payload: The pending question payload from the hook.

        Returns:
            The reply payload to cache and send to the hook.
        """
        tool_use_id = payload.tool_use_id
        escalated = AskQuestionDecisionPayload(decision=DECISION_ESCALATED)
        if self.state not in ACTIVE_STATES:
            # The developer is already engaged (escalated/completed/...): do
            # not raise a second escalation on top; let the picker render.
            self._log(
                DecisionLogKind.ASK_SKIPPED,
                f"question arrived in state {self.state!r}; picker left to "
                "the developer",
                tool_use_id=tool_use_id,
            )
            self._claim_menu(_OpenMenu(tool_use_id, None, None))
            return escalated
        try:
            questions = ask.parse_questions(payload.tool_input)
        except ask.AskInputError as exc:
            self._escalate_menu(
                tool_use_id, payload.tool_input, f"unusable question payload: {exc}"
            )
            return escalated
        if self.budget_count >= self.cfg.budget_max:
            self._escalate_menu(
                tool_use_id,
                payload.tool_input,
                "autonomy budget exhausted — this question is handed over "
                "rather than answered",
            )
            return escalated
        assert self._llm_call is not None
        try:
            async with asyncio.timeout(ASK_DECISION_TIMEOUT_S):
                events = self._read_transcript()
                result = await ask.decide_questions(
                    self._llm_call,
                    self.cfg.session_model,
                    intent=self._intent(),
                    events=events,
                    questions=questions,
                )
        except Exception as exc:
            # Deliberately broad: LLMCallError, AnswerValidationError (both
            # attempts), timeout, and transcript failure all escalate rather
            # than narrowing to one type and dropping the rest.
            self._escalate_menu(
                tool_use_id, payload.tool_input, f"{type(exc).__name__}: {exc}"
            )
            return escalated
        if isinstance(result, EscalateCall):
            self._escalate_menu(
                tool_use_id,
                payload.tool_input,
                result.reasoning,
                analysis=disclosure_of(result),
                task_summary=result.task_summary,
            )
            return escalated
        call, answers = result
        updated_input = dict(payload.tool_input)
        updated_input["answers"] = answers
        self._log(DecisionLogKind.ASK_ANSWERED, call.reasoning, json.dumps(answers))
        self._ask_expected[tool_use_id] = _InjectedAnswers(
            payload.tool_input, answers
        )
        self._spawn_ask_verify(tool_use_id)
        self.budget_count += 1
        # Queued, and captured by value: the hook is blocking on this return,
        # so master traffic never runs on the reply path, and the job must
        # report the count as of this answer, not whatever it is when the
        # queue drains.
        count = self.budget_count
        self.jobs.put_nowait(
            lambda: self._report_budget(count)
        )
        return AskQuestionDecisionPayload(
            decision=ASK_DECISION_ANSWER, updated_input=updated_input
        )

    async def _on_clarify_escalation(
        self, env: Envelope, req: ClarifyEscalationRequestPayload
    ) -> Response:
        """Answer a read-only question about the live escalation, inline.

        Reuses the broker's own LLM seam; never resolves the escalation or
        writes to the pane. Bound to escalation liveness: a retract or dispatch
        landing during the LLM call cancels it and the answer is dropped as
        resolved in the pane. Each accepted connection is its own task, so this
        await blocks only this connection.

        Args:
            env: Envelope the question arrived in.
            req: The developer's question about the live escalation.

        Returns:
            The answer reply, or an ``ok=False`` reply naming why no answer is
            given (escalation not live, resolved under the call, or LLM error).
        """
        escalation = self._decision_escalation
        if (
            self.state != SessionState.ESCALATED
            or escalation is None
            or escalation.payload.escalation_id != req.escalation_id
        ):
            return nack_response(
                env, "escalation no longer live", NackCode.WRONG_STATE
            )
        resolved = nack_response(
            env, "escalation resolved in the pane", NackCode.WRONG_STATE
        )
        assert self._llm_call is not None
        try:
            events = self._read_transcript()
        except Exception as exc:
            self._log(
                DecisionLogKind.CLARIFY_FAILED,
                f"{type(exc).__name__}: {exc}",
                escalation_id=req.escalation_id,
            )
            return nack_response(env, f"{type(exc).__name__}: {exc}", None)
        task = asyncio.create_task(
            clarify.clarify(
                self._llm_call,
                self.cfg.session_model,
                intent=self._intent(),
                escalation=escalation.payload,
                question=req.question,
                events=events,
            )
        )
        escalation.clarify_tasks.add(task)
        try:
            with self._status.activity(PHRASE_CLARIFY):
                async with asyncio.timeout(CLARIFY_TIMEOUT_S):
                    result = await task
        except asyncio.CancelledError:
            # The task is cancelled either way; only the connection task's own
            # cancellation (teardown) must propagate.
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return resolved
        except Exception as exc:
            self._log(
                DecisionLogKind.CLARIFY_FAILED,
                f"{type(exc).__name__}: {exc}",
                escalation_id=req.escalation_id,
            )
            return nack_response(env, f"{type(exc).__name__}: {exc}", None)
        finally:
            # Discard only: awaiting `task` forwards this connection task's
            # cancellation — the timeout's included — into it, so it is
            # always finished by the time control reaches here.
            escalation.clarify_tasks.discard(task)
        # Cancelling a task that already finished is a no-op, so an escalation
        # ending between the answer and this line still has to drop it.
        if (
            self.state != SessionState.ESCALATED
            or self._decision_escalation is not escalation
        ):
            return resolved
        self._log(DecisionLogKind.CLARIFIED, result.reasoning, result.answer)
        return Response(
            id=env.id,
            ok=True,
            payload=ClarifyEscalationReplyPayload(answer=result.answer).model_dump(),
        )

    def _escalate_menu(
        self,
        tool_use_id: str,
        tool_input: dict[str, Any],
        reason: str,
        *,
        analysis: EscalationDisclosure | None = None,
        task_summary: str = "Handed a pending AskUserQuestion menu to the developer",
    ) -> None:
        """Claim the open picker now and queue its question escalation.

        Args:
            tool_use_id: The pending AskUserQuestion call.
            tool_input: Its raw tool input.
            reason: Why the broker did not answer the menu itself.
            analysis: The ask LLM's disclosure when it chose to escalate.
            task_summary: The escalation's one-line Reason in the outcome history.
        """
        escalation_id = uuid.uuid4().hex
        self._claim_menu(_OpenMenu(tool_use_id, escalation_id, None))
        # Queued: the hook blocks on this reply and must never wait on master
        # traffic.
        self.jobs.put_nowait(
            lambda: self._raise_question(
                escalation_id,
                tool_input,
                reason,
                analysis,
                task_summary=task_summary,
            )
        )

    def _claim_menu(self, menu: _OpenMenu) -> None:
        """Record ``menu`` as the picker open in the pane."""
        previous = self._open_menu
        self._open_menu = menu
        self._status.mark_dirty()
        # A picker opening means the earlier one closed. The master supersedes
        # its escalation on the new raise, so only the log row is written,
        # queued behind the earlier raise so it follows that raise's row.
        if previous is not None and previous.escalation_id is not None:
            replaced = previous.escalation_id
            self.jobs.put_nowait(lambda: self._log_replaced_question(replaced))

    async def _on_hook_event(
        self, env: Envelope, hook: HookEventPayload
    ) -> None:
        """Route one fire-and-forget Claude Code hook event.

        Args:
            env: Envelope the event arrived in.
            hook: The hook event.
        """
        self.watchdog.reset()
        await self._dispatch_hook(hook)

    async def _on_approve_prompt(
        self, env: Envelope, approved: ApprovePromptPayload
    ) -> Response:
        """Release the pending proposal with the prompt the developer approved.

        Args:
            env: Envelope the approval arrived in.
            approved: The approved prompt and the proposal it answers.

        Returns:
            The ACK, or a ``STALE_PROPOSAL`` NACK when it answers no pending
            proposal.
        """
        if (
            self._pending is None
            or self._pending.future.done()
            or approved.proposal_id != self._pending.payload.proposal_id
        ):
            logger.warning("stale approve_prompt ignored")
            return nack_response(env, "stale proposal", NackCode.STALE_PROPOSAL)
        self._pending.future.set_result(approved)
        return Response(id=env.id, ok=True)

    async def _on_dispatch_decision(
        self, env: Envelope, decision: DispatchDecisionPayload
    ) -> Response:
        """Queue delivery of the developer's decision into the pane.

        Args:
            env: Envelope the decision arrived in.
            decision: The developer's decision on an escalation.

        Returns:
            The ACK.
        """
        self.jobs.put_nowait(lambda: self._deliver_decision(decision))
        return Response(id=env.id, ok=True)

    async def _on_reactivate(
        self, env: Envelope, reactivate: ReactivatePayload
    ) -> Response:
        """Queue a new task for a session that has completed its last one.

        Args:
            env: Envelope the request arrived in.
            reactivate: The new task's intent.

        Returns:
            The ACK, or a ``WRONG_STATE`` NACK unless the session is completed.
        """
        if self.state != SessionState.COMPLETED:
            return nack_response(
                env,
                f"session is {self.state!r}, not 'completed' — reassign a "
                "new broker instead of displacing the task this one is "
                "still driving",
                NackCode.WRONG_STATE,
            )
        # Closes the gate here, not in the job: a second reactivate
        # arriving before the queue drains must not pass it too.
        self._set_state(SessionState.GROUNDING)
        self.jobs.put_nowait(lambda: self._reactivate(reactivate))
        return Response(id=env.id, ok=True)

    async def _on_send_prompt(
        self, env: Envelope, prompt: SendPromptPayload
    ) -> Response:
        """Queue the developer's prompt for typing into the pane.

        Args:
            env: Envelope the prompt arrived in.
            prompt: The developer's prompt.

        Returns:
            The ACK, or a ``WRONG_STATE`` NACK when the broker no longer
            drives the pane or a native prompt holds it.
        """
        if self.state in (SessionState.ERROR, SessionState.STOPPED):
            return nack_response(
                env,
                f"session is {self.state!r} — this broker no longer drives "
                "its pane",
                NackCode.WRONG_STATE,
            )
        what = self._native_prompt()
        if what is not None:
            return nack_response(
                env,
                f"{what} is open in pane {self.pane_id or '?'} — the "
                "developer answers it there before a prompt can be typed",
                NackCode.WRONG_STATE,
            )
        self.jobs.put_nowait(lambda: self._send_developer_prompt(prompt))
        return Response(id=env.id, ok=True)

    async def _on_status(self, env: Envelope, p: StatusRequestPayload) -> Response:
        """Reply with the session's current status snapshot.

        Args:
            env: Envelope the request arrived in.
            p: The empty status request.

        Returns:
            The status reply.
        """
        live = self._live_status()
        return Response(
            id=env.id,
            ok=True,
            payload=StatusPayload(
                state=live.state,
                permission_prompt=live.permission_prompt,
                task_activity=live.task_activity,
                pending_proposal=(
                    self._pending.payload if self._pending is not None else None
                ),
            ).model_dump(),
        )

    async def _on_get_decision_log(
        self, env: Envelope, p: DecisionLogRequestPayload
    ) -> Response:
        """Reply with the rendered decision log.

        Args:
            env: Envelope the request arrived in.
            p: The empty decision-log request.

        Returns:
            The decision-log reply.
        """
        return Response(
            id=env.id,
            ok=True,
            payload=DecisionLogPayload(
                text=decision_log.render_decision_log(self.decision_log_path)
            ).model_dump(),
        )

    async def _on_get_permission_log(
        self, env: Envelope, p: PermissionLogRequestPayload
    ) -> Response:
        """Reply with the rendered permission log.

        Args:
            env: Envelope the request arrived in.
            p: The empty permission-log request.

        Returns:
            The permission-log reply.
        """
        return Response(
            id=env.id,
            ok=True,
            payload=PermissionLogPayload(
                text=render_permission_log(self.permission_log_path)
            ).model_dump(),
        )

    async def _on_shutdown(self, env: Envelope, p: ShutdownPayload) -> Response:
        """Stop serving and driving the session.

        Args:
            env: Envelope the request arrived in.
            p: The empty shutdown request.

        Returns:
            The ACK.
        """
        self._request_stop()
        return Response(id=env.id, ok=True)

    async def _dispatch_hook(self, hook: HookEventPayload) -> None:
        """Route one Claude Code hook event to state changes, queued jobs or a stop.

        ``Stop`` carries the classification input as ``last_assistant_message``,
        never the transcript tail. ``StopFailure`` is surfaced as fatal, not
        treated as a completed turn.

        Args:
            hook: Validated hook payload; ``raw`` is the untyped hook JSON.
        """
        raw = hook.raw
        try:
            event = HookEventName(hook.hook_event_name)
        except ValueError:
            logger.warning("unrecognized hook event %r", hook.hook_event_name)
            return
        match event:
            case HookEventName.SESSION_START:
                self._bind_session(raw)
            case HookEventName.STOP:
                self._set_perm_pending(False)
                message = str(raw.get("last_assistant_message", "") or "")
                self.jobs.put_nowait(lambda: self._on_turn_end(message))
            case HookEventName.STOP_FAILURE:
                error_class = str(
                    raw.get("matcher") or raw.get("error") or "stop_failure"
                )
                detail = str(raw.get("message") or raw)
                # Surfaced, NOT a completed turn.
                self.jobs.put_nowait(lambda: self._fatal(error_class, detail))
            case HookEventName.USER_PROMPT_SUBMIT | HookEventName.POST_TOOL_USE:
                if event == HookEventName.POST_TOOL_USE:
                    self._set_perm_pending(False)
                    tool_name, tool_input = _raw_tool(raw)
                    self.permission.note_tool_completed(tool_name, tool_input)
                    if tool_name == ASK_USER_QUESTION:
                        self._verify_ask(raw)
                else:
                    self.permission.note_developer_input()
                if self._open_menu is not None:
                    self.jobs.put_nowait(self._check_menu_answered)
                if self.state == SessionState.ESCALATED:
                    self.jobs.put_nowait(self._check_out_of_band_resolution)
            case HookEventName.NOTIFICATION:
                self._log(DecisionLogKind.NOTIFICATION, "", str(raw.get("message", "")))
                if raw.get("notification_type") == "permission_prompt":
                    self._set_perm_pending(True)
            case HookEventName.SESSION_END:
                self.permission.note_session_ended()
                self._set_state(SessionState.STOPPED)
                self._log(DecisionLogKind.SESSION_END, "", "SessionEnd hook received")
                # Inline, not queued: the terminal report to master must go
                # now, not wait behind whatever else is already queued.
                await self._on_session_end()
            case HookEventName.PRE_COMPACT | HookEventName.POST_COMPACT:
                self._log(DecisionLogKind.COMPACTION, "", event)  # continue normally
            case HookEventName.PRE_TOOL_USE | HookEventName.PERMISSION_REQUEST:
                # The hook routes these as permission_request/ask_question,
                # never as a hook_event; _dispatch_hook should never see them.
                logger.error("hook never routes %s through _dispatch_hook", event)
            case _:
                assert_never(event)

    def _bind_session(self, raw: dict[str, Any]) -> None:
        """Bind the Claude session id and transcript path from SessionStart.

        Sets ``session_bound`` unconditionally so ``_launch()`` stops waiting.

        Args:
            raw: Raw SessionStart hook JSON; ``session_id`` and
                ``transcript_path`` are read when present and non-empty.
        """
        session_id = raw.get("session_id")
        if isinstance(session_id, str) and session_id:
            self.claude_session_id = session_id
        transcript = raw.get("transcript_path")
        if isinstance(transcript, str) and transcript:
            self.transcript_path = transcript
        elif self.transcript_path is None and self.claude_session_id:
            # cwd derivation is the fallback, not an error.
            self.transcript_path = str(
                transcript_dir_for_cwd(Path(self.cfg.cwd))
                / f"{self.claude_session_id}.jsonl"
            )
        self.session_bound.set()
        self._status.mark_dirty()

    # ── queued jobs ───────────────────────────────────────────────────────

    async def _triage_turn(self, last_assistant_message: str) -> None:
        """Triage one turn boundary into an answer, escalation or completion.

        Only runs while driving. The transcript is read for context only — the
        classification input is the message itself. An exhausted budget
        converts an answer into a handover escalation rather than sending it
        silently.

        Args:
            last_assistant_message: ``last_assistant_message`` from the Stop
                payload, or the last assistant text when the watchdog
                reconciles.
        """
        if self.state not in ACTIVE_STATES:
            self._log(
                DecisionLogKind.NO_ACTION,
                f"turn boundary ignored in state {self.state!r}",
                "",
            )
            return
        self._set_state(SessionState.DRIVING)
        events = self._read_transcript()  # context ONLY; input is the message
        self._last_event_count = len(events)
        assert self._llm_call is not None
        with self._status.activity(PHRASE_TRIAGE):
            result = await triage(
                self._llm_call,
                self.cfg.session_model,
                intent=self._intent(),
                events=events,
                event_name=HookEventName.STOP,
                last_assistant_message=last_assistant_message,
            )
        if not isinstance(result, EscalateCall):
            self.task_activity = result.task_activity
            self._status.mark_dirty()
        if isinstance(result, AnswerCall):
            if self.budget_count >= self.cfg.budget_max:
                await self._escalate_handover(result, last_assistant_message, events)
            else:
                self._log(
                    DecisionLogKind.ANSWERED,
                    result.reasoning,
                    result.answer,
                    task_summary=result.task_summary,
                )
                await self._submit(result.answer)
                self.budget_count += 1
                await self._report_budget(self.budget_count)
        elif isinstance(result, EscalateCall):
            await self._raise_escalation(
                self._new_escalation(disclosure_of(result)),
                result.reasoning,
                events,
                task_summary=result.task_summary,
            )
        elif isinstance(result, CompleteCall):
            self._log(
                DecisionLogKind.COMPLETED,
                result.reasoning,
                "",
                task_summary=result.task_summary,
                headline=result.headline,
                supporting=result.supporting,
            )
            self._set_state(SessionState.COMPLETED)  # stop driving; keep serving
            await self._to_master(
                CompletionPayload(
                    headline=result.headline, supporting=result.supporting
                )
            )
        elif isinstance(
            result, NoActionCall  # pyright: ignore[reportUnnecessaryIsInstance]
        ):
            self._log(DecisionLogKind.NO_ACTION, result.reasoning, "")

    def _new_escalation(self, disclosure: EscalationDisclosure) -> EscalationPayload:
        """Build a decision escalation carrying this session's identifying preamble.

        Args:
            disclosure: The analysis the developer decides on.

        Returns:
            The escalation, ready to raise.
        """
        return EscalationPayload(
            escalation_id=uuid.uuid4().hex,
            session_id=self.cfg.name,
            task_context=self._intent(),
            disclosure=disclosure,
        )

    async def _escalate_handover(
        self,
        result: AnswerCall,
        last_assistant_message: str,
        events: list[TranscriptEvent],
    ) -> None:
        """Convert a budget-exhausted answer into a handover escalation.

        The answer the broker would have sent is preserved verbatim as the
        recommendation — the developer sees what the broker was about to do,
        never a rewrite of it.

        Args:
            result: The answer the budget cap prevented from being submitted.
            last_assistant_message: The question that answer was replying to.
            events: Transcript events used to baseline out-of-band resolution.
        """
        disclosure = EscalationDisclosure(
            escalation_title="Autonomous answer budget exhausted",
            situation=(
                f"Autonomous answer budget exhausted: {self.budget_count} "
                "consecutive autonomous answers without developer contact. "
                "This broker is handing over."
            ),
            what_was_asked=last_assistant_message,
            what_is_at_stake=(
                "Continuing unsupervised would exceed the drift bound the "
                "budget exists to enforce."
            ),
            alternatives=[
                Alternative(
                    option="Send the broker's prepared answer (below)",
                    pros="The session continues immediately",
                    cons="It has not been reviewed by you",
                ),
                Alternative(
                    option="Answer differently in your own words",
                    pros="Full control after a long autonomous stretch",
                    cons="Requires reading the question",
                ),
            ],
            recommendation=result.answer,
            uncertainty=(
                "The budget cap, not doubt about the answer, forced this "
                f"escalation. Broker reasoning: {result.reasoning}"
            ),
            what_would_change_my_mind=(
                "Any developer response resets the budget and resumes "
                "autonomous operation."
            ),
        )
        await self._raise_escalation(
            self._new_escalation(disclosure),
            result.reasoning,
            events,
            task_summary="Handed over when the autonomous answer budget ran out",
        )

    async def _raise_question(
        self,
        escalation_id: str,
        tool_input: dict[str, Any],
        reason: str,
        analysis: EscalationDisclosure | None,
        *,
        task_summary: str,
    ) -> None:
        """Show the master an AskUserQuestion menu the developer answers in the pane.

        Args:
            escalation_id: The id claimed with the open picker.
            tool_input: Raw ``AskUserQuestion`` tool input.
            reason: Why the broker did not answer the menu itself.
            analysis: The ask LLM's disclosure when it chose to escalate.
            task_summary: The escalation's one-line Reason in the outcome history.
        """
        try:
            questions = ask.parse_questions(tool_input)
        except ask.AskInputError:
            questions = []  # the reason already names the failure
        self._log(
            DecisionLogKind.ESCALATION_RAISED,
            reason,
            f"AskUserQuestion menu open in pane {self.pane_id or '?'}",
            task_summary=task_summary,
            escalation_id=escalation_id,
        )
        payload = QuestionEscalationPayload(
            escalation_id=escalation_id,
            session_id=self.cfg.name,
            task_context=self._intent(),
            menu=ask.render_questions(questions),
            first_question=questions[0].question if questions else "",
            reason=reason,
            analysis=analysis,
        )
        nack = await self._to_master(payload, tolerated=frozenset(NackCode))
        if nack is None:
            return
        # The picker is still in the pane, so the claim stays; only the
        # escalation the master refused is dropped.
        self._log(
            DecisionLogKind.QUESTION_REFUSED,
            "question escalation refused",
            describe_refusal(payload.MESSAGE_TYPE, nack),
        )
        menu = self._open_menu
        if menu is not None and menu.escalation_id == escalation_id:
            menu.escalation_id = None

    async def _log_replaced_question(self, escalation_id: str) -> None:
        """Close the question escalation of a picker a newer one replaced."""
        self._log(
            DecisionLogKind.RETRACTED,
            "a newer AskUserQuestion menu replaced it",
            "",
            task_summary="Replaced by a newer question",
            escalation_id=escalation_id,
        )

    async def _retract_question(
        self, escalation_id: str, reason: str, summary: str
    ) -> None:
        """End a question escalation without a dispatch and tell the master.

        Args:
            escalation_id: The question escalation being withdrawn.
            reason: Why it ended; shown to the developer.
            summary: The escalation's Solution line in the outcome history.
        """
        self._log(
            DecisionLogKind.RETRACTED,
            reason,
            "",
            task_summary=summary,
            escalation_id=escalation_id,
        )
        await self._to_master(
            PaneRetractPayload(escalation_id=escalation_id, reason=reason)
        )

    async def _check_menu_answered(self) -> None:
        """Release the open picker once the transcript records its answer."""
        # Every way the picker closes writes an answer for its tool use, so
        # the transcript is the complete clearing signal.
        menu = self._open_menu
        if menu is None:
            return
        answer = next(
            (
                e
                for e in self._read_transcript()
                if isinstance(e, AskUserAnswer) and e.id == menu.tool_use_id
            ),
            None,
        )
        if answer is None:
            return
        self._open_menu = None
        self._status.mark_dirty()
        if menu.escalation_id is None:
            return
        answered_by_developer = (
            menu.injected is None or answer.answers != menu.injected
        )
        if answered_by_developer:
            reason = "answered in pane"
        else:
            reason = "the broker's answer was recorded late"
            self._log(
                DecisionLogKind.ASK_VERIFIED,
                "recorded late",
                tool_use_id=menu.tool_use_id,
            )
        # A late answer matching the broker's cannot be told apart from the
        # developer picking the same options, so both read as the developer's.
        await self._retract_question(
            menu.escalation_id, reason, "User answered the questions in the pane"
        )
        if answered_by_developer:
            await self._note_developer_contact()

    def _verify_ask(self, raw: dict[str, Any]) -> None:
        """Compare the PostToolUse echo against the injected answers.

        Args:
            raw: Raw ``PostToolUse`` hook JSON for an AskUserQuestion.
        """
        tool_use_id = str(raw.get("tool_use_id", "") or "")
        injected = self._ask_expected.pop(tool_use_id, None)
        if injected is None:
            return  # not broker-answered, or already verified
        task = self._ask_verify_tasks.pop(tool_use_id, None)
        if task is not None:
            task.cancel()
        response = raw.get("tool_response")
        echoed: Any = (
            cast(dict[str, Any], response).get("answers")
            if isinstance(response, dict)
            else None
        )
        if echoed == injected.answers:
            self._log(
                DecisionLogKind.ASK_VERIFIED,
                "PostToolUse echo matches",
                tool_use_id=tool_use_id,
            )
            return
        self.jobs.put_nowait(
            lambda: self._ask_verify_failed(
                tool_use_id,
                injected,
                "the session recorded different answers than the broker "
                "injected — it is proceeding on those answers",
                answer_recorded=True,
            )
        )

    def _spawn_ask_verify(self, tool_use_id: str) -> None:
        """Arm the transcript backstop for one injected answer."""
        task = asyncio.create_task(self._ask_verify_backstop(tool_use_id))
        self._ask_verify_tasks[tool_use_id] = task
        task.add_done_callback(
            lambda _: self._ask_verify_tasks.pop(tool_use_id, None)
        )

    async def _ask_verify_backstop(self, tool_use_id: str) -> None:
        """Check the transcript when no PostToolUse confirmed the answer.

        Transcript writes are asynchronous and may lag the hooks, so this is
        a bounded second look, not the primary signal.

        Args:
            tool_use_id: The injected answer being verified.
        """
        await asyncio.sleep(ASK_VERIFY_TIMEOUT_S)
        injected = self._ask_expected.pop(tool_use_id, None)
        if injected is None:
            return  # verified by PostToolUse in the meantime
        try:
            answer = next(
                (
                    e
                    for e in self._read_transcript()
                    if isinstance(e, AskUserAnswer) and e.id == tool_use_id
                ),
                None,
            )
        except Exception as exc:
            # A raise here would vanish into the task and drop verification
            # silently. Whether an answer was recorded is unknown, so take
            # the non-retracting arm. `exc` is cleared when this block
            # exits, before the queued job runs — bind the reason now.
            reason = (
                f"verification itself failed ({type(exc).__name__}: {exc}) "
                "— the broker cannot confirm its answers were delivered"
            )
            self.jobs.put_nowait(
                lambda: self._ask_verify_failed(
                    tool_use_id, injected, reason, answer_recorded=True
                )
            )
            return
        if answer is not None and answer.answers == injected.answers:
            self._log(
                DecisionLogKind.ASK_VERIFIED,
                "transcript backstop",
                tool_use_id=tool_use_id,
            )
            return
        if answer is None:
            reason = (
                "no answer was recorded — the injected answers may never "
                "have been delivered and the menu may still be on screen"
            )
        else:
            reason = (
                "the recorded answers differ from what the broker injected"
            )
        answer_recorded = answer is not None
        self.jobs.put_nowait(
            lambda: self._ask_verify_failed(
                tool_use_id, injected, reason, answer_recorded=answer_recorded
            )
        )

    async def _ask_verify_failed(
        self,
        tool_use_id: str,
        injected: _InjectedAnswers,
        reason: str,
        *,
        answer_recorded: bool,
    ) -> None:
        """Escalate a failed answer verification; the state is the message.

        Args:
            tool_use_id: The injected answer that failed verification.
            injected: The tool input and the answers the broker injected.
            reason: What the verification found; shown to the developer.
            answer_recorded: Whether the session already recorded an answer
                for this id.
        """
        self._log(DecisionLogKind.ASK_VERIFY_FAILED, reason, tool_use_id=tool_use_id)
        task_summary = "Escalated an AskUserQuestion answer that failed verification"
        # With no answer recorded the menu may still be in the pane, and its
        # answer clears it. Otherwise the session already proceeded on
        # something, and only a decision in the master chat resolves it.
        if not answer_recorded:
            escalation_id = uuid.uuid4().hex
            self._claim_menu(_OpenMenu(tool_use_id, escalation_id, injected.answers))
            await self._raise_question(
                escalation_id,
                injected.tool_input,
                reason,
                None,
                task_summary=task_summary,
            )
            return
        disclosure = EscalationDisclosure(
            escalation_title="AskUserQuestion answer verification failed",
            situation=(
                "AskUserQuestion answer verification failed — " + reason
                + f" Check pane {self.pane_id or '?'} and the session's "
                "recent turns."
            ),
            what_was_asked=(
                "Confirm what the session actually proceeded on, and correct "
                "it in the pane if needed."
            ),
            what_is_at_stake=(
                "The session may be running on answers the broker did not "
                "choose."
            ),
            alternatives=[
                Alternative(
                    option="inspect the pane",
                    pros="the pane and transcript are authoritative",
                    cons="",
                )
            ],
            recommendation="inspect the pane",
            uncertainty=reason,
            what_would_change_my_mind="Only your explicit decision resolves this.",
        )
        await self._raise_escalation(
            self._new_escalation(disclosure),
            reason,
            None,
            task_summary=task_summary,
        )

    async def _raise_escalation(
        self,
        payload: EscalationPayload,
        reasoning: str,
        events: list[TranscriptEvent] | None,
        *,
        task_summary: str,
    ) -> None:
        """Send a decision escalation to the master and go quiescent until it resolves.

        The broker stops driving the session entirely — it neither answers nor
        acts again until a dispatched decision arrives or the escalation is
        retracted.

        Args:
            payload: The escalation as it will reach the developer.
            reasoning: Why it was raised; recorded in the decision log.
            events: Transcript events used to baseline the user-prompt count for
                out-of-band resolution. ``None`` means only a dispatched
                decision resolves it.
            task_summary: The escalation's one-line Reason in the outcome history.
        """
        self._log(
            DecisionLogKind.ESCALATION_RAISED,
            reasoning,
            payload.disclosure.situation,
            task_summary=task_summary,
            escalation_id=payload.escalation_id,
        )
        self._end_decision_escalation()
        self._decision_escalation = DecisionEscalation(
            payload,
            _count_user_prompts(events) if events is not None else None,
            set(),
        )
        self._set_state(SessionState.ESCALATED)  # QUIESCENT until dispatch or retract
        await self._to_master(payload)

    async def _on_turn_end(self, last_assistant_message: str) -> None:
        """Triage one turn boundary, clearing a resolved escalation first.

        The Stop that ends the turn the developer resolved in the pane is the
        same Stop that carries the question they left unanswered. Retracting
        without triaging it drops the boundary: the pane sits idle with nobody
        driving it until the watchdog deadline expires.

        Args:
            last_assistant_message: ``last_assistant_message`` from the Stop
                payload.
        """
        if self._open_menu is not None:
            await self._check_menu_answered()
        if self.state == SessionState.ESCALATED:
            await self._check_out_of_band_resolution()
            if self.state == SessionState.ESCALATED:
                return  # still escalated: stay quiescent
        await self._triage_turn(last_assistant_message)

    async def _check_out_of_band_resolution(self) -> None:
        """Retract the active decision escalation if the developer already answered.

        A user prompt beyond the recorded baseline counts as resolution.
        Resolving returns the session to driving.
        """
        escalation = self._decision_escalation
        if (
            self.state != SessionState.ESCALATED
            or escalation is None
            or escalation.user_prompt_baseline is None
        ):
            return
        baseline = escalation.user_prompt_baseline
        if _count_user_prompts(self._read_transcript()) <= baseline:
            return
        await self._retract_decision("resolved in pane", "User answered in the pane")
        await self._note_developer_contact()

    async def _retract_decision(self, reason: str, summary: str) -> None:
        """End the active decision escalation without a dispatch and tell the master.

        Args:
            reason: Why it ended; shown to the developer.
            summary: The escalation's Solution line in the outcome history.
        """
        assert self._decision_escalation is not None
        escalation_id = self._decision_escalation.payload.escalation_id
        self._log(
            DecisionLogKind.RETRACTED,
            reason,
            "",
            task_summary=summary,
            escalation_id=escalation_id,
        )
        self._end_decision_escalation()
        self._set_state(SessionState.DRIVING)
        await self._to_master(
            EscalationRetractPayload(escalation_id=escalation_id, reason=reason)
        )

    async def _deliver_decision(self, decision: DispatchDecisionPayload) -> None:
        """Submit the developer's decision, then confirm the outcome upstream.

        The decision text is typed into the pane exactly as written — developer
        text is wrapped, never rewritten. Contact with the developer resets the
        autonomous answer budget. The master resolves the escalation only on the
        delivery confirmation this sends: a decision whose id no longer matches
        the active escalation, or one whose pane write fails, reports back as
        undelivered instead of resolving.

        Args:
            decision: The dispatched decision, carrying the escalation id it
                answers and the response text to submit.
        """
        active = self._decision_escalation
        if active is None or decision.escalation_id != active.payload.escalation_id:
            # The broker has moved past this escalation, so the master should
            # drop its queue entry (still_live=False). Reachable when a master
            # restart re-surfaces an escalation this broker already answered.
            self._log(
                DecisionLogKind.DISPATCH_STALE,
                "stale dispatch_decision ignored",
                escalation_id=decision.escalation_id,
            )
            await self._to_master(
                DecisionUndeliveredPayload(
                    escalation_id=decision.escalation_id,
                    detail="session had already moved past this escalation",
                    still_live=False,
                )
            )
            return
        try:
            await self._submit(decision.response)
        except Exception as exc:
            # The master resolves only on confirmed delivery, so a failed pane
            # write — an occupied pane included — leaves the escalation live
            # there. Report the miss loudly (still_live=True); it stays
            # surfaced for a re-decide.
            detail = f"{type(exc).__name__}: {exc}"
            self._log(
                DecisionLogKind.DISPATCH_FAILED,
                "pane submission failed",
                detail,
                escalation_id=decision.escalation_id,
            )
            await self._to_master(
                DecisionUndeliveredPayload(
                    escalation_id=decision.escalation_id,
                    detail=detail,
                    still_live=True,
                )
            )
            return
        self._end_decision_escalation()
        await self._note_developer_contact()
        self._set_state(SessionState.DRIVING)
        self._log(
            DecisionLogKind.DISPATCHED,
            "developer decision delivered",
            decision.response,
            escalation_id=decision.escalation_id,
        )
        # Resolution waits for this: the escalation clears on the master only
        # now that the decision has actually reached the pane.
        await self._to_master(
            DecisionDeliveredPayload(escalation_id=decision.escalation_id)
        )

    async def _reactivate(self, payload: ReactivatePayload) -> None:
        """Drive a new task through the session that just completed one.

        The pane, chat and transcript carry over untouched — only the task
        changes. Direct developer contact resets the autonomous answer budget.

        Args:
            payload: The new task intent, grounded before anything is typed.
        """
        self._log(
            DecisionLogKind.REACTIVATED, "new task in the same session", payload.intent
        )
        self._end_decision_escalation()
        await self._note_developer_contact()
        await self._ground_and_submit(payload.intent)

    async def _send_developer_prompt(self, prompt: SendPromptPayload) -> None:
        """Relay a developer-authored prompt into the pane verbatim.

        Args:
            prompt: The text the master relayed, submitted unmodified.
        """
        try:
            await self._submit(prompt.text)
        except PaneOccupiedError as exc:
            self._log(
                DecisionLogKind.DISPATCH_FAILED, "developer prompt not typed", str(exc)
            )
            await self._to_master(PromptUndeliveredPayload(detail=str(exc)))
            return
        if self._decision_escalation is not None:
            await self._retract_decision(
                "superseded by a developer prompt", "Superseded by a developer prompt"
            )
        self._set_state(SessionState.DRIVING)
        await self._note_developer_contact()
        self._log(DecisionLogKind.DEVELOPER_PROMPT, "relayed by master", prompt.text)

    async def _note_developer_contact(self) -> None:
        """Reset the autonomous answer budget and report it to the master."""
        self.budget_count = 0
        await self._report_budget(0)

    async def _report_budget(self, count: int) -> None:
        """Tell the master the autonomous answer budget stands at ``count``."""
        await self._to_master(BudgetUpdatePayload(count=count))

    def _end_decision_escalation(self) -> None:
        """Clear the live decision escalation and cancel any clarification bound to it.

        Every path that ends an escalation (dispatch, out-of-band retract,
        developer prompt, reactivation, a newer raise, a fatal error) routes
        here, so an in-flight clarify LLM call can never outlive the
        escalation it is about.
        """
        escalation = self._decision_escalation
        self._decision_escalation = None
        if escalation is not None:
            for task in escalation.clarify_tasks:
                task.cancel()

    async def _reconcile(self) -> None:
        """Enqueue reconciliation work on watchdog expiry."""
        self.jobs.put_nowait(self._reconcile_job)

    async def _reconcile_job(self) -> None:
        """Recover a turn boundary that produced no hook event.

        While escalated it defers to the out-of-band resolution check;
        otherwise it only acts when driving. A new assistant message is
        classified as if a Stop hook had delivered it, so a dropped hook
        cannot silently strand the session.
        """
        if self._open_menu is not None:
            await self._check_menu_answered()
        if self.state == SessionState.ESCALATED:
            await self._check_out_of_band_resolution()
        # A native prompt open means the turn is blocked on it, not over.
        if self._native_prompt() is not None or self.state not in ACTIVE_STATES:
            return
        events = self._read_transcript()
        if len(events) == self._last_event_count:
            return  # nothing new — sleep again
        last_text = next(
            (e.text for e in reversed(events) if isinstance(e, AssistantText)),
            None,
        )
        if last_text is None:
            return
        self._log(
            DecisionLogKind.WATCHDOG_RECONCILIATION,
            "no hook event before deadline; herdr reports idle/blocked",
            "",
        )
        await self._triage_turn(last_text)

    # ── plumbing ──────────────────────────────────────────────────────────

    def _intent(self) -> str:
        """Return the authoritative intent of the task being driven."""
        return self._task.authoritative

    def _set_task(self, raw: str, approved: str | None) -> None:
        """Replace the task being driven, and judge permissions against it."""
        self._task = _TaskIntent(raw, approved)
        self.permission.set_intent(self._task.authoritative)

    def _read_transcript(self) -> list[TranscriptEvent]:
        """Read the session transcript as cleaned events.

        Returns:
            Every event in the transcript, oldest first.

        Raises:
            FatalSessionError: No transcript path is bound yet. Reading a
                partial or guessed transcript is never an option — fail loud.
        """
        if self.transcript_path is None:
            raise FatalSessionError(
                "transcript_unbound", "no transcript path bound"
            )
        events, report = read_cleaned(Path(self.transcript_path))
        self._log_transcript_drift(report)
        return events

    def _log_transcript_drift(self, report: ReadReport) -> None:
        """Log each new transcript drift signal once, to the diagnostic log only.

        Args:
            report: What the adapter lost or flagged on one transcript read.
        """
        for warning in report.warnings:
            if warning not in self._logged_drift_warnings:
                self._logged_drift_warnings.add(warning)
                logger.warning("transcript drift: %s", warning)
        if not report.skipped_records and not report.unknown_types:
            return
        versions = frozenset(report.versions)
        if versions in self._logged_drift_versions:
            return
        self._logged_drift_versions.add(versions)
        logger.warning(
            "transcript versions %s: %d records failed validation, "
            "unknown record types %s",
            sorted(versions),
            report.skipped_records,
            dict(report.unknown_types),
        )

    def _native_prompt(self) -> str | None:
        """Name the native prompt open in the pane, or ``None`` when there is none."""
        if self._open_menu is not None:
            return "an AskUserQuestion menu"
        if self._permission_prompt_pending:
            return "a permission prompt"
        return None

    def _reports_permission_prompt(self) -> bool:
        """Whether to report a permission prompt upstream."""
        # An open picker's own permission check can raise a permission-prompt
        # notification; the picker is the prompt the developer sees.
        return self._permission_prompt_pending and self._open_menu is None

    async def _submit(self, text: str) -> None:
        """Type ``text`` into the session and submit it, with an explicit timeout.

        Args:
            text: Prompt text, submitted exactly as given.

        Raises:
            PaneOccupiedError: A native prompt is open in the pane; typing
                would land in it.
        """
        what = self._native_prompt()
        if what is not None:
            raise PaneOccupiedError(what, self.pane_id)
        await asyncio.to_thread(
            driver.agent_prompt,
            self.cfg.name,
            text,
            timeout_s=SUBMIT_TIMEOUT_S,
        )

    async def _to_master(
        self,
        payload: WireMessage,
        *,
        tolerated: frozenset[NackCode] = frozenset(),
    ) -> NackPayload | None:
        """Send one message to the master and wait for its reply.

        Args:
            payload: The message; its ``MESSAGE_TYPE`` is the envelope type.
            tolerated: Refusal codes the caller handles itself.

        Returns:
            ``None`` once accepted, otherwise the tolerated refusal.

        Raises:
            MasterRefusedError: The master refused it with a code outside
                ``tolerated``, or with no code.
        """
        return await send_to_master(
            Path(self.cfg.master_socket_path),
            self.cfg.name,
            payload,
            tolerated=tolerated,
        )

    async def _fatal(self, error_class: str, detail: str) -> None:
        """Log the failure, enter the error state, and tell the master.

        If the master is unreachable, the exception is logged and not
        re-raised: the pane simply degrades to stock Claude Code.

        Args:
            error_class: Machine-readable failure class.
            detail: Human-readable detail for the developer.
        """
        logger.error("fatal: %s: %s", error_class, detail)
        self._log(DecisionLogKind.ERROR, error_class, detail)
        self._end_decision_escalation()
        self._set_state(SessionState.ERROR)
        try:
            await self._to_master(
                FatalErrorPayload(error_class=error_class, detail=detail)
            )
        except Exception:
            # Master unreachable: the pane degrades to stock Claude Code.
            logger.exception("could not report fatal error to master")

    async def _on_session_end(self) -> None:
        """Report the terminal end to the master, then stop serving and exit."""
        try:
            await self._to_master(SessionEndedPayload())
        except Exception:
            logger.exception("could not report session end to master")
        self._request_stop()

    def _log(
        self,
        kind: DecisionLogKind,
        reasoning: str,
        detail: str = "",
        *,
        task_summary: str | None = None,
        escalation_id: str | None = None,
        tool_use_id: str | None = None,
        headline: str | None = None,
        supporting: str | None = None,
    ) -> None:
        """Append one entry to this session's decision log."""
        decision_log.append(
            self.decision_log_path,
            DecisionLogRow(
                kind=kind,
                reasoning=reasoning,
                detail=detail,
                task_summary=task_summary,
                escalation_id=escalation_id,
                tool_use_id=tool_use_id,
                headline=headline,
                supporting=supporting,
            ),
        )

    def _set_state(self, state: SessionState) -> None:
        """Assign a new state and log the transition."""
        logger.info("session %s: %s -> %s", self.cfg.name, self.state, state)
        self.state = state
        self._status.mark_dirty()

    def _set_perm_pending(self, pending: bool) -> None:
        """Record whether the session sits on a native permission prompt."""
        self._permission_prompt_pending = pending
        self._status.mark_dirty()

    def _herdr_state(self) -> AgentStatus:
        """Report the agent's state as Herdr sees it, for the watchdog gate.

        Returns:
            Herdr's status, or ``"unknown"`` if the query fails.
        """
        try:
            return driver.agent_get(self.cfg.name, timeout_s=10.0).agent_status
        except Exception:
            return "unknown"  # gates the watchdog read out; never classifies


def _raw_tool(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Read the tool name and input out of a raw hook payload.

    Args:
        raw: Raw hook JSON.

    Returns:
        The tool name and its input, each defaulting to empty when absent or
        of an unexpected shape.
    """
    name = raw.get("tool_name")
    tool_input = raw.get("tool_input")
    return (
        name if isinstance(name, str) else "",
        cast(dict[str, Any], tool_input) if isinstance(tool_input, dict) else {},
    )


def _count_user_prompts(events: list[TranscriptEvent]) -> int:
    """Count the user prompts among ``events``."""
    return sum(1 for e in events if isinstance(e, UserPrompt))
