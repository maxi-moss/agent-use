"""SessionBroker: headless driver of one Claude Code session.

Structure:
- `handle()` is the socket handler and does NO slow work: it replies, then
  enqueues. The permission_request reply is the hot path.
- The serial event queue is the only place LLM work and pane writes happen.
- Classification input is `last_assistant_message` from the Stop payload,
  never the transcript tail. The watchdog reconciliation is the
  one sanctioned pure-transcript read.
- Every pane write is the two-step `agent prompt` + `pane send-keys enter`
  via driver.submit_prompt with an explicit timeout.
"""

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)
from pydantic import ValidationError

from broker import llm as llm_module
from broker.llm import LLMCaller, ToolCall
from broker.config import AdoptedSession, BrokerConfig, SessionBrokerConfig
from broker.paths import BrokerPaths
from broker.herdr import driver
from broker.claude.paths import transcript_dir_for_cwd
from broker.protocol import client
from broker.protocol.constants import (
    ACTIVE_STATES,
    DECISION_ESCALATED,
    SessionState,
    T_APPROVE_PROMPT,
    T_BUDGET_UPDATE,
    T_COMPLETION,
    T_DISPATCH_DECISION,
    T_ESCALATION,
    T_FATAL_ERROR,
    T_GET_DECISION_LOG,
    T_HOOK_EVENT,
    T_PERMISSION_REQUEST,
    T_PROMPT_PROPOSAL,
    T_REACTIVATE,
    T_RETRACT,
    T_SEND_PROMPT,
    T_SHUTDOWN,
    T_STATUS,
)
from broker.protocol.server import serve_unix
from broker.protocol.schemas import (
    Alternative,
    ApprovePromptPayload,
    DecisionLogPayload,
    DispatchDecisionPayload,
    Envelope,
    EscalationPayload,
    HookEventPayload,
    PermissionDecisionPayload,
    PermissionRequestPayload,
    ReactivatePayload,
    Response,
    SendPromptPayload,
    StatusPayload,
)
from broker.session import decision_log
from broker.session.triage import (
    AnswerCall,
    CompleteCall,
    EscalateCall,
    NoActionCall,
    ground_intent,
    triage,
)
from broker.session.watchdog import Watchdog
from broker.transcript.adapter import read_cleaned
from broker.transcript.schemas import (
    AskUserAnswer,
    AssistantText,
    Question,
    TranscriptEvent,
    UserPrompt,
)

logger = logging.getLogger(__name__)

SUBMIT_TIMEOUT_S = 15.0
MASTER_TIMEOUT_S = 10.0
SESSION_BIND_TIMEOUT_S = 60.0

# Forwarded to the claude binary at spawn. "auto" classifies each tool call and
# still prompts on the risky ones, so the hook's escalation path survives;
# "bypassPermissions" would silently approve every escalation.
CLAUDE_AGENT_ARGS = ["--model", "opus", "--permission-mode", "auto"]

Job = Callable[[], Awaitable[None]]


class FatalSessionError(Exception):
    def __init__(self, error_class: str, detail: str) -> None:
        """Record the machine-readable class and human detail of the failure."""
        super().__init__(f"{error_class}: {detail}")
        self.error_class = error_class
        self.detail = detail


def _bind_llm(client: AsyncAnthropic) -> LLMCaller[ToolCall]:
    """Adapt an Anthropic client into the keyword-only ``LLMCaller`` shape.

    Args:
        client: Anthropic client passed through to ``llm.call_tool``.

    Returns:
        A callable that forwards every triage call to that one client.
    """

    async def call(
        *,
        model: str,
        max_tokens: int,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        tools: list[ToolParam],
        tool_choice: ToolChoiceParam,
    ) -> ToolCall:
        """Invoke the bound client and return its single tool call."""
        return await llm_module.call_tool(
            client,
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

    return call


class SessionBroker:
    def __init__(
        self,
        cfg: SessionBrokerConfig,
        *,
        llm_call: LLMCaller[ToolCall] | None = None,
    ) -> None:
        """Build the broker's state without touching the socket or the pane.

        Args:
            cfg: Session identity, socket paths, cwd, intent and budget limits.
            llm_call: Injected tool-calling backend. When omitted, ``run()``
                builds one from an Anthropic client — tests pass a fake.
        """
        self.cfg = cfg
        self.broker_cfg = BrokerConfig(
            model_id=cfg.model_id,
            max_tokens=cfg.max_tokens,
            watchdog_seconds=cfg.watchdog_seconds,
            budget_max=cfg.budget_max,
            broker_home=cfg.broker_home,
        )
        self._llm_call = llm_call
        self.state: SessionState = SessionState.SPAWNING
        self.pane_id: str | None = None
        self.claude_session_id: str | None = None
        self.transcript_path: Path | None = None
        self.budget_count = cfg.budget_count
        self.intent = cfg.intent  # replaced outright on reactivation
        self.approved_prompt: str | None = None

        self.session_bound = asyncio.Event()
        self._approval: asyncio.Future[ApprovePromptPayload] | None = None
        self._proposal_id: str | None = None
        self.queue: asyncio.Queue[Job | None] = asyncio.Queue()
        self._shutdown = asyncio.Event()

        self._active_escalation: EscalationPayload | None = None
        self._pending_ask_id: str | None = None
        self._seen_ask_ids: set[str] = set()
        self._user_prompt_baseline = 0
        self._last_event_count = -1

        self.decision_log_path = BrokerPaths(cfg.broker_home).session_decisions(
            cfg.name
        )
        self.watchdog = Watchdog(
            cfg.watchdog_seconds, self._herdr_state, self._reconcile
        )

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Serve the socket, launch the session, then drain the event queue."""
        # Bind FIRST — hooks may fire before the pane exists.
        server = await serve_unix(Path(self.cfg.socket_path), self.handle)
        if self._llm_call is None:
            self._llm_call = _bind_llm(llm_module.build_client(self.broker_cfg))
        self.watchdog.start()
        try:
            await self._launch()
            await self._event_loop()
        except FatalSessionError as exc:
            await self._fatal(exc.error_class, exc.detail)
        finally:
            await self.watchdog.stop()
            server.close()
            await server.wait_closed()

    async def _launch(self) -> None:
        """Take over or start a Claude session, then ground and submit the task.

        Raises:
            FatalSessionError: A fresh start saw no SessionStart hook event
                within ``SESSION_BIND_TIMEOUT_S``.
        """
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
        self.transcript_path = Path(adopt.transcript_path)
        self.session_bound.set()
        self._log(
            "adopted",
            "reassigned to a session that was already running",
            f"pane={adopt.pane_id} claude_session={adopt.claude_session_id}",
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
            env={"BROKER_SOCKET": cfg.socket_path},
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
            agent_args=CLAUDE_AGENT_ARGS,
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
        """
        self.intent = intent
        self.approved_prompt = None  # superseded until the developer approves
        self._set_state(SessionState.GROUNDING)
        assert self._llm_call is not None
        proposal = await ground_intent(
            self._llm_call,
            self.broker_cfg,
            intent=intent,
            cwd=Path(self.cfg.cwd),
        )
        self._proposal_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        self._approval = loop.create_future()
        self._set_state(SessionState.AWAITING_APPROVAL)
        await self._to_master(
            T_PROMPT_PROPOSAL,
            {
                "proposal_id": self._proposal_id,
                "proposed_prompt": proposal.prompt,
                "grounding_summary": proposal.reasoning,
            },
        )
        # Approval is synchronous and blocking — no timeout.
        approved = await self._approval
        self.approved_prompt = approved.prompt
        await self._submit(approved.prompt)
        self._set_state(SessionState.DRIVING)

    async def _event_loop(self) -> None:
        """Run queued jobs one at a time until shutdown or the sentinel."""
        while not self._shutdown.is_set():
            job = await self.queue.get()
            if job is None:
                break
            try:
                await job()
            except FatalSessionError as exc:
                await self._fatal(exc.error_class, exc.detail)
            except Exception as exc:  # fail loud, keep serving the socket
                await self._fatal(type(exc).__name__, str(exc))

    # ── socket handler: reply, then enqueue — never slow work ─────────────

    async def handle(self, env: Envelope) -> Response | None:
        """Dispatch one socket envelope, turning payload rejects into replies.

        Args:
            env: Decoded envelope from the master or the hook client.

        Returns:
            The reply to write back, or ``None`` for fire-and-forget messages.
        """
        try:
            return await self._handle(env)
        except ValidationError as exc:
            logger.error("invalid %s payload: %s", env.type, exc)
            return Response(id=env.id, ok=False, payload={"error": str(exc)})

    def _on_permission_request(self, env: Envelope) -> Response:
        """Answer a permission request on the hot path, then enqueue the rest.

        The reply is built with zero LLM work: this is the synchronous hook
        path, and every millisecond here is a millisecond the supervised tool
        call is blocked.

        Args:
            env: Envelope carrying a ``PermissionRequestPayload``.

        Returns:
            The decision reply for the hook.
        """
        payload = PermissionRequestPayload.model_validate(env.payload)
        self.queue.put_nowait(lambda: self._permission_passthrough(payload))
        return Response(
            id=env.id,
            ok=True,
            payload=PermissionDecisionPayload(
                decision=DECISION_ESCALATED
            ).model_dump(),
        )

    async def _handle(self, env: Envelope) -> Response | None:
        """Reply to one message type, enqueuing anything slow onto the queue.

        Args:
            env: Decoded envelope; its ``type`` selects the branch and its
                ``payload`` is validated per branch.

        Returns:
            The reply to write back, or ``None`` for ``hook_event``. Unknown
            types get an ``ok=False`` reply.
        """
        if env.type == T_PERMISSION_REQUEST:
            return self._on_permission_request(env)

        if env.type == T_HOOK_EVENT:
            self.watchdog.reset()
            hook = HookEventPayload.model_validate(env.payload)
            self._dispatch_hook(hook)
            return None  # fire-and-forget

        if env.type == T_APPROVE_PROMPT:
            approved = ApprovePromptPayload.model_validate(env.payload)
            if (
                self._approval is None
                or self._approval.done()
                or approved.proposal_id != self._proposal_id
            ):
                logger.warning("stale approve_prompt ignored")
                return Response(
                    id=env.id, ok=False, payload={"error": "stale proposal"}
                )
            self._approval.set_result(approved)
            return Response(id=env.id, ok=True)

        if env.type == T_DISPATCH_DECISION:
            decision = DispatchDecisionPayload.model_validate(env.payload)
            self.queue.put_nowait(lambda: self._deliver_decision(decision))
            return Response(id=env.id, ok=True)

        if env.type == T_REACTIVATE:
            reactivate = ReactivatePayload.model_validate(env.payload)
            if self.state != SessionState.COMPLETED:
                return Response(
                    id=env.id,
                    ok=False,
                    payload={
                        "error": (
                            f"session is {self.state!r}, not 'completed' — "
                            "reassign a new broker instead of displacing the "
                            "task this one is still driving"
                        )
                    },
                )
            # Closes the gate here, not in the job: a second reactivate
            # arriving before the queue drains must not pass it too.
            self._set_state(SessionState.GROUNDING)
            self.queue.put_nowait(lambda: self._reactivate(reactivate))
            return Response(id=env.id, ok=True)

        if env.type == T_SEND_PROMPT:
            prompt = SendPromptPayload.model_validate(env.payload)
            self.queue.put_nowait(lambda: self._send_developer_prompt(prompt))
            return Response(id=env.id, ok=True)

        if env.type == T_STATUS:
            return Response(
                id=env.id,
                ok=True,
                payload=StatusPayload(
                    state=self.state,
                    pane_id=self.pane_id,
                    claude_session_id=self.claude_session_id,
                    transcript_path=(
                        str(self.transcript_path) if self.transcript_path else None
                    ),
                ).model_dump(),
            )

        if env.type == T_GET_DECISION_LOG:
            return Response(
                id=env.id,
                ok=True,
                payload=DecisionLogPayload(
                    text=decision_log.render_log(self.decision_log_path)
                ).model_dump(),
            )

        if env.type == T_SHUTDOWN:
            self._shutdown.set()
            self.queue.put_nowait(None)
            return Response(id=env.id, ok=True)

        return Response(
            id=env.id, ok=False, payload={"error": f"unknown type {env.type!r}"}
        )

    def _dispatch_hook(self, hook: HookEventPayload) -> None:
        """Route one Claude Code hook event to state changes or queued jobs.

        ``Stop`` carries the classification input as ``last_assistant_message``,
        never the transcript tail, and is diverted to an out-of-band resolution
        check while escalated. ``StopFailure`` is surfaced as fatal, not treated
        as a completed turn.

        Args:
            hook: Validated hook payload; ``raw`` is the untyped hook JSON.
        """
        raw = hook.raw
        name = hook.hook_event_name
        if name == "SessionStart":
            self._bind_session(raw)
        elif name == "Stop":
            message = str(raw.get("last_assistant_message", "") or "")
            if self.state == SessionState.ESCALATED:
                self.queue.put_nowait(self._check_out_of_band_resolution)
            else:
                self.queue.put_nowait(lambda: self._classify(message))
        elif name == "StopFailure":
            error_class = str(
                raw.get("matcher") or raw.get("error") or "stop_failure"
            )
            detail = str(raw.get("message") or raw)
            # Surfaced, NOT a completed turn.
            self.queue.put_nowait(lambda: self._fatal(error_class, detail))
        elif name in {"UserPromptSubmit", "PostToolUse"}:
            if self.state == SessionState.ESCALATED:
                self.queue.put_nowait(self._check_out_of_band_resolution)
        elif name == "Notification":
            self._log("notification", "", str(raw.get("message", "")))
            if raw.get("notification_type") == "permission_prompt":
                if self.state == SessionState.DRIVING:
                    self._set_state(SessionState.BLOCKED_PERMISSION)
        elif name == "SessionEnd":
            self._set_state(SessionState.STOPPED)
            self._log("session_end", "", "SessionEnd hook received")
        elif name in {"PreCompact", "PostCompact"}:
            self._log("compaction", "", name)  # continue normally
        else:
            logger.debug("unhandled hook event %s", name)

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
            self.transcript_path = Path(transcript)
        elif self.transcript_path is None and self.claude_session_id:
            # cwd derivation is the fallback, not an error.
            self.transcript_path = (
                transcript_dir_for_cwd(Path(self.cfg.cwd))
                / f"{self.claude_session_id}.jsonl"
            )
        self.session_bound.set()

    # ── queued jobs ───────────────────────────────────────────────────────

    async def _permission_passthrough(self, p: PermissionRequestPayload) -> None:
        """Record a passed-through permission request and branch on the tool.

        ``AskUserQuestion`` becomes a mechanical escalation, deduped by
        ``tool_use_id`` since PreToolUse can fire multiple times per logical
        operation. Any other tool leaves the native prompt on screen and only
        marks the session blocked.

        Args:
            p: The permission request that was answered ``escalated``.
        """
        self._log(
            "permission_passthrough",
            "Every permission_request is answered 'escalated' with no LLM "
            "work; the native prompt appears",
            f"tool={p.tool_name} id={p.tool_use_id}",
        )
        if p.tool_name == "AskUserQuestion":
            # PreToolUse can fire 3x per logical op — dedupe.
            if p.tool_use_id in self._seen_ask_ids:
                return
            self._seen_ask_ids.add(p.tool_use_id)
            await self._ask_user_question(p.tool_input, p.tool_use_id)
        else:
            if self.state == SessionState.DRIVING:
                self._set_state(SessionState.BLOCKED_PERMISSION)

    async def _classify(self, last_assistant_message: str) -> None:
        """Triage one turn boundary into an answer, escalation or completion.

        Only runs while driving or blocked on a permission prompt. The
        transcript is read for context only — the classification input is the
        message itself. An exhausted budget converts an answer into a handover
        escalation rather than sending it silently.

        Args:
            last_assistant_message: ``last_assistant_message`` from the Stop
                payload, or the last assistant text when the watchdog
                reconciles.
        """
        if self.state not in ACTIVE_STATES:
            self._log(
                "no_action",
                f"turn boundary ignored in state {self.state!r}",
                "",
            )
            return
        self._set_state(SessionState.DRIVING)
        events = self._read_transcript()  # context ONLY; input is the message
        self._last_event_count = len(events)
        assert self._llm_call is not None
        result = await triage(
            self._llm_call,
            self.broker_cfg,
            intent=self._intent(),
            events=events,
            event_name="Stop",
            last_assistant_message=last_assistant_message,
        )
        if isinstance(result, AnswerCall):
            if self.budget_count >= self.cfg.budget_max:
                await self._escalate_handover(result, last_assistant_message, events)
            else:
                self._log("answered", result.reasoning, result.answer)
                await self._submit(result.answer)
                self.budget_count += 1
                await self._to_master(
                    T_BUDGET_UPDATE, {"count": self.budget_count}
                )
        elif isinstance(result, EscalateCall):
            payload = self._new_escalation(
                situation=result.situation,
                what_was_asked=result.what_was_asked,
                what_is_at_stake=result.what_is_at_stake,
                alternatives=result.alternatives,
                recommendation=result.recommendation,
                uncertainty=result.uncertainty,
                what_would_change_my_mind=result.what_would_change_my_mind,
            )
            await self._raise_escalation(payload, result.reasoning, events)
        elif isinstance(result, CompleteCall):
            self._log("completed", result.reasoning, result.summary)
            await self._to_master(T_COMPLETION, {"summary": result.summary})
            self._set_state(SessionState.COMPLETED)  # stop driving; keep serving
        elif isinstance(result, NoActionCall):  # pyright: ignore[reportUnnecessaryIsInstance]
            self._log("no_action", result.reasoning, "")

    def _new_escalation(
        self,
        *,
        situation: str,
        what_was_asked: str,
        what_is_at_stake: str,
        alternatives: list[Alternative],
        recommendation: str,
        uncertainty: str,
        what_would_change_my_mind: str,
    ) -> EscalationPayload:
        """Build an escalation carrying this session's identifying preamble.

        Args:
            situation: What is happening that needs a decision.
            what_was_asked: The question, verbatim.
            what_is_at_stake: Consequences of getting it wrong.
            alternatives: The options open to the developer.
            recommendation: The broker's suggested option.
            uncertainty: What the broker is unsure about.
            what_would_change_my_mind: What would flip the recommendation.

        Returns:
            The escalation, ready to raise.
        """
        return EscalationPayload(
            escalation_id=uuid.uuid4().hex,
            session_id=self.cfg.name,
            task_context=self._intent(),
            situation=situation,
            what_was_asked=what_was_asked,
            what_is_at_stake=what_is_at_stake,
            alternatives=alternatives,
            recommendation=recommendation,
            uncertainty=uncertainty,
            what_would_change_my_mind=what_would_change_my_mind,
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
        payload = self._new_escalation(
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
        await self._raise_escalation(payload, result.reasoning, events)

    async def _ask_user_question(
        self, tool_input: dict[str, Any], tool_use_id: str
    ) -> None:
        """Escalate a pending AskUserQuestion mechanically, without the LLM.

        The broker cannot drive the native menu, so the escalation points the
        developer at the pane and is retracted once they answer there.

        Args:
            tool_input: Raw ``AskUserQuestion`` tool input.
            tool_use_id: Recorded as the pending ask so
                ``_check_out_of_band_resolution`` can match the answer.
        """
        rendered, alternatives, recommendation = _render_ask_user(tool_input)
        payload = self._new_escalation(
            situation=(
                "AskUserQuestion pending — manual input required in pane "
                f"{self.pane_id or '?'}. The native menu is on screen; "
                "answer it directly in that pane."
            ),
            what_was_asked=rendered,
            what_is_at_stake=(
                "The session is blocked on this menu until it is answered."
            ),
            alternatives=alternatives,
            recommendation=recommendation,
            uncertainty="This broker cannot drive menus.",
            what_would_change_my_mind=(
                "Answering in the pane retracts this escalation "
                "automatically."
            ),
        )
        self._pending_ask_id = tool_use_id
        await self._raise_escalation(payload, "AskUserQuestion (mechanical)", None)

    async def _raise_escalation(
        self,
        payload: EscalationPayload,
        reasoning: str,
        events: list[TranscriptEvent] | None,
    ) -> None:
        """Send an escalation to the master and go quiescent until it resolves.

        The broker stops driving the session entirely — it neither answers nor
        acts again until a dispatched decision arrives or the escalation is
        retracted.

        Args:
            payload: The escalation as it will reach the developer.
            reasoning: Why it was raised; recorded in the decision log.
            events: Transcript events used to baseline the user-prompt count for
                out-of-band resolution. ``None`` disables that check (the
                pending-ask id is used instead).
        """
        self._log("escalation_raised", reasoning, payload.situation)
        self._active_escalation = payload
        self._user_prompt_baseline = (
            _count_user_prompts(events) if events is not None else -1
        )
        self._set_state(SessionState.ESCALATED)  # QUIESCENT until dispatch or retract
        await self._to_master(T_ESCALATION, payload.model_dump())

    async def _check_out_of_band_resolution(self) -> None:
        """Retract the active escalation if the developer already answered.

        A pending ``AskUserQuestion`` is resolved by a matching answer event;
        otherwise a user prompt beyond the recorded baseline counts as
        resolution. Resolving returns the session to driving.
        """
        if self.state != SessionState.ESCALATED or self._active_escalation is None:
            return
        events = self._read_transcript()
        resolved = False
        if self._pending_ask_id is not None:
            resolved = any(
                isinstance(e, AskUserAnswer) and e.id == self._pending_ask_id
                for e in events
            )
        elif self._user_prompt_baseline >= 0:
            resolved = _count_user_prompts(events) > self._user_prompt_baseline
        if not resolved:
            return
        escalation_id = self._active_escalation.escalation_id
        self._log("retracted", "resolved in pane", escalation_id)
        self._active_escalation = None
        self._pending_ask_id = None
        self._set_state(SessionState.DRIVING)
        await self._to_master(
            T_RETRACT,
            {"escalation_id": escalation_id, "reason": "resolved in pane"},
        )

    async def _deliver_decision(self, decision: DispatchDecisionPayload) -> None:
        """Submit the developer's decision and resume driving the session.

        The decision text is typed into the pane exactly as written — developer
        text is wrapped, never rewritten. Contact with the developer resets the
        autonomous answer budget. A decision whose escalation id does not match
        the active escalation is stale and dropped rather than acted on.

        Args:
            decision: The dispatched decision, carrying the escalation id it
                answers and the response text to submit.
        """
        active = self._active_escalation
        if active is None or decision.escalation_id != active.escalation_id:
            # Stale; the master's dispatch-time liveness should have caught it.
            self._log(
                "error",
                "stale dispatch_decision ignored",
                decision.escalation_id,
            )
            return
        await self._submit(decision.response)
        self._active_escalation = None
        self._pending_ask_id = None
        self.budget_count = 0
        await self._to_master(T_BUDGET_UPDATE, {"count": 0})
        self._set_state(SessionState.DRIVING)
        self._log("dispatched", "developer decision delivered", decision.response)

    async def _reactivate(self, payload: ReactivatePayload) -> None:
        """Drive a new task through the session that just completed one.

        The pane, chat and transcript carry over untouched — only the task
        changes. Direct developer contact resets the autonomous answer budget.

        Args:
            payload: The new task intent, grounded before anything is typed.
        """
        self._log("reactivated", "new task in the same session", payload.intent)
        self._active_escalation = None
        self._pending_ask_id = None
        self.budget_count = 0
        await self._to_master(T_BUDGET_UPDATE, {"count": 0})
        await self._ground_and_submit(payload.intent)

    async def _send_developer_prompt(self, prompt: SendPromptPayload) -> None:
        """Relay a developer-authored prompt into the pane verbatim.

        Direct developer contact clears any active escalation and resets the
        autonomous answer budget.

        Args:
            prompt: The text the master relayed, submitted unmodified.
        """
        await self._submit(prompt.text)
        self.budget_count = 0  # developer-relayed — resets
        await self._to_master(T_BUDGET_UPDATE, {"count": 0})
        self._active_escalation = None
        self._pending_ask_id = None
        self._set_state(SessionState.DRIVING)
        self._log("developer_prompt", "relayed by master", prompt.text)

    async def _reconcile(self) -> None:
        """Enqueue reconciliation work on watchdog expiry."""
        self.queue.put_nowait(self._reconcile_job)

    async def _reconcile_job(self) -> None:
        """Recover a turn boundary that produced no hook event.

        While escalated it defers to the out-of-band resolution check;
        otherwise it only acts when driving or blocked. A new assistant
        message is classified as if a Stop hook had delivered it, so a
        dropped hook cannot silently strand the session.
        """
        if self.state == SessionState.ESCALATED:
            await self._check_out_of_band_resolution()
            return
        if self.state not in ACTIVE_STATES:
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
            "watchdog_reconciliation",
            "no hook event before deadline; herdr reports idle/blocked",
            "",
        )
        await self._classify(last_text)

    # ── plumbing ──────────────────────────────────────────────────────────

    def _intent(self) -> str:
        """Return the authoritative intent: the approved prompt if any."""
        return self.approved_prompt or self.intent

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
        return read_cleaned(self.transcript_path)

    async def _submit(self, text: str) -> None:
        """Type ``text`` into the pane and press enter, with an explicit timeout.

        Herdr's ``agent prompt`` types without submitting, so this always
        follows it with an enter keystroke.

        Args:
            text: Prompt text, submitted exactly as given.

        Raises:
            FatalSessionError: No pane id is bound, so there is nowhere to type.
        """
        if self.pane_id is None:
            raise FatalSessionError("pane_unbound", "no pane id")
        await asyncio.to_thread(
            driver.submit_prompt,
            self.cfg.name,
            self.pane_id,
            text,
            timeout_s=SUBMIT_TIMEOUT_S,
        )

    async def _to_master(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Send one envelope to the master and wait for its reply.

        Args:
            msg_type: Protocol message type constant.
            payload: Already-serialized payload for that type.
        """
        env = Envelope(
            id=uuid.uuid4().hex,
            type=msg_type,
            session_id=self.cfg.name,
            payload=payload,
        )
        await client.request(
            Path(self.cfg.master_socket_path), env, timeout_s=MASTER_TIMEOUT_S
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
        self._log("error", error_class, detail)
        self._set_state(SessionState.ERROR)
        try:
            await self._to_master(
                T_FATAL_ERROR, {"error_class": error_class, "detail": detail}
            )
        except Exception:
            # Master unreachable: the pane degrades to stock Claude Code.
            logger.exception("could not report fatal error to master")

    def _log(self, kind: str, reasoning: str, detail: str) -> None:
        """Append one entry to this session's decision log."""
        decision_log.append(
            self.decision_log_path, kind=kind, reasoning=reasoning, detail=detail
        )

    def _set_state(self, state: SessionState) -> None:
        """Assign a new state and log the transition."""
        logger.info("session %s: %s -> %s", self.cfg.name, self.state, state)
        self.state = state

    def _herdr_state(self) -> str:
        """Report the agent's state as Herdr sees it, for the watchdog gate.

        Returns:
            Herdr's status string, or ``"unknown"`` if the query fails.
        """
        try:
            return driver.agent_status(
                driver.agent_get(self.cfg.name, timeout_s=10.0)
            )
        except Exception:
            return "unknown"  # gates the watchdog read out; never classifies


def _count_user_prompts(events: list[TranscriptEvent]) -> int:
    """Count the user prompts among ``events``."""
    return sum(1 for e in events if isinstance(e, UserPrompt))


def _render_ask_user(
    tool_input: dict[str, Any],
) -> tuple[str, list[Alternative], str]:
    """Render ``AskUserQuestion`` tool input into escalation fields.

    Question and option text pass through untouched. Entries that do not
    validate are skipped rather than guessed at, so a malformed payload still
    yields a usable "answer in the pane" escalation.

    Args:
        tool_input: Raw ``AskUserQuestion`` tool input.

    Returns:
        The rendered question text, the alternatives list (falling back to a
        single "answer in the pane" entry), and the recommendation — the first
        option's label when it is marked ``(Recommended)``.
    """
    lines: list[str] = []
    alternatives: list[Alternative] = []
    recommendation = "answer in the pane"
    raw_questions = tool_input.get("questions")
    questions = (
        cast(list[Any], raw_questions) if isinstance(raw_questions, list) else []
    )
    for raw_question in questions:
        try:
            question = Question.model_validate(raw_question)
        except ValidationError:
            logger.warning("skipping malformed AskUserQuestion entry")
            continue
        lines.append(f"{question.question} [{question.header}]")
        for i, option in enumerate(question.options):
            lines.append(f"- {option.label}: {option.description}")
            alternatives.append(
                Alternative(
                    option=option.label, pros=option.description, cons=""
                )
            )
            if i == 0 and "(Recommended)" in option.label:
                recommendation = option.label
    if not alternatives:
        alternatives = [
            Alternative(
                option="answer in the pane",
                pros="the native menu is authoritative",
                cons="",
            )
        ]
    rendered = "\n".join(lines) or "(question content unavailable)"
    return rendered, alternatives, recommendation
