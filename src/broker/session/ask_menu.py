"""AskMenu: the AskUserQuestion decision, the menu it leaves open, and answer verification.

`SessionPane` owns the open-menu slot; `AskMenu` claims and releases it only
through the pane. `ask.py` is the LLM call site this subsystem drives.
"""

import asyncio
import contextlib
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from broker.config import SessionModelConfig
from broker.decision_log import DecisionLogKind, DecisionLogRow
from broker.llm import LLMCaller, ToolCall
from broker.protocol.constants import (
    ACTIVE_STATES,
    ASK_DECISION_ANSWER,
    DECISION_ESCALATED,
    NackCode,
    SessionState,
)
from broker.protocol.schemas import (
    Alternative,
    AskQuestionDecisionPayload,
    AskQuestionRequestPayload,
    EscalationDisclosure,
    PaneRetractPayload,
    QuestionEscalationPayload,
)
from broker.session import ask
from broker.session.llm_stack import EscalateCall, disclosure_of
from broker.session.master_link import (
    LiveStatusPusher,
    MasterSender,
    describe_refusal,
)
from broker.session.pane import OpenMenu, SessionPane
from broker.transcript.schemas import AnswerValue, AskUserAnswer, TranscriptEvent

# Both waits are named and bounded (global rule). The decision deadline sits
# inside HOOK_WAIT_SECONDS (30) so the hook never gives up while the broker
# still intends to answer — a reply after hook death would be an answer the
# broker believes in and Claude Code never saw.
ASK_DECISION_TIMEOUT_S = 20.0
# updatedInput delivery is instantaneous when it works (duration_ms: 0
# observed); this only fires when the mechanism broke or a hook was dropped.
ASK_VERIFY_TIMEOUT_S = 30.0

PHRASE_ASK = "deciding a question…"

_Job = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class _InjectedAnswers:
    """Answers the broker injected into a menu, awaiting verification."""

    tool_input: dict[str, Any]
    answers: dict[str, AnswerValue]


class AskMenu:
    def __init__(
        self,
        *,
        pane: SessionPane,
        status: LiveStatusPusher,
        llm_call: LLMCaller[ToolCall],
        model_cfg: SessionModelConfig,
        session_id: str,
        state: Callable[[], SessionState],
        intent: Callable[[], str],
        read_transcript: Callable[[], list[TranscriptEvent]],
        log: Callable[[DecisionLogRow], None],
        send: MasterSender,
        enqueue: Callable[[_Job], None],
        budget_exhausted: Callable[[], bool],
        spend_budget: Callable[[], None],
        note_developer_contact: Callable[[], Awaitable[None]],
        raise_decision_escalation: Callable[
            [EscalationDisclosure, str, str], Awaitable[None]
        ],
    ) -> None:
        """Wire the question path to the session it answers for.

        Args:
            pane: Owner of the open-menu slot.
            status: Shows the decision on the dashboard while it runs.
            llm_call: The session's tool-calling seam.
            model_cfg: Model id and token cap for the ask call.
            session_id: This session's name, carried on escalations.
            state: The session's current state.
            intent: The authoritative intent of the task being driven.
            read_transcript: Reads the session transcript as cleaned events.
            log: Appends one row to the session's decision log.
            send: Sends one message to the master.
            enqueue: Queues a job on the session's serial event queue.
            budget_exhausted: Whether the autonomous answer budget is spent.
            spend_budget: Counts one autonomous answer against the budget.
            note_developer_contact: Resets the budget after developer contact.
            raise_decision_escalation: Raises a decision escalation, given its
                disclosure, reasoning and task summary, that only a dispatched
                decision resolves.
        """
        self._pane = pane
        self._status = status
        self._llm_call = llm_call
        self._model_cfg = model_cfg
        self._session_id = session_id
        self._state = state
        self._intent = intent
        self._read_transcript = read_transcript
        self._log = log
        self._send = send
        self._enqueue = enqueue
        self._budget_exhausted = budget_exhausted
        self._spend_budget = spend_budget
        self._note_developer_contact = note_developer_contact
        self._raise_decision_escalation = raise_decision_escalation
        self._hook_replies: dict[str, AskQuestionDecisionPayload] = {}
        self._injected: dict[str, _InjectedAnswers] = {}
        self._verify_tasks: dict[str, asyncio.Task[None]] = {}

    async def aclose(self) -> None:
        """Cancel every pending verification backstop."""
        for task in list(self._verify_tasks.values()):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def decide(
        self, payload: AskQuestionRequestPayload
    ) -> AskQuestionDecisionPayload:
        """Return the hook reply for a pending AskUserQuestion, answer or escalated.

        Args:
            payload: The pending question.

        Returns:
            The decision reply for the hook.
        """
        # Cached per tool_use_id: PreToolUse can fire several times per logical
        # operation, and a duplicate must get the same reply without a second
        # LLM call.
        reply = self._hook_replies.get(payload.tool_use_id)
        if reply is None:
            with self._status.activity(PHRASE_ASK):
                reply = await self._decide(payload)
            self._hook_replies[payload.tool_use_id] = reply
        return reply

    async def _decide(
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
        state = self._state()
        if state not in ACTIVE_STATES:
            # The developer is already engaged (escalated/completed/...): do
            # not raise a second escalation on top; let the picker render.
            self._log(
                DecisionLogRow(
                    kind=DecisionLogKind.ASK_SKIPPED,
                    reasoning=f"question arrived in state {state!r}; picker left "
                    "to the developer",
                    tool_use_id=tool_use_id,
                )
            )
            self._claim(OpenMenu(tool_use_id, None, None))
            return escalated
        try:
            questions = ask.parse_questions(payload.tool_input)
        except ask.AskInputError as exc:
            self._escalate(
                tool_use_id, payload.tool_input, f"unusable question payload: {exc}"
            )
            return escalated
        if self._budget_exhausted():
            self._escalate(
                tool_use_id,
                payload.tool_input,
                "autonomy budget exhausted — this question is handed over "
                "rather than answered",
            )
            return escalated
        try:
            async with asyncio.timeout(ASK_DECISION_TIMEOUT_S):
                events = self._read_transcript()
                result = await ask.decide_questions(
                    self._llm_call,
                    self._model_cfg,
                    intent=self._intent(),
                    events=events,
                    questions=questions,
                )
        except Exception as exc:
            # Deliberately broad: LLMCallError, AnswerValidationError (both
            # attempts), timeout, and transcript failure all escalate rather
            # than narrowing to one type and dropping the rest.
            self._escalate(
                tool_use_id, payload.tool_input, f"{type(exc).__name__}: {exc}"
            )
            return escalated
        if isinstance(result, EscalateCall):
            self._escalate(
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
        self._log(
            DecisionLogRow(
                kind=DecisionLogKind.ASK_ANSWERED,
                reasoning=call.reasoning,
                detail=json.dumps(answers),
            )
        )
        self._injected[tool_use_id] = _InjectedAnswers(payload.tool_input, answers)
        self._spawn_verify(tool_use_id)
        self._spend_budget()
        return AskQuestionDecisionPayload(
            decision=ASK_DECISION_ANSWER, updated_input=updated_input
        )

    def _escalate(
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
        self._claim(OpenMenu(tool_use_id, escalation_id, None))
        # Queued: the hook blocks on this reply and must never wait on master
        # traffic.
        self._enqueue(
            lambda: self._raise_question(
                escalation_id,
                tool_input,
                reason,
                analysis,
                task_summary=task_summary,
            )
        )

    def _claim(self, menu: OpenMenu) -> None:
        """Record ``menu`` as the picker open in the pane."""
        previous = self._pane.claim_menu(menu)
        # A picker opening means the earlier one closed. The master supersedes
        # its escalation on the new raise, so only the log row is written,
        # queued behind the earlier raise so it follows that raise's row.
        if previous is not None and previous.escalation_id is not None:
            replaced = previous.escalation_id
            self._enqueue(lambda: self._log_replaced_question(replaced))

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
            DecisionLogRow(
                kind=DecisionLogKind.ESCALATION_RAISED,
                reasoning=reason,
                detail=f"AskUserQuestion menu open in pane {self._pane.pane_id or '?'}",
                task_summary=task_summary,
                escalation_id=escalation_id,
            )
        )
        payload = QuestionEscalationPayload(
            escalation_id=escalation_id,
            session_id=self._session_id,
            task_context=self._intent(),
            menu=ask.render_questions(questions),
            first_question=questions[0].question if questions else "",
            reason=reason,
            analysis=analysis,
        )
        nack = await self._send(payload, tolerated=frozenset(NackCode))
        if nack is None:
            return
        # The picker is still in the pane, so the claim stays; only the
        # escalation the master refused is dropped.
        self._log(
            DecisionLogRow(
                kind=DecisionLogKind.QUESTION_REFUSED,
                reasoning="question escalation refused",
                detail=describe_refusal(payload.MESSAGE_TYPE, nack),
            )
        )
        menu = self._pane.open_menu
        if menu is not None and menu.escalation_id == escalation_id:
            menu.escalation_id = None

    async def _log_replaced_question(self, escalation_id: str) -> None:
        """Close the question escalation of a picker a newer one replaced."""
        self._log(
            DecisionLogRow(
                kind=DecisionLogKind.RETRACTED,
                reasoning="a newer AskUserQuestion menu replaced it",
                task_summary="Replaced by a newer question",
                escalation_id=escalation_id,
            )
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
            DecisionLogRow(
                kind=DecisionLogKind.RETRACTED,
                reasoning=reason,
                task_summary=summary,
                escalation_id=escalation_id,
            )
        )
        await self._send(PaneRetractPayload(escalation_id=escalation_id, reason=reason))

    async def check_answered(self) -> None:
        """Release the open picker once the transcript records its answer."""
        # Every way the picker closes writes an answer for its tool use, so
        # the transcript is the complete clearing signal.
        menu = self._pane.open_menu
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
        self._pane.release_menu()
        self._hook_replies.pop(menu.tool_use_id, None)
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
                DecisionLogRow(
                    kind=DecisionLogKind.ASK_VERIFIED,
                    reasoning="recorded late",
                    tool_use_id=menu.tool_use_id,
                )
            )
        # A late answer matching the broker's cannot be told apart from the
        # developer picking the same options, so both read as the developer's.
        await self._retract_question(
            menu.escalation_id, reason, "User answered the questions in the pane"
        )
        if answered_by_developer:
            await self._note_developer_contact()

    def verify(self, raw: dict[str, Any]) -> None:
        """Compare the PostToolUse echo against the injected answers.

        Args:
            raw: Raw ``PostToolUse`` hook JSON for an AskUserQuestion.
        """
        tool_use_id = str(raw.get("tool_use_id", "") or "")
        # PostToolUse ends the tool call: no further PreToolUse asks for it.
        self._hook_replies.pop(tool_use_id, None)
        injected = self._injected.pop(tool_use_id, None)
        if injected is None:
            return  # not broker-answered, or already verified
        task = self._verify_tasks.pop(tool_use_id, None)
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
                DecisionLogRow(
                    kind=DecisionLogKind.ASK_VERIFIED,
                    reasoning="PostToolUse echo matches",
                    tool_use_id=tool_use_id,
                )
            )
            return
        self._enqueue(
            lambda: self._verification_failed(
                tool_use_id,
                injected,
                "the session recorded different answers than the broker "
                "injected — it is proceeding on those answers",
                answer_recorded=True,
            )
        )

    def _spawn_verify(self, tool_use_id: str) -> None:
        """Arm the transcript backstop for one injected answer."""
        task = asyncio.create_task(self._verify_backstop(tool_use_id))
        self._verify_tasks[tool_use_id] = task
        task.add_done_callback(lambda _: self._verify_tasks.pop(tool_use_id, None))

    async def _verify_backstop(self, tool_use_id: str) -> None:
        """Check the transcript when no PostToolUse confirmed the answer.

        Transcript writes are asynchronous and may lag the hooks, so this is
        a bounded second look, not the primary signal.

        Args:
            tool_use_id: The injected answer being verified.
        """
        await asyncio.sleep(ASK_VERIFY_TIMEOUT_S)
        injected = self._injected.pop(tool_use_id, None)
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
            self._enqueue(
                lambda: self._verification_failed(
                    tool_use_id, injected, reason, answer_recorded=True
                )
            )
            return
        if answer is not None and answer.answers == injected.answers:
            self._log(
                DecisionLogRow(
                    kind=DecisionLogKind.ASK_VERIFIED,
                    reasoning="transcript backstop",
                    tool_use_id=tool_use_id,
                )
            )
            return
        if answer is None:
            reason = (
                "no answer was recorded — the injected answers may never "
                "have been delivered and the menu may still be on screen"
            )
        else:
            reason = "the recorded answers differ from what the broker injected"
        answer_recorded = answer is not None
        self._enqueue(
            lambda: self._verification_failed(
                tool_use_id, injected, reason, answer_recorded=answer_recorded
            )
        )

    async def _verification_failed(
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
        self._log(
            DecisionLogRow(
                kind=DecisionLogKind.ASK_VERIFY_FAILED,
                reasoning=reason,
                tool_use_id=tool_use_id,
            )
        )
        task_summary = "Escalated an AskUserQuestion answer that failed verification"
        # With no answer recorded the menu may still be in the pane, and its
        # answer clears it. Otherwise the session already proceeded on
        # something, and only a decision in the master chat resolves it.
        if not answer_recorded:
            escalation_id = uuid.uuid4().hex
            self._claim(OpenMenu(tool_use_id, escalation_id, injected.answers))
            await self._raise_question(
                escalation_id,
                injected.tool_input,
                reason,
                None,
                task_summary=task_summary,
            )
            return
        await self._raise_decision_escalation(
            _verification_failed_disclosure(reason, self._pane.pane_id),
            reason,
            task_summary,
        )


def _verification_failed_disclosure(
    reason: str, pane_id: str | None
) -> EscalationDisclosure:
    """Build the disclosure for a failed verification the session proceeded past.

    Args:
        reason: What the verification found.
        pane_id: The session's pane, for the developer to inspect.

    Returns:
        The disclosure the developer decides on.
    """
    return EscalationDisclosure(
        escalation_title="AskUserQuestion answer verification failed",
        situation=(
            "AskUserQuestion answer verification failed — " + reason
            + f" Check pane {pane_id or '?'} and the session's recent turns."
        ),
        what_was_asked=(
            "Confirm what the session actually proceeded on, and correct "
            "it in the pane if needed."
        ),
        what_is_at_stake=(
            "The session may be running on answers the broker did not choose."
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
