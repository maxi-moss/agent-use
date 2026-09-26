"""DecisionEscalationFlow: the one live decision escalation, from its raise until it ends.

A decision escalation hands the session to the developer through the master:
it is raised, clarified on request, and ends on a dispatched decision, an
out-of-band answer in the pane, a developer prompt, a reactivation, a newer
raise or a fatal error.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from broker.config import SessionModelConfig
from broker.decision_log import DecisionLogKind, DecisionLogRow
from broker.llm import LLMCaller, ToolCall
from broker.protocol.constants import NackCode, SessionState
from broker.protocol.schemas import (
    Alternative,
    ClarifyEscalationReplyPayload,
    ClarifyEscalationRequestPayload,
    DecisionDeliveredPayload,
    DecisionUndeliveredPayload,
    DispatchDecisionPayload,
    Envelope,
    EscalationDisclosure,
    EscalationPayload,
    EscalationRetractPayload,
    Response,
    nack_response,
)
from broker.session import clarify
from broker.session.master_link import LiveStatusPusher, MasterSender
from broker.session.pane import SessionPane
from broker.session.triage import AnswerCall
from broker.transcript.schemas import TranscriptEvent, UserPrompt

# Sits under the master's CLARIFY_ESCALATION_TIMEOUT_S so the broker's own deadline
# expires first and it fails loud on its own terms.
CLARIFY_TIMEOUT_S = 45.0

PHRASE_CLARIFY = "answering a question about the escalation…"


@dataclass(slots=True)
class DecisionEscalation:
    """The one live decision escalation, from its raise until it ends."""

    payload: EscalationPayload
    user_prompt_baseline: int | None  # None: only a dispatched decision resolves it
    clarify_tasks: set[asyncio.Task[clarify.ClarifyCall]]


class DecisionEscalationFlow:
    def __init__(
        self,
        *,
        pane: SessionPane,
        status: LiveStatusPusher,
        llm_call: LLMCaller[ToolCall],
        model_cfg: SessionModelConfig,
        session_id: str,
        state: Callable[[], SessionState],
        set_state: Callable[[SessionState], None],
        intent: Callable[[], str],
        read_transcript: Callable[[], list[TranscriptEvent]],
        log: Callable[[DecisionLogRow], None],
        send: MasterSender,
        note_developer_contact: Callable[[], Awaitable[None]],
        budget_count: Callable[[], int],
    ) -> None:
        """Wire the decision-escalation lifecycle to the session it escalates for.

        Args:
            pane: The session's pane, which a dispatched decision is typed into.
            status: Shows a clarification on the dashboard while it runs.
            llm_call: The session's tool-calling seam.
            model_cfg: Model id and token cap for the clarify call.
            session_id: This session's name, carried on escalations.
            state: The session's current state.
            set_state: Assigns the session's state.
            intent: The authoritative intent of the task being driven.
            read_transcript: Reads the session transcript as cleaned events.
            log: Appends one row to the session's decision log.
            send: Sends one message to the master.
            note_developer_contact: Resets the budget after developer contact.
            budget_count: The autonomous answer budget spent so far.
        """
        self._pane = pane
        self._status = status
        self._llm_call = llm_call
        self._model_cfg = model_cfg
        self._session_id = session_id
        self._state = state
        self._set_state = set_state
        self._intent = intent
        self._read_transcript = read_transcript
        self._log = log
        self._send = send
        self._note_developer_contact = note_developer_contact
        self._budget_count = budget_count
        self._live: DecisionEscalation | None = None

    @property
    def live(self) -> DecisionEscalation | None:
        """Return the live decision escalation, if there is one."""
        return self._live

    async def clarify(
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
        escalation = self._live
        if (
            self._state() != SessionState.ESCALATED
            or escalation is None
            or escalation.payload.escalation_id != req.escalation_id
        ):
            return nack_response(
                env, "escalation no longer live", NackCode.WRONG_STATE
            )
        resolved = nack_response(
            env, "escalation resolved in the pane", NackCode.WRONG_STATE
        )
        try:
            events = self._read_transcript()
        except Exception as exc:
            self._log_clarify_failed(req.escalation_id, exc)
            return nack_response(env, f"{type(exc).__name__}: {exc}", None)
        task = asyncio.create_task(
            clarify.clarify(
                self._llm_call,
                self._model_cfg,
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
            self._log_clarify_failed(req.escalation_id, exc)
            return nack_response(env, f"{type(exc).__name__}: {exc}", None)
        finally:
            # Discard only: awaiting `task` forwards this connection task's
            # cancellation — the timeout's included — into it, so it is
            # always finished by the time control reaches here.
            escalation.clarify_tasks.discard(task)
        # Cancelling a task that already finished is a no-op, so an escalation
        # ending between the answer and this line still has to drop it.
        if self._state() != SessionState.ESCALATED or self._live is not escalation:
            return resolved
        self._log(
            DecisionLogRow(
                kind=DecisionLogKind.CLARIFIED,
                reasoning=result.reasoning,
                detail=result.answer,
            )
        )
        return Response(
            id=env.id,
            ok=True,
            payload=ClarifyEscalationReplyPayload(answer=result.answer).model_dump(),
        )

    def _log_clarify_failed(self, escalation_id: str, exc: Exception) -> None:
        """Record a clarification that produced no answer."""
        self._log(
            DecisionLogRow(
                kind=DecisionLogKind.CLARIFY_FAILED,
                reasoning=f"{type(exc).__name__}: {exc}",
                escalation_id=escalation_id,
            )
        )

    async def escalate_handover(
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
        await self.raise_escalation(
            _budget_handover_disclosure(
                self._budget_count(),
                last_assistant_message,
                result.answer,
                result.reasoning,
            ),
            result.reasoning,
            events,
            task_summary="Handed over when the autonomous answer budget ran out",
        )

    async def raise_escalation(
        self,
        disclosure: EscalationDisclosure,
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
            disclosure: The analysis the developer decides on.
            reasoning: Why it was raised; recorded in the decision log.
            events: Transcript events used to baseline the user-prompt count for
                out-of-band resolution. ``None`` means only a dispatched
                decision resolves it.
            task_summary: The escalation's one-line Reason in the outcome history.
        """
        payload = self._new_escalation(disclosure)
        self._log(
            DecisionLogRow(
                kind=DecisionLogKind.ESCALATION_RAISED,
                reasoning=reasoning,
                detail=payload.disclosure.situation,
                task_summary=task_summary,
                escalation_id=payload.escalation_id,
            )
        )
        self.end()
        self._live = DecisionEscalation(
            payload,
            _count_user_prompts(events) if events is not None else None,
            set(),
        )
        self._set_state(SessionState.ESCALATED)  # QUIESCENT until dispatch or retract
        await self._send(payload)

    def _new_escalation(self, disclosure: EscalationDisclosure) -> EscalationPayload:
        """Build a decision escalation carrying this session's identifying preamble.

        Args:
            disclosure: The analysis the developer decides on.

        Returns:
            The escalation, ready to raise.
        """
        return EscalationPayload(
            escalation_id=uuid.uuid4().hex,
            session_id=self._session_id,
            task_context=self._intent(),
            disclosure=disclosure,
        )

    async def check_out_of_band_resolution(self) -> None:
        """Retract the live decision escalation if the developer already answered.

        A user prompt beyond the recorded baseline counts as resolution.
        Resolving returns the session to driving.
        """
        escalation = self._live
        if (
            self._state() != SessionState.ESCALATED
            or escalation is None
            or escalation.user_prompt_baseline is None
        ):
            return
        baseline = escalation.user_prompt_baseline
        if _count_user_prompts(self._read_transcript()) <= baseline:
            return
        await self.retract("resolved in pane", "User answered in the pane")
        await self._note_developer_contact()

    async def retract(self, reason: str, summary: str) -> None:
        """End the live decision escalation without a dispatch and tell the master.

        Args:
            reason: Why it ended; shown to the developer.
            summary: The escalation's Solution line in the outcome history.
        """
        assert self._live is not None
        escalation_id = self._live.payload.escalation_id
        self._log(
            DecisionLogRow(
                kind=DecisionLogKind.RETRACTED,
                reasoning=reason,
                task_summary=summary,
                escalation_id=escalation_id,
            )
        )
        self.end()
        self._set_state(SessionState.DRIVING)
        await self._send(
            EscalationRetractPayload(escalation_id=escalation_id, reason=reason)
        )

    async def deliver(self, decision: DispatchDecisionPayload) -> None:
        """Submit the developer's decision, then confirm the outcome upstream.

        The decision text is typed into the pane exactly as written — developer
        text is wrapped, never rewritten. Contact with the developer resets the
        autonomous answer budget. The master resolves the escalation only on the
        delivery confirmation this sends: a decision whose id no longer matches
        the live escalation, or one whose pane write fails, reports back as
        undelivered instead of resolving.

        Args:
            decision: The dispatched decision, carrying the escalation id it
                answers and the response text to submit.
        """
        live = self._live
        if live is None or decision.escalation_id != live.payload.escalation_id:
            # The broker has moved past this escalation, so the master should
            # drop its queue entry (still_live=False). Reachable when a master
            # restart re-surfaces an escalation this broker already answered.
            self._log(
                DecisionLogRow(
                    kind=DecisionLogKind.DISPATCH_STALE,
                    reasoning="stale dispatch_decision ignored",
                    escalation_id=decision.escalation_id,
                )
            )
            await self._send(
                DecisionUndeliveredPayload(
                    escalation_id=decision.escalation_id,
                    detail="session had already moved past this escalation",
                    still_live=False,
                )
            )
            return
        try:
            await self._pane.submit(decision.response)
        except Exception as exc:
            # The master resolves only on confirmed delivery, so a failed pane
            # write — an occupied pane included — leaves the escalation live
            # there. Report the miss loudly (still_live=True); it stays
            # surfaced for a re-decide.
            detail = f"{type(exc).__name__}: {exc}"
            self._log(
                DecisionLogRow(
                    kind=DecisionLogKind.DISPATCH_FAILED,
                    reasoning="pane submission failed",
                    detail=detail,
                    escalation_id=decision.escalation_id,
                )
            )
            await self._send(
                DecisionUndeliveredPayload(
                    escalation_id=decision.escalation_id,
                    detail=detail,
                    still_live=True,
                )
            )
            return
        self.end()
        await self._note_developer_contact()
        self._set_state(SessionState.DRIVING)
        self._log(
            DecisionLogRow(
                kind=DecisionLogKind.DISPATCHED,
                reasoning="developer decision delivered",
                detail=decision.response,
                escalation_id=decision.escalation_id,
            )
        )
        # Resolution waits for this: the escalation clears on the master only
        # now that the decision has actually reached the pane.
        await self._send(DecisionDeliveredPayload(escalation_id=decision.escalation_id))

    def end(self) -> None:
        """Clear the live decision escalation and cancel any clarification bound to it.

        Every path that ends an escalation (dispatch, out-of-band retract,
        developer prompt, reactivation, a newer raise, a fatal error) routes
        here, so an in-flight clarify LLM call can never outlive the
        escalation it is about.
        """
        escalation = self._live
        self._live = None
        if escalation is not None:
            for task in escalation.clarify_tasks:
                task.cancel()


def _budget_handover_disclosure(
    count: int, asked: str, answer: str, reasoning: str
) -> EscalationDisclosure:
    """Build the disclosure handing the session over when the answer budget runs out.

    Args:
        count: Consecutive autonomous answers given without developer contact.
        asked: The question the withheld answer was replying to.
        answer: The answer the budget cap withheld, kept verbatim.
        reasoning: The broker's reasoning for that answer.

    Returns:
        The disclosure the developer decides on.
    """
    return EscalationDisclosure(
        escalation_title="Autonomous answer budget exhausted",
        situation=(
            f"Autonomous answer budget exhausted: {count} "
            "consecutive autonomous answers without developer contact. "
            "This broker is handing over."
        ),
        what_was_asked=asked,
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
        recommendation=answer,
        uncertainty=(
            "The budget cap, not doubt about the answer, forced this "
            f"escalation. Broker reasoning: {reasoning}"
        ),
        what_would_change_my_mind=(
            "Any developer response resets the budget and resumes "
            "autonomous operation."
        ),
    )


def _count_user_prompts(events: list[TranscriptEvent]) -> int:
    """Count the user prompts among ``events``."""
    return sum(1 for e in events if isinstance(e, UserPrompt))
