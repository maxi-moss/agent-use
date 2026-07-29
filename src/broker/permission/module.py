"""Permission triage for one session: judge, log, escalate, retract.

The module answers every permission prompt the native rules did not settle.
Two properties are load-bearing and everything else here serves them: the log
entry is written before the decision is returned, so a command that runs is
never a command with no record; and a failure of any kind resolves to
escalated, so a broken classifier costs the developer a prompt rather than an
unasked approval.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from broker.config import ClassifierConfig
from broker.permission import llm as llm_module
from broker.permission import permission_log
from broker.permission.cache import CacheEntry, ReuseCache, cache_key
from broker.permission.llm import PermissionCaller
from broker.permission.schemas import AllowCall
from broker.protocol import client
from broker.protocol.constants import (
    DECISION_ALLOW,
    DECISION_ESCALATED,
    NACK_SLOT_OCCUPIED,
    T_PERMISSION_ESCALATION,
    T_RETRACT,
)
from broker.protocol.schemas import (
    Envelope,
    PermissionEscalationPayload,
    PermissionSuggestion,
    RaiserIdentity,
    Response,
    RetractPayload,
)

logger = logging.getLogger(__name__)

MASTER_TIMEOUT_S = 10.0

ASK_USER_QUESTION = "AskUserQuestion"

_ASK_REASON = (
    "the session is putting a question to the developer; that question travels"
    " its own path to them, so this prompt is neither judged nor raised here"
)
_SUPPRESSED_NOTE = (
    " (not raised: the developer already has an unanswered prompt from this"
    " session waiting in the pane)"
)
_RETRACT_COMPLETED = "the tool call completed, so the prompt is gone"
_RETRACT_DEVELOPER_INPUT = "the developer typed into the session"
_RETRACT_REPEAT = "the session repeated the request, so the prompt was answered"
_RETRACT_SESSION_ENDED = "the session ended"


@dataclass
class _LiveEscalation:
    """The one escalation this module currently has with the developer."""

    escalation_id: str
    key: str


class PermissionModule:
    """Judges one session's permission prompts and owns its escalation."""

    def __init__(
        self,
        cfg: ClassifierConfig,
        *,
        session_name: str,
        master_socket_path: str,
        log_path: Path,
        intent: str,
        llm_call: PermissionCaller | None = None,
    ) -> None:
        """Build the module without touching the network or the log file.

        Args:
            cfg: Supplies the classifier's model id and token cap.
            session_name: Session this module belongs to.
            master_socket_path: Master socket escalations and retractions go to.
            log_path: Append-only permission log for this session.
            intent: Authoritative task intent every call is judged against.
            llm_call: Injected tool-calling backend. When omitted, one is built
                from an Anthropic client on first use — tests pass a fake.
        """
        self.cfg = cfg
        self.session_name = session_name
        self.master_socket_path = Path(master_socket_path)
        self.log_path = log_path
        self.intent = intent
        self._llm_call = llm_call
        self._cache = ReuseCache()
        self._live: _LiveEscalation | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    # ── the five calls the session broker makes ───────────────────────────

    async def decide(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        suggestions: list[PermissionSuggestion],
    ) -> str:
        """Judge one permission request and return the session's decision.

        Args:
            tool_name: Name of the tool the session is asking to run.
            tool_input: Arguments the session passed to it.
            suggestions: Permission suggestions carried by the request.

        Returns:
            ``allow`` when the call may execute, ``escalated`` otherwise.
        """
        if tool_name == ASK_USER_QUESTION:
            self._append(
                tool_name,
                tool_input,
                DECISION_ESCALATED,
                _ASK_REASON,
                None,
                0,
                cached=False,
            )
            return DECISION_ESCALATED

        key = cache_key(tool_name, tool_input)
        started = time.monotonic()
        entry = self._cache.get(key)
        if entry is not None:
            self._append(
                tool_name,
                tool_input,
                entry.decision,
                entry.reason,
                entry.model_id,
                _elapsed_ms(started),
                cached=True,
            )
            live = self._live
            if live is not None and live.key == key:
                self._retract(live, _RETRACT_REPEAT)
            return entry.decision

        try:
            result = await llm_module.classify(
                self._caller(),
                self.cfg,
                intent=self.intent,
                tool_name=tool_name,
                tool_input=tool_input,
                suggestions=llm_module.render_suggestions(suggestions),
            )
        except Exception as exc:
            logger.warning("permission classification failed: %s", exc)
            self._append(
                tool_name,
                tool_input,
                DECISION_ESCALATED,
                f"{type(exc).__name__}: {exc}",
                self.cfg.model_id,
                _elapsed_ms(started),
                cached=False,
            )
            return DECISION_ESCALATED

        decision = (
            DECISION_ALLOW if isinstance(result, AllowCall) else DECISION_ESCALATED
        )
        suppressed = decision == DECISION_ESCALATED and self._live is not None
        self._append(
            tool_name,
            tool_input,
            decision,
            result.reasoning + (_SUPPRESSED_NOTE if suppressed else ""),
            self.cfg.model_id,
            _elapsed_ms(started),
            cached=False,
        )
        self._cache.put(
            key,
            CacheEntry(
                decision=decision,
                reason=result.reasoning,
                model_id=self.cfg.model_id,
            ),
        )
        if decision == DECISION_ESCALATED and not suppressed:
            self._raise(key, tool_name, tool_input, result.reasoning, suggestions)
        return decision

    def note_tool_completed(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> None:
        """Retract the live escalation when its own tool call has finished.

        Args:
            tool_name: Name of the tool that completed.
            tool_input: Arguments it completed with.
        """
        live = self._live
        if live is not None and live.key == cache_key(tool_name, tool_input):
            self._retract(live, _RETRACT_COMPLETED)

    def note_developer_input(self) -> None:
        """Retract the live escalation when the developer types into the pane.

        Input can only reach the session through the pane the prompt is
        sitting in, so it post-dates the raise and settles it either way.
        """
        live = self._live
        if live is not None:
            self._retract(live, _RETRACT_DEVELOPER_INPUT)

    def note_session_ended(self) -> None:
        """Retract the live escalation when the session is over."""
        live = self._live
        if live is not None:
            self._retract(live, _RETRACT_SESSION_ENDED)

    def set_intent(self, intent: str) -> None:
        """Replace the task intent and drop every cached judgement.

        Args:
            intent: The intent every later call is judged against.
        """
        self.intent = intent
        self._cache.clear()

    # ── internals ─────────────────────────────────────────────────────────

    def _caller(self) -> PermissionCaller:
        """Return the injected caller, building one on first use."""
        if self._llm_call is None:
            self._llm_call = llm_module.bind(
                llm_module.build_classifier_client(self.cfg)
            )
        return self._llm_call

    def _append(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        decision: str,
        reason: str,
        model_id: str | None,
        latency_ms: int,
        *,
        cached: bool,
    ) -> None:
        """Append one entry to this session's permission log."""
        permission_log.append(
            self.log_path,
            tool_name=tool_name,
            tool_input=tool_input,
            decision=decision,
            reason=reason,
            model_id=model_id,
            latency_ms=latency_ms,
            cached=cached,
        )

    def _raise(
        self,
        key: str,
        tool_name: str,
        tool_input: dict[str, Any],
        reason: str,
        suggestions: list[PermissionSuggestion],
    ) -> None:
        """Claim the escalation slot and send the raise off the decision path.

        Args:
            key: Reuse key this escalation belongs to.
            tool_name: Name of the tool the session is asking to run.
            tool_input: Arguments the session passed to it.
            reason: The classifier's reasoning, passed to the developer intact.
            suggestions: Permission suggestions carried by the request.
        """
        payload = PermissionEscalationPayload(
            escalation_id=uuid.uuid4().hex,
            session_id=self.session_name,
            tool_name=tool_name,
            tool_input=tool_input,
            task_intent=self.intent,
            reason=reason,
            raised_at=datetime.now(UTC).isoformat(timespec="seconds"),
            raiser=RaiserIdentity(
                component="permission", session_id=self.session_name
            ),
            permission_suggestions=suggestions,
        )
        self._live = _LiveEscalation(
            escalation_id=payload.escalation_id, key=key
        )
        self._spawn(self._send_escalation(payload))

    def _retract(self, live: _LiveEscalation, reason: str) -> None:
        """Release the escalation slot and send the retraction off the path.

        Args:
            live: The escalation being resolved out of band.
            reason: Why it no longer needs the developer.
        """
        self._live = None
        payload = RetractPayload(
            escalation_id=live.escalation_id, reason=reason
        )
        self._spawn(self._send_retract(payload))

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run ``coro`` off the caller's path, keeping a reference to it."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _send_escalation(
        self, payload: PermissionEscalationPayload
    ) -> None:
        """Send one permission escalation and release the slot if it is refused.

        A capacity refusal is routine rather than a failure: the session's
        prompt is already in the pane, so the developer sees the decision
        either way.

        Args:
            payload: The escalation to raise.
        """
        response = await self._send(
            T_PERMISSION_ESCALATION, payload.model_dump(), payload.escalation_id
        )
        if response is not None and response.ok:
            return
        if response is not None:
            reason_code = response.payload.get("reason_code")
            if reason_code == NACK_SLOT_OCCUPIED:
                logger.info(
                    "permission escalation %s refused for capacity",
                    payload.escalation_id,
                )
            else:
                logger.error(
                    "permission escalation %s refused: %s (%s)",
                    payload.escalation_id,
                    response.payload.get("error", ""),
                    reason_code,
                )
        # Nothing is waiting with the developer, so the slot must not stay
        # claimed — a later call has to be free to raise.
        self._release(payload.escalation_id)

    async def _send_retract(self, payload: RetractPayload) -> None:
        """Send one retraction; a failed send leaves nothing to undo.

        Args:
            payload: The retraction to send.
        """
        await self._send(
            T_RETRACT, payload.model_dump(), payload.escalation_id
        )

    async def _send(
        self, msg_type: str, payload: dict[str, Any], escalation_id: str
    ) -> Response | None:
        """Send one envelope to the master and wait for its reply.

        Args:
            msg_type: Protocol message type constant.
            payload: Already-serialized payload for that type.
            escalation_id: Escalation the send concerns, for the diagnostic log.

        Returns:
            The master's reply, or ``None`` when the master was unreachable.
        """
        env = Envelope(
            id=uuid.uuid4().hex,
            type=msg_type,
            session_id=self.session_name,
            payload=payload,
        )
        try:
            return await client.request(
                self.master_socket_path, env, timeout_s=MASTER_TIMEOUT_S
            )
        except Exception:
            logger.exception(
                "could not send %s for escalation %s", msg_type, escalation_id
            )
            return None

    def _release(self, escalation_id: str) -> None:
        """Clear the slot if ``escalation_id`` still holds it."""
        if self._live is not None and self._live.escalation_id == escalation_id:
            self._live = None


def _elapsed_ms(started: float) -> int:
    """Milliseconds since ``started``, rounded down."""
    return int((time.monotonic() - started) * 1000)
