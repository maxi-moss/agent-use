"""Master routing agent.

Structural thin-master rule: the escalation block enters the context as the
runtime-rendered string, byte-identical — this layer never sees a payload it
could re-summarise. tool_choice is auto (never forced): the loop exits on a
text-only response, and forced choice would suppress that text.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)
from pydantic import BaseModel, ConfigDict, ValidationError

from broker import prompts
from broker.config import BrokerConfig
from broker.paths import BrokerPaths
from broker.llm import (
    LLMCallError,
    ToolCall,
    TurnResult,
    call_turn,
    strict_tool,
)
from broker.master.runtime import MasterRuntime, render_escalation, render_proposal

MAX_TOOL_ROUNDS = 6


class LLMTurnCaller(Protocol):
    """Injected seam: tests pass a fake; production binds llm.call_turn."""

    async def __call__(
        self,
        *,
        model: str,
        max_tokens: int,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        tools: list[ToolParam],
        tool_choice: ToolChoiceParam,
    ) -> TurnResult:
        ...


class SpawnSessionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    cwd: str


class ApprovePromptArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    prompt: str


class DispatchDecisionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    escalation_id: str
    decision: str


class ListSessionsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SendToSessionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    prompt: str


class GetDecisionLogArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str


class StopSessionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str


class ReactivateSessionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    intent: str


class ReassignSessionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    intent: str


@dataclass(frozen=True, slots=True)
class MasterTool[M: BaseModel]:
    """A master tool's wire schema paired with the runtime call it dispatches to."""

    name: str
    description: str
    model: type[M]
    handler: Callable[[MasterRuntime, M], Awaitable[str]]


async def _list_sessions(runtime: MasterRuntime, _: ListSessionsArgs) -> str:
    """Async adapter for the one runtime accessor that is synchronous."""
    return runtime.render_registry_summary()


# Deliberately unannotated: a tuple[MasterTool[Any], ...] annotation would solve
# M as Any at every entry and stop pyright checking handlers against their model.
_REGISTRY = (
    MasterTool(
        "spawn_session",
        "Launch a supervised session for a new task. Pass the developer's "
        "intent VERBATIM — grounding belongs to the session broker.",
        SpawnSessionArgs,
        lambda rt, a: rt.spawn_session(a.intent, a.cwd),
    ),
    MasterTool(
        "approve_prompt",
        "Approve (or relay the developer's revision of) a proposed initial "
        "prompt. The final text passes through verbatim.",
        ApprovePromptArgs,
        lambda rt, a: rt.approve_prompt(a.proposal_id, a.prompt),
    ),
    MasterTool(
        "dispatch_decision",
        "Relay the developer's resolution of the active escalation to the "
        "owning session broker, unchanged.",
        DispatchDecisionArgs,
        lambda rt, a: rt.dispatch(a.escalation_id, a.decision),
    ),
    MasterTool(
        "list_sessions",
        "Current sessions with state, budget, and intent.",
        ListSessionsArgs,
        _list_sessions,
    ),
    MasterTool(
        "send_to_session",
        "Push a new developer instruction into an existing session.",
        SendToSessionArgs,
        lambda rt, a: rt.send_prompt(a.session_id, a.prompt),
    ),
    MasterTool(
        "get_decision_log",
        "Retrieve a session broker's triage reasoning.",
        GetDecisionLogArgs,
        lambda rt, a: rt.get_decision_log(a.session_id),
    ),
    MasterTool(
        "stop_session",
        "Terminate a session and its broker.",
        StopSessionArgs,
        lambda rt, a: rt.stop_session(a.session_id),
    ),
    MasterTool(
        "reactivate_session",
        "Give a completed session a new task, keeping its broker and its "
        "chat. Pass the developer's intent VERBATIM — the session broker "
        "grounds it. Refused unless the session is completed.",
        ReactivateSessionArgs,
        lambda rt, a: rt.reactivate_session(a.session_id, a.intent),
    ),
    MasterTool(
        "reassign_session",
        "Replace a session's broker with a fresh one and give it a new task, "
        "keeping the session's pane and chat. Use when the broker is stuck or "
        "errored. Pass the developer's intent VERBATIM.",
        ReassignSessionArgs,
        lambda rt, a: rt.reassign_session(a.session_id, a.intent),
    ),
)

MASTER_TOOLS: list[ToolParam] = [
    strict_tool(t.name, t.description, t.model) for t in _REGISTRY
]

_BY_NAME: dict[str, MasterTool[Any]] = {t.name: t for t in _REGISTRY}

AUTO_ONE: ToolChoiceParam = {
    "type": "auto",
    "disable_parallel_tool_use": True,
}

_MASTER_PROMPT = prompts.load("master")


class ConversationLog:
    """The master conversation log — append-only; only a bounded window is
    ever loaded into context. Timestamps stay in the file, never in
    the rendered context (cache determinism)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, role: str, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "role": role,
            "text": text,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
            fh.flush()

    def tail(self, n: int) -> list[tuple[str, str]]:
        if not self.path.exists() or n <= 0:
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        out: list[tuple[str, str]] = []
        for line in lines[-n:]:
            if not line.strip():
                continue
            parsed: Any = json.loads(line)
            if not isinstance(parsed, dict):
                continue
            typed = cast(dict[str, Any], parsed)
            role = typed.get("role")
            text = typed.get("text")
            if isinstance(role, str) and isinstance(text, str):
                out.append((role, text))
        return out


class MasterLLM:
    def __init__(
        self,
        llm_call: LLMTurnCaller,
        runtime: MasterRuntime,
        cfg: BrokerConfig,
    ) -> None:
        self.llm_call = llm_call
        self.runtime = runtime
        self.cfg = cfg
        self.log = ConversationLog(BrokerPaths(cfg.broker_home).master_conversation)

    async def handle_developer_message(self, text: str) -> str:
        messages = self._assemble(text)
        self.log.append("developer", text)
        reply = ""
        for round_no in range(MAX_TOOL_ROUNDS):
            result = await self.llm_call(
                model=self.cfg.model_id,
                max_tokens=self.cfg.max_tokens,
                system=self._system(),
                messages=messages,
                tools=MASTER_TOOLS,
                tool_choice=AUTO_ONE,
            )
            if not result.tool_calls:
                reply = result.text
                break
            ids = [
                f"call_{round_no}_{i}" for i in range(len(result.tool_calls))
            ]
            messages.append(_assistant_message(result, ids))
            tool_results: list[dict[str, Any]] = []
            for i, call in enumerate(result.tool_calls):
                outcome = await self._execute(call)
                self.log.append(
                    "tool", f"{call.name}({json.dumps(call.input)}) -> {outcome}"
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": ids[i],
                        "content": outcome,
                    }
                )
            messages.append(
                cast(MessageParam, {"role": "user", "content": tool_results})
            )
            if result.text:
                reply = result.text  # keep the latest text if the cap hits
        else:
            reply = reply or "(tool loop reached its round cap with no reply)"
        self.log.append("assistant", reply)
        return reply

    def _system(self) -> list[TextBlockParam]:
        return [
            {
                "type": "text",
                "text": _MASTER_PROMPT,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]

    def _assemble(self, developer_message: str) -> list[MessageParam]:
        """Fresh per turn: registry summary, the single active
        escalation (verbatim block), any pending prompt proposals (verbatim
        blocks), a bounded window of recent turns, then the developer's
        message."""
        blocks: list[TextBlockParam] = [
            {
                "type": "text",
                "text": "# Session registry\n"
                + self.runtime.render_registry_summary(),
            }
        ]
        active = self.runtime.slot.active
        if active is not None:
            blocks.append({"type": "text", "text": "# Active escalation"})
            # Byte-identical to the runtime rendering — its own block, so
            # nothing is prepended to or reflowed around the broker's words.
            blocks.append({"type": "text", "text": render_escalation(active)})
        for pending in self.runtime.pending_proposals():
            blocks.append(
                {
                    "type": "text",
                    "text": "# Prompt proposal awaiting approval — session "
                    + pending.session_name,
                }
            )
            blocks.append(
                {"type": "text", "text": render_proposal(pending.payload)}
            )
        window = self.log.tail(self.cfg.recent_turns_window)
        if window:
            rendered = "\n".join(f"{role}: {text}" for role, text in window)
            blocks.append(
                {
                    "type": "text",
                    "text": "# Recent conversation\n" + rendered,
                }
            )
        blocks.append(
            {"type": "text", "text": "# Developer message\n" + developer_message}
        )
        return [cast(MessageParam, {"role": "user", "content": blocks})]

    async def _execute(self, call: ToolCall) -> str:
        tool = _BY_NAME.get(call.name)
        if tool is None:
            raise LLMCallError(f"unknown master tool {call.name!r}")
        try:
            args = tool.model.model_validate(call.input)
        except ValidationError as exc:
            raise LLMCallError(
                f"invalid input for master tool {call.name!r}: {exc}"
            ) from exc
        return await tool.handler(self.runtime, args)


def bind_call_turn(client: AsyncAnthropic) -> LLMTurnCaller:
    """Production binding of the injected seam (tests pass a fake)."""

    async def call(
        *,
        model: str,
        max_tokens: int,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        tools: list[ToolParam],
        tool_choice: ToolChoiceParam,
    ) -> TurnResult:
        return await call_turn(
            client,
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )

    return call


def _assistant_message(result: TurnResult, ids: list[str]) -> MessageParam:
    content: list[dict[str, Any]] = []
    if result.text:
        content.append({"type": "text", "text": result.text})
    for tool_use_id, call in zip(ids, result.tool_calls, strict=True):
        content.append(
            {
                "type": "tool_use",
                "id": tool_use_id,  # pairs with the tool_result appended next
                "name": call.name,
                "input": call.input,
            }
        )
    return cast(MessageParam, {"role": "assistant", "content": content})
