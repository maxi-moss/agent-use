"""MasterLLM with a scripted fake llm_call and a recording MasterRuntime
subclass — proves the plumbing never rewrites developer or broker text."""

import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)

from broker.config import BrokerConfig
from broker.llm import ToolCall, TurnResult
from broker.master.llm import MAX_TOOL_ROUNDS, MASTER_TOOLS, MasterLLM
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry, SessionRecord
from broker.paths import BrokerPaths
from broker.protocol.constants import SessionState
from broker.master.runtime import (
    MasterRuntime,
    PendingProposal,
    render_escalation,
    render_permission_escalation,
    render_proposal,
)
from broker.protocol.schemas import (
    EscalationPayload,
    PermissionEscalationPayload,
    PromptProposalPayload,
)


class RecordingRuntime(MasterRuntime):
    """Real runtime object; session-control methods record instead of act."""

    def __init__(self, registry: Registry, cfg: BrokerConfig) -> None:
        queue = EscalationQueue.load(
            BrokerPaths(cfg.broker_home).escalation_queue
        )
        super().__init__(
            lambda _msg: None, registry, queue, cfg, anchor_pane="%1"
        )
        self.spawned: list[tuple[str, str]] = []
        self.dispatched: list[tuple[str, str]] = []
        self.sent: list[tuple[str, str]] = []

    async def spawn_session(self, intent: str, cwd: str) -> str:
        self.spawned.append((intent, cwd))
        return "spawned"

    async def dispatch(self, escalation_id: str, decision: str) -> str:
        self.dispatched.append((escalation_id, decision))
        return "dispatched"

    async def send_prompt(self, session_id: str, text: str) -> str:
        self.sent.append((session_id, text))
        return "sent"


class FakeLLM:
    """Returns scripted TurnResults in order; records every call's params."""

    def __init__(self, results: list[TurnResult]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        if len(self.results) > 1:
            return self.results.pop(0)
        return self.results[0]


ESCALATION = EscalationPayload.model_validate(
    {
        "escalation_id": "e1",
        "session_id": "s1",
        "raiser": {"component": "broker", "session_id": "s1"},
        "task_context": "ctx",
        "situation": "sit",
        "what_was_asked": "asked",
        "what_is_at_stake": "stake",
        "alternatives": [{"option": "B", "pros": "p", "cons": "c"}],
        "recommendation": "rec",
        "uncertainty": "unc",
        "what_would_change_my_mind": "change",
    }
)

PERMISSION_ESCALATION = PermissionEscalationPayload.model_validate(
    {
        "escalation_id": "p1",
        "session_id": "s1",
        "raiser": {"component": "permission", "session_id": "s1"},
        "tool_name": "Bash",
        "tool_input": {"command": "git push"},
        "task_intent": "intent",
        "reason": "publishes work outside the machine",
        "raised_at": "2026-07-29T12:00:00+00:00",
    }
)

PROPOSAL = PromptProposalPayload.model_validate(
    {
        "proposal_id": "p1",
        "proposed_prompt": "prompt text",
        "grounding_summary": "grounding",
    }
)


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        monkeypatch.setenv("BROKER_HOME", td)
        yield Path(td)


@pytest.fixture
def runtime(home: Path) -> RecordingRuntime:
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    registry.upsert(
        SessionRecord(
            name="s1",
            socket_path=str(home / "s" / "s1.sock"),
            cwd="/private/tmp",
            anchor_pane="%1",
            state=SessionState.DRIVING,
        )
    )
    return RecordingRuntime(registry, cfg)


def make_master(runtime: RecordingRuntime, fake: FakeLLM) -> MasterLLM:
    return MasterLLM(fake, runtime, runtime.cfg)


def _block_texts(call: dict[str, Any]) -> list[str]:
    messages = cast(list[dict[str, Any]], call["messages"])
    texts: list[str] = []
    for message in messages:
        content = cast(list[dict[str, Any]], message["content"])
        for block in content:
            if block.get("type") == "text":
                texts.append(cast(str, block["text"]))
    return texts


async def test_intent_passes_through_unrewritten(
    runtime: RecordingRuntime,
) -> None:
    intent = "please add oauth login, exactly as I typed it -- no rewriting!"
    fake = FakeLLM(
        [
            TurnResult(
                tool_calls=[
                    ToolCall(
                        name="spawn_session",
                        input={"intent": intent, "cwd": "/private/tmp"},
                    )
                ]
            ),
            TurnResult(text="spawned s1"),
        ]
    )
    master = make_master(runtime, fake)
    reply = await master.handle_developer_message(intent)
    assert runtime.spawned == [(intent, "/private/tmp")]  # exact string
    assert reply == "spawned s1"


async def test_decision_dispatches_with_active_escalation(
    runtime: RecordingRuntime,
) -> None:
    runtime.queue.accept(ESCALATION)
    fake = FakeLLM(
        [
            TurnResult(
                tool_calls=[
                    ToolCall(
                        name="dispatch_decision",
                        input={
                            "escalation_id": "e1",
                            "decision": "use option B",
                        },
                    )
                ]
            ),
            TurnResult(text="done"),
        ]
    )
    master = make_master(runtime, fake)
    await master.handle_developer_message("use option B")
    assert runtime.dispatched == [("e1", "use option B")]


async def test_escalation_block_is_byte_identical(
    runtime: RecordingRuntime,
) -> None:
    runtime.queue.accept(ESCALATION)
    fake = FakeLLM([TurnResult(text="ok")])
    master = make_master(runtime, fake)
    await master.handle_developer_message("what is s1 waiting on?")
    texts = _block_texts(fake.calls[0])
    # The rendered escalation is its OWN context block, byte-identical to the
    # runtime renderer's output (structural thin-master rule).
    assert render_escalation(ESCALATION) in texts


async def test_active_permission_escalation_reaches_llm_context(
    runtime: RecordingRuntime,
) -> None:
    runtime.queue.accept(PERMISSION_ESCALATION)
    fake = FakeLLM([TurnResult(text="ok")])
    master = make_master(runtime, fake)
    await master.handle_developer_message("what is s1 waiting on?")
    texts = _block_texts(fake.calls[0])
    # Without it the master would answer "nothing is blocked" while a session
    # sits on a native prompt.
    assert (
        render_permission_escalation(
            PERMISSION_ESCALATION, runtime.pane_of("s1")
        )
        in texts
    )


async def test_llm_context_carries_only_the_surfaced_head(
    runtime: RecordingRuntime,
) -> None:
    runtime.queue.accept(ESCALATION)
    runtime.queue.accept(PERMISSION_ESCALATION)
    fake = FakeLLM([TurnResult(text="ok")])
    master = make_master(runtime, fake)
    await master.handle_developer_message("status?")
    texts = _block_texts(fake.calls[0])
    joined = "\n".join(texts)
    # Exactly the surfaced head, as its own verbatim block.
    assert render_escalation(ESCALATION) in texts
    assert "e1" in joined
    # The waiting escalation and the queue itself never enter LLM context.
    assert "p1" not in joined
    assert "queue" not in joined.lower()
    assert (
        render_permission_escalation(
            PERMISSION_ESCALATION, runtime.pane_of("s1")
        )
        not in texts
    )


async def test_pending_proposal_reaches_llm_context(
    runtime: RecordingRuntime,
) -> None:
    runtime.proposals["p1"] = PendingProposal("s1", PROPOSAL)
    fake = FakeLLM([TurnResult(text="ok")])
    master = make_master(runtime, fake)
    await master.handle_developer_message("approve it")
    texts = _block_texts(fake.calls[0])
    assert any("p1" in t for t in texts)  # the id the tool needs
    assert render_proposal(PROPOSAL) in texts  # verbatim, its own block


async def test_tool_loop_terminates_at_cap(
    runtime: RecordingRuntime,
) -> None:
    fake = FakeLLM(
        [
            TurnResult(
                tool_calls=[ToolCall(name="list_sessions", input={})]
            )
        ]
    )
    master = make_master(runtime, fake)
    reply = await master.handle_developer_message("loop forever")
    assert len(fake.calls) == MAX_TOOL_ROUNDS
    assert reply  # a reply exists even when the cap hits


async def test_context_order_and_cache_breakpoint(
    runtime: RecordingRuntime,
) -> None:
    fake = FakeLLM([TurnResult(text="ok")])
    master = make_master(runtime, fake)
    master.log.append("developer", "earlier message")
    master.log.append("assistant", "earlier reply")
    await master.handle_developer_message("current message")
    call = fake.calls[0]
    system = cast(list[dict[str, Any]], call["system"])
    assert len(system) == 1
    assert system[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    texts = _block_texts(call)
    registry_i = next(
        i for i, t in enumerate(texts) if t.startswith("# Session registry")
    )
    recent_i = next(
        i for i, t in enumerate(texts) if t.startswith("# Recent conversation")
    )
    dev_i = next(
        i for i, t in enumerate(texts) if t.startswith("# Developer message")
    )
    assert registry_i < recent_i < dev_i
    assert "earlier message" in texts[recent_i]
    assert "current message" in texts[dev_i]
    # The current message is not duplicated into the recent window.
    assert "current message" not in texts[recent_i]


async def test_conversation_log_windowed_not_wholesale(
    runtime: RecordingRuntime,
) -> None:
    fake = FakeLLM([TurnResult(text="ok")])
    master = make_master(runtime, fake)
    for i in range(runtime.cfg.recent_turns_window + 15):
        master.log.append("developer", f"old-entry-{i}")
    await master.handle_developer_message("now")
    texts = _block_texts(fake.calls[0])
    recent = next(t for t in texts if t.startswith("# Recent conversation"))
    assert "old-entry-0" not in recent  # oldest entries stay on disk
    assert f"old-entry-{runtime.cfg.recent_turns_window + 14}" in recent


async def test_on_activity_reports_thinking_then_the_tool_phrase(
    runtime: RecordingRuntime,
) -> None:
    fake = FakeLLM(
        [
            TurnResult(
                tool_calls=[
                    ToolCall(
                        name="spawn_session",
                        input={"intent": "task", "cwd": "/private/tmp"},
                    )
                ]
            ),
            TurnResult(text="spawned"),
        ]
    )
    master = make_master(runtime, fake)
    activities: list[str] = []
    await master.handle_developer_message("go", on_activity=activities.append)
    # One "thinking…" per LLM round, the tool's own phrase before it runs.
    assert activities == ["thinking…", "spawning a session…", "thinking…"]


def test_master_tools_are_strict_and_complete() -> None:
    names = [t["name"] for t in MASTER_TOOLS]
    assert names == [
        "spawn_session",
        "approve_prompt",
        "dispatch_decision",
        "list_sessions",
        "send_prompt_to_session",
        "get_decision_log",
        "get_permission_log",
        "stop_session",
        "reactivate_session",
        "reassign_session",
        "attach_session",
    ]
    for tool in MASTER_TOOLS:
        assert tool.get("strict") is True
        schema = cast(dict[str, Any], tool["input_schema"])
        assert schema["additionalProperties"] is False
        assert sorted(schema["required"]) == sorted(schema["properties"])
