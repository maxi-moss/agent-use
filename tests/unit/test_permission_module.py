"""PermissionModule against a fake classifier and a stub master socket.

The module is exercised through its five public calls only; the master side is
a real unix socket so the reply-before-raise ordering is observable.
"""

import asyncio
import json
import tempfile
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pytest
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)

from broker.config import ClassifierConfig
from broker.permission import PermissionModule
from broker.permission.llm import PermissionCallError, ToolCall
from broker.protocol.constants import (
    DECISION_ALLOW,
    DECISION_ESCALATED,
    NACK_MALFORMED,
    NACK_SLOT_OCCUPIED,
    T_PANE_ESCALATION,
    T_PANE_RETRACT,
)
from broker.protocol.schemas import Envelope, Response
from broker.protocol.server import serve_unix

CFG = ClassifierConfig()

READ_INPUT: dict[str, Any] = {"file_path": "/repo/a.py"}
PUSH_INPUT: dict[str, Any] = {"command": "git push origin main"}
DEPLOY_INPUT: dict[str, Any] = {"command": "./deploy.sh"}

ALLOW = ToolCall(name="allow", input={"reasoning": "reversible read"})
ESCALATE = ToolCall(name="escalate", input={"reasoning": "publishes to a remote"})
ESCALATE_2 = ToolCall(name="escalate", input={"reasoning": "deploys"})

# Long enough for an in-process socket round trip to finish, short enough that
# a test asserting nothing happened still runs fast.
SETTLE_S = 0.05


class FakeLLM:
    """Scripted classifier. An empty script means an unexpected extra call."""

    def __init__(self, *results: ToolCall) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.never_resolve = False

    async def __call__(
        self,
        *,
        model: str,
        max_tokens: int,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        tools: list[ToolParam],
        tool_choice: ToolChoiceParam,
    ) -> ToolCall:
        self.calls.append({"model": model, "system": system, "messages": messages})
        if self.never_resolve:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return self.results.pop(0)

    def sent_text(self, index: int) -> str:
        content = cast(
            list[dict[str, Any]], self.calls[index]["messages"][0]["content"]
        )
        return "".join(cast(str, block["text"]) for block in content)


class StubMaster:
    """Records every envelope; ACKs unless a nack payload is configured."""

    def __init__(self, nack: dict[str, Any] | None = None) -> None:
        self.received: list[Envelope] = []
        self.nack = nack

    async def __call__(self, env: Envelope) -> Response | None:
        self.received.append(env)
        if self.nack is not None and env.type == T_PANE_ESCALATION:
            return Response(id=env.id, ok=False, payload=dict(self.nack))
        return Response(id=env.id, ok=True)

    def of_type(self, msg_type: str) -> list[Envelope]:
        return [e for e in self.received if e.type == msg_type]

    async def wait_for(
        self, msg_type: str, *, count: int = 1, timeout: float = 5.0
    ) -> Envelope:
        async with asyncio.timeout(timeout):
            while len(self.of_type(msg_type)) < count:
                await asyncio.sleep(0.01)
        return self.of_type(msg_type)[count - 1]


@pytest.fixture
def home() -> Iterator[Path]:
    # /private/tmp keeps the socket path inside the platform's length limit.
    with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
        yield Path(d)


def entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@asynccontextmanager
async def _module(
    home: Path,
    llm: FakeLLM,
    *,
    master: StubMaster | None = None,
    reachable: bool = True,
) -> AsyncGenerator[tuple[PermissionModule, StubMaster, Path]]:
    master = master or StubMaster()
    sock = home / "m.sock"
    server = await serve_unix(sock, master) if reachable else None
    log_path = home / "permissions.ndjson"
    module = PermissionModule(
        CFG,
        session_name="s1",
        master_socket_path=str(sock),
        log_path=log_path,
        intent="add a login page",
        llm_call=llm,
    )
    try:
        yield module, master, log_path
    finally:
        if server is not None:
            server.close()
            await server.wait_closed()


async def test_allow_logs_before_reply(home: Path) -> None:
    """The entry is on disk the instant the decision exists, never after."""
    llm = FakeLLM(ALLOW)
    async with _module(home, llm) as (module, master, log):
        decision = await module.decide("Read", READ_INPUT, [])
        written = entries(log)  # read before the loop can run anything else
    assert decision == DECISION_ALLOW
    assert len(written) == 1
    assert written[0]["decision"] == DECISION_ALLOW
    assert written[0]["reason"] == "reversible read"
    assert written[0]["model_id"] == "claude-haiku-4-5"
    assert written[0]["tool_input"] == READ_INPUT
    assert master.received == []


@pytest.mark.parametrize(
    "error",
    [
        PermissionCallError("model refused (stop_reason=refusal)"),
        PermissionCallError("APITimeoutError: took too long"),
        RuntimeError("something nobody predicted"),
    ],
)
async def test_error_resolves_to_escalated(home: Path, error: Exception) -> None:
    llm = FakeLLM()
    llm.error = error
    async with _module(home, llm) as (module, master, log):
        decision = await module.decide("Bash", PUSH_INPUT, [])
        await asyncio.sleep(SETTLE_S)
    assert decision == DECISION_ESCALATED
    written = entries(log)
    assert len(written) == 1
    assert str(error) in written[0]["reason"]
    assert written[0]["decision"] == DECISION_ESCALATED
    assert master.received == []


async def test_askuserquestion_gate_no_inference(home: Path) -> None:
    llm = FakeLLM()
    llm.never_resolve = True  # any inference at all would hang this test
    async with _module(home, llm) as (module, master, log):
        async with asyncio.timeout(2.0):
            decision = await module.decide(
                "AskUserQuestion", {"question": "which provider?"}, []
            )
        await asyncio.sleep(SETTLE_S)
    assert decision == DECISION_ESCALATED
    assert llm.calls == []
    assert master.received == []
    written = entries(log)
    assert len(written) == 1
    assert written[0]["model_id"] is None


def _completed(module: PermissionModule) -> None:
    module.note_tool_completed("Bash", PUSH_INPUT)


def _developer_input(module: PermissionModule) -> None:
    module.note_developer_input()


def _session_ended(module: PermissionModule) -> None:
    module.note_session_ended()


@pytest.mark.parametrize(
    "signal",
    [_completed, _developer_input, _session_ended],
    ids=["tool_completed", "developer_input", "session_ended"],
)
async def test_signals_1_2_4_retract(
    home: Path, signal: Callable[[PermissionModule], None]
) -> None:
    llm = FakeLLM(ESCALATE)
    async with _module(home, llm) as (module, master, _):
        assert await module.decide("Bash", PUSH_INPUT, []) == DECISION_ESCALATED
        raised = await master.wait_for(T_PANE_ESCALATION)
        signal(module)
        retract = await master.wait_for(T_PANE_RETRACT)
        # A second signal has nothing left to resolve.
        signal(module)
        await asyncio.sleep(SETTLE_S)
        assert len(master.of_type(T_PANE_RETRACT)) == 1
    assert retract.payload["escalation_id"] == raised.payload["escalation_id"]
    assert retract.payload["reason"]


async def test_unrelated_tool_completion_does_not_retract(home: Path) -> None:
    llm = FakeLLM(ESCALATE)
    async with _module(home, llm) as (module, master, _):
        await module.decide("Bash", PUSH_INPUT, [])
        await master.wait_for(T_PANE_ESCALATION)
        module.note_tool_completed("Read", READ_INPUT)
        await asyncio.sleep(SETTLE_S)
        assert master.of_type(T_PANE_RETRACT) == []


async def test_second_escalation_supersedes_the_first(home: Path) -> None:
    """Reaching a second prompt means the first was answered in the pane.

    The first must be retracted rather than orphaned in the master's slot, and
    the second must still reach the developer — swallowing it would leave the
    master silent for the rest of the session.
    """
    llm = FakeLLM(ESCALATE, ESCALATE_2)
    async with _module(home, llm) as (module, master, log):
        assert await module.decide("Bash", PUSH_INPUT, []) == DECISION_ESCALATED
        first = await master.wait_for(T_PANE_ESCALATION)
        assert await module.decide("Bash", DEPLOY_INPUT, []) == DECISION_ESCALATED
        second = await master.wait_for(T_PANE_ESCALATION, count=2)
        retract = await master.wait_for(T_PANE_RETRACT)
        await asyncio.sleep(SETTLE_S)
    assert retract.payload["escalation_id"] == first.payload["escalation_id"]
    assert second.payload["tool_input"] == DEPLOY_INPUT
    # The slot is one-per-session, so the retraction has to land first or the
    # replacement is refused for capacity by the escalation it replaces.
    order = [e.type for e in master.received]
    assert order.index(T_PANE_RETRACT) < order.index(T_PANE_ESCALATION, 1)
    written = entries(log)
    assert [e["reason"] for e in written] == ["publishes to a remote", "deploys"]


async def test_slot_occupied_nack_is_routine(home: Path) -> None:
    master = StubMaster(
        nack={
            "error": "an escalation is already live",
            "reason_code": NACK_SLOT_OCCUPIED,
        }
    )
    llm = FakeLLM(ESCALATE, ESCALATE_2)
    async with _module(home, llm, master=master) as (module, _, log):
        assert await module.decide("Bash", PUSH_INPUT, []) == DECISION_ESCALATED
        await master.wait_for(T_PANE_ESCALATION)
        await asyncio.sleep(SETTLE_S)  # the refusal lands and frees the slot
        assert await module.decide("Bash", DEPLOY_INPUT, []) == DECISION_ESCALATED
        # A refused raise is not a live escalation, so the next one still goes.
        await master.wait_for(T_PANE_ESCALATION, count=2)
    written = entries(log)
    assert [e["reason"] for e in written] == ["publishes to a remote", "deploys"]


async def test_reply_arrives_with_master_unreachable(home: Path) -> None:
    """A dead master costs the developer nothing on the blocking path."""
    llm = FakeLLM(ESCALATE)
    async with _module(home, llm, reachable=False) as (module, _, log):
        async with asyncio.timeout(2.0):
            decision = await module.decide("Bash", PUSH_INPUT, [])
        assert decision == DECISION_ESCALATED
        assert len(entries(log)) == 1
        await asyncio.sleep(SETTLE_S)


async def test_set_intent_changes_what_calls_are_judged_against(
    home: Path,
) -> None:
    llm = FakeLLM(ALLOW, ALLOW)
    async with _module(home, llm) as (module, _, _log):
        assert await module.decide("Read", READ_INPUT, []) == DECISION_ALLOW
        module.set_intent("rewrite the billing exporter")
        assert await module.decide("Read", READ_INPUT, []) == DECISION_ALLOW
        await asyncio.sleep(SETTLE_S)
    assert "add a login page" in llm.sent_text(0)
    assert "rewrite the billing exporter" in llm.sent_text(1)


async def test_non_capacity_nack_still_frees_the_slot(home: Path) -> None:
    master = StubMaster(
        nack={"error": "payload rejected", "reason_code": NACK_MALFORMED}
    )
    llm = FakeLLM(ESCALATE)
    async with _module(home, llm, master=master) as (module, _, _log):
        assert await module.decide("Bash", PUSH_INPUT, []) == DECISION_ESCALATED
        await master.wait_for(T_PANE_ESCALATION)
        await asyncio.sleep(SETTLE_S)
        # Nothing reached the developer, so there is nothing to retract.
        module.note_session_ended()
        await asyncio.sleep(SETTLE_S)
        assert master.of_type(T_PANE_RETRACT) == []
