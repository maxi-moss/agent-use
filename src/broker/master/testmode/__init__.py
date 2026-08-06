"""Synthetic escalation test mode: scenarios driven over the real master socket.

Master-internal tooling. It speaks the wire protocol and calls the runtime's
public methods; nothing below the socket is stubbed.
"""

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)

from broker.llm import TurnResult
from broker.master.testmode.fake_broker import FakeBrokerClient, FakeSessionSocket
from broker.master.testmode.runner import load_scenario, run_scenario
from broker.master.testmode.schemas import (
    Scenario,
    ScenarioError,
    ScenarioReport,
    StepResult,
)

_TEST_MODE_REPLY = "test mode — LLM disabled; type /inject <scenario>"


async def test_mode_llm_call(
    *,
    model: str,
    max_tokens: int,
    system: list[TextBlockParam],
    messages: list[MessageParam],
    tools: list[ToolParam],
    tool_choice: ToolChoiceParam,
) -> TurnResult:
    """Return a fixed text-only turn without contacting any model."""
    return TurnResult(text=_TEST_MODE_REPLY)


__all__ = [
    "FakeBrokerClient",
    "FakeSessionSocket",
    "Scenario",
    "ScenarioError",
    "ScenarioReport",
    "StepResult",
    "load_scenario",
    "run_scenario",
    "test_mode_llm_call",
]
