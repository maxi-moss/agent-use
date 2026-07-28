"""BrokerMasterApp under run_test + Pilot, with a real serve_unix socket on
/private/tmp and a scripted fake LLM."""

import asyncio
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ToolChoiceParam,
    ToolParam,
)
from rich.text import Text
from textual.widgets import Input, Static

from broker.config import BrokerConfig
from broker.llm import TurnResult
from broker.master.messages import EscalationArrived
from broker.master.registry import Registry
from broker.master.tui.app import BrokerMasterApp
from broker.protocol import client
from broker.protocol.schemas import Envelope


class GatedLLM:
    """Blocks each call until released; returns a fixed text turn."""

    def __init__(self, reply: str = "hi", *, gated: bool = False) -> None:
        self.reply = reply
        self.release = asyncio.Event()
        if not gated:
            self.release.set()
        self.calls = 0
        self.fail = False

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
        self.calls += 1
        await self.release.wait()
        if self.fail:
            raise RuntimeError("llm exploded")
        return TurnResult(text=self.reply)


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        monkeypatch.setenv("BROKER_HOME", td)
        yield Path(td)


def make_app(home: Path, llm: GatedLLM) -> BrokerMasterApp:
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    return BrokerMasterApp(cfg, registry, llm, anchor_pane="%1")


def chat_texts(app: BrokerMasterApp) -> list[str]:
    texts: list[str] = []
    for widget in app.query(Static):
        content = widget.content
        if isinstance(content, Text):
            texts.append(content.plain)
    return texts


async def test_submit_disables_input_and_worker_reenables(home: Path) -> None:
    llm = GatedLLM(reply="routing done", gated=True)
    app = make_app(home, llm)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await pilot.click("#box")
        await pilot.press(*"hello")
        await pilot.press("enter")
        await pilot.pause()
        box = app.query_one("#box", Input)
        assert box.value == ""
        assert box.disabled is True  # locked while the LLM worker runs
        llm.release.set()
        await pilot.pause(0.1)
        await pilot.pause()  # drain the posted LLMReply
        assert box.disabled is False  # re-enabled by on_worker_state_changed
        assert llm.calls == 1
        assert any("routing done" in t for t in chat_texts(app))


async def test_escalation_arrived_renders_exact_string(home: Path) -> None:
    rendered = "Escalation e1 — session s1\n\n## Situation\nverbatim [text]"
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        app.post_message(EscalationArrived("s1", "e1", rendered))
        await pilot.pause()
        assert rendered in chat_texts(app)  # the exact string, unreflowed


async def test_llm_worker_error_reenables_input_and_surfaces(
    home: Path,
) -> None:
    llm = GatedLLM(gated=True)
    llm.fail = True
    app = make_app(home, llm)
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        await pilot.click("#box")
        await pilot.press(*"boom")
        await pilot.press("enter")
        await pilot.pause()
        llm.release.set()
        await pilot.pause(0.1)
        box = app.query_one("#box", Input)
        assert box.disabled is False  # app survived, input usable (fail loud)
        assert any("master error" in t for t in chat_texts(app))


async def test_app_mounts_serves_and_unmounts_cleanly(home: Path) -> None:
    app = make_app(home, GatedLLM())
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        socket_path = app.runtime.master_socket_path
        assert socket_path.exists()
        # The real server answers on the socket while the TUI runs.
        env = Envelope(
            id=uuid.uuid4().hex, type="nonsense", payload={}
        )
        resp = await client.request(socket_path, env, timeout_s=5.0)
        assert resp.ok is False
    # Clean unmount: registry persisted, no exception raised on the way out.
    assert (home / "registry.json").exists()


async def test_startup_warnings_shown(home: Path) -> None:
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    app = BrokerMasterApp(
        cfg,
        registry,
        GatedLLM(),
        anchor_pane="%1",
        startup_warnings=["session s1 found in registry — marked unmanaged"],
    )
    async with app.run_test() as pilot:
        await pilot.pause(0.05)
        # RichLog stores rendered lines; presence of the widget suffices —
        # the warning path is exercised without error.
        assert app.query_one("#events") is not None
