"""Scenario harness: a real MasterRuntime behind a real master socket, driven
by fake broker clients over the wire. No subprocess spawns and no LLM — the
post sink is a plain list, exactly as the runtime unit harness does it."""

import asyncio
import tempfile
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from broker.config import BrokerConfig
from broker.master.pane_escalations import PaneEscalations
from broker.master.queue import EscalationQueue
from broker.master.registry import Registry
from broker.master.runtime import MasterRuntime
from broker.master.testmode import SCENARIO_DIR, load_scenario, run_scenario
from broker.master.testmode.schemas import Scenario

pytestmark = pytest.mark.scenarios


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(dir="/private/tmp") as td:
        monkeypatch.setenv("BROKER_HOME", td)
        yield Path(td)


@pytest.fixture
async def rt(home: Path) -> AsyncIterator[tuple[MasterRuntime, list[Any]]]:
    cfg = BrokerConfig(model_id="test-model", broker_home=home)
    registry = Registry.load(home / "registry.json")
    queue = EscalationQueue.load(home / "escalation-queue.json")
    panes = PaneEscalations.load(home / "pane-escalations.json")
    posts: list[Any] = []
    runtime = MasterRuntime(
        posts.append,
        registry,
        queue,
        panes,
        cfg,
        anchor_pane="%1",
        claude_json=home / "claude.json",
    )
    runtime.start()
    for _ in range(200):
        if runtime.master_socket_path.exists():
            break
        await asyncio.sleep(0.01)
    else:
        raise TimeoutError("master socket never bound")
    yield runtime, posts
    await runtime.aclose()


async def _run(
    rt: tuple[MasterRuntime, list[Any]], name: str
) -> None:
    runtime, posts = rt
    scenario = load_scenario(SCENARIO_DIR / f"{name}.json")
    report = await run_scenario(
        runtime, posts, scenario, paths=runtime.paths
    )
    assert report.passed, "\n".join(
        f"[{r.index}] {r.op}: {r.detail}"
        for r in report.results
        if not r.passed
    )


async def test_fifo(rt: tuple[MasterRuntime, list[Any]]) -> None:
    await _run(rt, "fifo")


async def test_retract_before_surface(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    await _run(rt, "retract-before-surface")


async def test_retract_while_surfaced(
    rt: tuple[MasterRuntime, list[Any]]
) -> None:
    await _run(rt, "retract-while-surfaced")


async def test_dispatch_race(rt: tuple[MasterRuntime, list[Any]]) -> None:
    await _run(rt, "dispatch-race")


async def test_duplicate_escalation(rt: tuple[MasterRuntime, list[Any]]) -> None:
    await _run(rt, "duplicate-escalation")


async def test_question_parallel(rt: tuple[MasterRuntime, list[Any]]) -> None:
    await _run(rt, "question-parallel")


async def test_broker_death(rt: tuple[MasterRuntime, list[Any]]) -> None:
    await _run(rt, "broker-death")


async def test_attach_refusal(rt: tuple[MasterRuntime, list[Any]]) -> None:
    # Attach probes the socket once and never polls, so the refusal is
    # immediate — no wait to compress.
    await _run(rt, "attach-refusal")


def test_scenario_files_all_validate() -> None:
    files = sorted(SCENARIO_DIR.glob("*.json"))
    assert files, "no scenario files found"
    for path in files:
        Scenario.model_validate_json(path.read_text(encoding="utf-8"))
