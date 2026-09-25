"""The test-mode ``/inject <scenario>`` composer command."""

from pathlib import Path

from broker.master.runtime import MasterRuntime
from broker.master.testmode.runner import (
    SCENARIO_DIR,
    load_scenario,
    run_scenario,
    scenario_names,
)
from broker.master.viewmodel import ViewEvent


class InjectCommand:
    """Runs a named scenario against the live runtime from the composer."""

    def __init__(
        self,
        runtime: MasterRuntime,
        posts: list[ViewEvent],
        scenario_dir: Path = SCENARIO_DIR,
    ) -> None:
        """Bind the command to the runtime it drives and the events it asserts on.

        Args:
            runtime: The running master runtime.
            posts: Every view event the runtime emits, in order.
            scenario_dir: Directory holding the ``*.json`` scenarios.
        """
        self.runtime = runtime
        self.posts = posts
        self.scenario_dir = scenario_dir

    def handles(self, text: str) -> bool:
        """Return whether a submitted composer line is a command for this handler."""
        return text.startswith("/")

    async def run(self, text: str) -> list[str]:
        """Run one command line and report its outcome.

        Args:
            text: The submitted line, starting with ``/``.

        Returns:
            The chat lines reporting the outcome, in display order.
        """
        parts = text.split(maxsplit=1)
        command = parts[0]
        if command != "/inject":
            return [
                f"unknown command {command!r}; available: /inject <name> "
                f"where <name> is one of "
                f"{', '.join(scenario_names(self.scenario_dir))}"
            ]
        if len(parts) < 2 or not parts[1].strip():
            return ["usage: /inject <scenario>"]
        name = parts[1].strip()
        scenario = load_scenario(self.scenario_dir / f"{name}.json")
        report = await run_scenario(
            self.runtime, self.posts, scenario, paths=self.runtime.paths
        )
        lines = [
            f"{'PASS' if result.passed else 'FAIL'} "
            f"[{result.index}] {result.op} — {result.detail}"
            for result in report.results
        ]
        passed = sum(1 for r in report.results if r.passed)
        summary = "PASS" if report.passed else "FAIL"
        lines.append(
            f"scenario {report.name}: {summary} "
            f"({passed}/{len(report.results)} steps)"
        )
        return lines
