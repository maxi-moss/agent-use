"""SessionPane: the session's herdr pane and what occupies it.

The one session module that calls `broker.herdr.driver`: it starts Claude in a
pane, types into it, and reads its herdr state. It is also the single owner of
pane occupancy — the open AskUserQuestion picker and the native permission
prompt — so nothing is typed while either holds the pane.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from broker.herdr import driver
from broker.herdr.schemas import AgentStatus
from broker.transcript.schemas import AnswerValue

SUBMIT_TIMEOUT_S = 15.0
PANE_SPLIT_TIMEOUT_S = 15.0
AGENT_START_TIMEOUT_MS = 30000
AGENT_GET_TIMEOUT_S = 10.0

# Forwarded to the claude binary at spawn. "auto" classifies each tool call and
# still prompts on the risky ones, so the hook's escalation path survives;
# "bypassPermissions" would silently approve every escalation.
CLAUDE_AGENT_ARGS = ["--model", "opus", "--permission-mode", "auto"]


@dataclass(slots=True)
class OpenMenu:
    """An AskUserQuestion picker the broker left open in the pane."""

    tool_use_id: str
    escalation_id: str | None  # None: nothing escalated for it
    injected: dict[str, AnswerValue] | None  # unverified injected answers


class PaneOccupiedError(Exception):
    def __init__(self, what: str, pane_id: str | None) -> None:
        """Record which native prompt holds the pane."""
        super().__init__(
            f"{what} is open in pane {pane_id or '?'}; nothing is typed into "
            "the pane while it is"
        )
        self.what = what
        self.pane_id = pane_id


class SessionPane:
    def __init__(self, agent_name: str, *, on_change: Callable[[], None]) -> None:
        """Bind the pane owner to the session's herdr agent.

        Args:
            agent_name: The herdr agent name Claude runs under.
            on_change: Called whenever what occupies the pane changes.
        """
        self._agent_name = agent_name
        self._on_change = on_change
        self.pane_id: str | None = None
        self.permission_prompt = False
        self.open_menu: OpenMenu | None = None

    async def start(
        self,
        anchor_pane: str,
        *,
        cwd: Path,
        env: dict[str, str],
        claude_settings_path: str,
    ) -> str | None:
        """Split a pane beside ``anchor_pane`` and start Claude in it.

        Args:
            anchor_pane: The pane to split from.
            cwd: Working directory for the new pane.
            env: Environment the pane's processes, the hook included, inherit.
            claude_settings_path: The ``--settings`` file Claude starts with.

        Returns:
            The Claude session id herdr reported at start, if it reported one.
        """
        pane = await asyncio.to_thread(
            driver.pane_split,
            anchor_pane,
            direction="right",
            cwd=cwd,
            env=env,
            focus=False,
            timeout_s=PANE_SPLIT_TIMEOUT_S,
        )
        self.pane_id = pane.pane_id  # the DURABLE handle
        start = await asyncio.to_thread(
            driver.agent_start,
            self._agent_name,
            kind="claude",
            pane_id=self.pane_id,
            timeout_ms=AGENT_START_TIMEOUT_MS,
            agent_args=[*CLAUDE_AGENT_ARGS, "--settings", claude_settings_path],
        )
        session = start.agent_session
        if session is not None and session.kind == "id":
            return session.value
        return None

    def adopt(self, pane_id: str) -> None:
        """Take over a pane a previous broker started."""
        self.pane_id = pane_id

    def native_prompt(self) -> str | None:
        """Name the native prompt open in the pane, or ``None`` when there is none."""
        if self.open_menu is not None:
            return "an AskUserQuestion menu"
        if self.permission_prompt:
            return "a permission prompt"
        return None

    def reports_permission_prompt(self) -> bool:
        """Whether to report a permission prompt upstream."""
        # An open picker's own permission check can raise a permission-prompt
        # notification; the picker is the prompt the developer sees.
        return self.permission_prompt and self.open_menu is None

    def set_permission_prompt(self, pending: bool) -> None:
        """Record whether the session sits on a native permission prompt."""
        self.permission_prompt = pending
        self._on_change()

    def claim_menu(self, menu: OpenMenu) -> OpenMenu | None:
        """Record ``menu`` as the picker open in the pane.

        Args:
            menu: The picker now open.

        Returns:
            The picker it replaced, if one was open.
        """
        previous = self.open_menu
        self.open_menu = menu
        self._on_change()
        return previous

    def release_menu(self) -> None:
        """Record that the open picker has closed."""
        self.open_menu = None
        self._on_change()

    async def submit(self, text: str) -> None:
        """Type ``text`` into the session and submit it, with an explicit timeout.

        Args:
            text: Prompt text, submitted exactly as given.

        Raises:
            PaneOccupiedError: A native prompt is open in the pane; typing
                would land in it.
        """
        what = self.native_prompt()
        if what is not None:
            raise PaneOccupiedError(what, self.pane_id)
        await asyncio.to_thread(
            driver.agent_prompt,
            self._agent_name,
            text,
            timeout_s=SUBMIT_TIMEOUT_S,
        )

    def herdr_state(self) -> AgentStatus:
        """Report the agent's state as Herdr sees it, for the watchdog gate.

        Returns:
            Herdr's status, or ``"unknown"`` if the query fails.
        """
        try:
            return driver.agent_get(
                self._agent_name, timeout_s=AGENT_GET_TIMEOUT_S
            ).agent_status
        except Exception:
            return "unknown"  # gates the watchdog read out; never classifies
