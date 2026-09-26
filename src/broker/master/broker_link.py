"""The master's link to each session broker: the processes it spawned and
every socket call it makes to them.

A refused call returns its refusal as text: a NACK, no reply, or a broker that
will not go. ``request`` returns the raw reply and raises on no reply.
"""

import asyncio
import contextlib
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from broker.claude.settings import write_session_permissions
from broker.claude.trust import seed_trust
from broker.config import (
    AdoptedSession,
    BrokerConfig,
    ResumedTask,
    SessionBrokerConfig,
    SessionModelConfig,
)
from broker.master.registry import SessionRecord
from broker.paths import BrokerPaths
from broker.protocol import client
from broker.protocol.schemas import Response, ShutdownPayload, WireMessage, parse_nack

REQUEST_TIMEOUT_S = 10.0
STOP_WAIT_S = 10.0
SOCKET_POLL_S = 0.1
SOCKET_PROBE_TIMEOUT_S = 2.0

# Every way a socket call ends without a reply line.
LINK_FAILURES = (OSError, ConnectionError, TimeoutError)

logger = logging.getLogger(__name__)


async def broker_is_listening(path: Path) -> bool:
    """Report whether anything still accepts connections on a session socket.

    Args:
        path: Session socket to probe.

    Returns:
        ``True`` when the connection is accepted, and also when the probe
        itself is inconclusive — an ambiguous result must never read as free.
    """
    try:
        async with asyncio.timeout(SOCKET_PROBE_TIMEOUT_S):
            _, writer = await asyncio.open_unix_connection(str(path))
    except TimeoutError:
        return True  # BEFORE OSError, which TimeoutError subclasses
    except OSError:
        return False  # nothing bound, or a stale file refusing connections
    writer.close()
    with contextlib.suppress(OSError, ConnectionError):
        await writer.wait_closed()
    return True


def adoption_fields(record: SessionRecord) -> AdoptedSession:
    """Build the block a replacement broker needs to adopt a live session.

    Args:
        record: Registry record of the session a replacement broker takes over.

    Returns:
        The pane id, Claude session id and transcript path, all present.

    Raises:
        ValueError: Any of them is unknown. A broker must never adopt a
            session it only partly knows.
    """
    pane_id = record.pane_id
    claude_session_id = record.claude_session_id
    transcript_path = record.transcript_path
    if not (pane_id and claude_session_id and transcript_path):
        missing = sorted(
            field
            for field, value in (
                ("pane_id", pane_id),
                ("claude_session_id", claude_session_id),
                ("transcript_path", transcript_path),
            )
            if not value
        )
        raise ValueError(
            f"session {record.name} cannot be adopted: the registry has no "
            + ", ".join(missing)
        )
    return AdoptedSession(
        pane_id=pane_id,
        claude_session_id=claude_session_id,
        transcript_path=transcript_path,
    )


class BrokerLink:
    def __init__(
        self,
        paths: BrokerPaths,
        cfg: BrokerConfig,
        master_socket_path: Path,
        claude_json: Path,
        on_exit: Callable[[str, int], None],
    ) -> None:
        """Hold what a spawned broker is configured from.

        Args:
            paths: Layout of the broker home.
            cfg: Broker configuration.
            master_socket_path: Socket every spawned broker reports to.
            claude_json: Claude Code's ``~/.claude.json`` state file.
            on_exit: Called with the session name and exit code when a tracked
                broker exits without a ``stop`` having asked it to.
        """
        self.paths = paths
        self.cfg = cfg
        self.master_socket_path = master_socket_path
        self._claude_json = claude_json
        self._on_exit = on_exit
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._stopping: set[asyncio.subprocess.Process] = set()
        self._watchers: set[asyncio.Task[None]] = set()

    async def spawn(
        self,
        record: SessionRecord,
        *,
        adopt: AdoptedSession | None,
        resume: ResumedTask | None,
    ) -> int:
        """Start a session-broker subprocess for ``record`` and track it.

        Args:
            record: Supplies the identity, socket, cwd, intent and budget the
                broker starts from.
            adopt: Pane, Claude session and transcript of a running session the
                broker takes over; ``None`` starts a fresh one.
            resume: Approved prompt and completed-ness the broker resumes
                instead of grounding; ``None`` grounds a new task.

        Returns:
            The spawned process's pid.
        """
        if adopt is None:
            # BEFORE spawn — the dialog eats input.
            seed_trust(Path(record.cwd), self._claude_json)
        settings_path = self.paths.session_claude_settings(record.name)
        # The rules have to be on disk before the session reads them: the
        # master is the only writer of anything outside the repo.
        write_session_permissions(
            settings_path, self.cfg.permission_rules.model_dump()
        )
        config = SessionBrokerConfig(
            name=record.name,
            socket_path=record.socket_path,
            master_socket_path=str(self.master_socket_path),
            broker_home=self.paths.home,
            cwd=record.cwd,
            anchor_pane=record.anchor_pane,
            intent=record.intent,
            budget_count=record.budget_count,
            session_model=SessionModelConfig(
                model_id=self.cfg.model_id, max_tokens=self.cfg.max_tokens
            ),
            classifier=self.cfg.classifier,
            embedding=self.cfg.embedding,
            watchdog_seconds=self.cfg.watchdog_seconds,
            budget_max=self.cfg.budget_max,
            claude_settings_path=str(settings_path),
            adopt=adopt,
            resume=resume,
        )
        stderr_path = self.paths.session_stderr(record.name)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        # By module string, never by import — keeps the module boundary
        # structural. -I isolates the subprocess from the master's cwd and
        # PYTHONPATH.
        with stderr_path.open("ab") as stderr_file:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-m",
                "broker.session",
                "--config-json",
                config.model_dump_json(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stderr_file,
                stderr=asyncio.subprocess.STDOUT,
            )
        self._procs[record.name] = proc
        watcher = asyncio.create_task(self._watch(record.name, proc))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        return proc.pid

    async def _watch(self, name: str, proc: asyncio.subprocess.Process) -> None:
        """Report a tracked broker's exit unless a stop asked for it."""
        returncode = await proc.wait()
        if proc in self._stopping:
            self._stopping.discard(proc)
            return
        if self._procs.get(name) is not proc:
            return
        del self._procs[name]
        try:
            self._on_exit(name, returncode)
        except Exception:
            logger.exception("session %s: broker exit handler failed", name)

    async def stop(self, record: SessionRecord) -> str | None:
        """Ask a session's broker to shut down and wait for the process to exit.

        Only a process this master spawned is waited on or terminated.

        Args:
            record: Registry record of the session to stop.

        Returns:
            ``None`` once stopped, otherwise the refusal: the spawned process
            was still running ``STOP_WAIT_S`` after it was terminated.
        """
        proc = self._procs.get(record.name)
        if proc is not None:
            self._stopping.add(proc)
        with contextlib.suppress(*LINK_FAILURES):
            await self.request(record, ShutdownPayload(), timeout_s=REQUEST_TIMEOUT_S)
        if proc is None:
            return None
        try:
            async with asyncio.timeout(STOP_WAIT_S):
                await proc.wait()
        except TimeoutError:
            proc.terminate()
            try:
                async with asyncio.timeout(STOP_WAIT_S):
                    await proc.wait()
            except TimeoutError:
                return (
                    f"session {record.name}: broker pid {proc.pid} still "
                    f"running {STOP_WAIT_S:.0f} s after terminate"
                )
        if self._procs.get(record.name) is proc:
            del self._procs[record.name]
        return None

    def forget(self, name: str) -> None:
        """Stop tracking a session's broker process without waiting on it."""
        self._procs.pop(name, None)

    async def aclose(self) -> None:
        """Stop watching every tracked broker; the processes keep running."""
        watchers = set(self._watchers)
        if not watchers:
            return
        for watcher in watchers:
            watcher.cancel()
        await asyncio.wait(watchers)

    async def require_socket_free(self, record: SessionRecord) -> str | None:
        """Wait until nothing answers on a session's socket.

        The replacement broker binds the same path, and a unix socket rebind
        over a live listener succeeds silently — two brokers would then split
        the pane's hook traffic between them.

        Args:
            record: Registry record of the session being reassigned.

        Returns:
            ``None`` once the socket is free, otherwise the refusal: a broker
            was still accepting connections after ``STOP_WAIT_S``.
        """
        path = Path(record.socket_path)
        deadline = asyncio.get_running_loop().time() + STOP_WAIT_S
        while await broker_is_listening(path):
            if asyncio.get_running_loop().time() >= deadline:
                return (
                    f"session {record.name}: a broker is still serving "
                    f"{path} after {STOP_WAIT_S:.0f} s — refusing to reassign"
                )
            await asyncio.sleep(SOCKET_POLL_S)
        return None

    async def deliver(
        self, record: SessionRecord, payload: WireMessage, *, rejection: str
    ) -> str | None:
        """Send ``payload`` to a session's broker, reporting a refusal.

        Args:
            record: Registry record of the target session.
            payload: Message to send.
            rejection: Message returned when the session NACKs.

        Returns:
            ``None`` when the session ACKed, otherwise the refusal ``query``
            returns.
        """
        reply = await self.query(
            record, payload, timeout_s=REQUEST_TIMEOUT_S, rejection=rejection
        )
        return reply if isinstance(reply, str) else None

    async def query(
        self,
        record: SessionRecord,
        payload: WireMessage,
        *,
        timeout_s: float,
        rejection: str,
    ) -> Response | str:
        """Send ``payload`` to a session's broker and return its ACK or the refusal.

        Args:
            record: Registry record of the target session.
            payload: Message to send.
            timeout_s: Hard deadline for the whole exchange.
            rejection: Message returned when the session NACKs.

        Returns:
            The ACK; on a NACK, ``rejection`` with the broker's own reason
            appended when it sent one; on no reply, a line naming the failure
            and whether the message may have reached the session.
        """
        try:
            resp = await self.request(record, payload, timeout_s=timeout_s)
        except (FileNotFoundError, ConnectionRefusedError) as exc:
            return (
                f"session {record.name} did not reply to "
                f"{payload.MESSAGE_TYPE}: {exc!r} — nothing was sent"
            )
        except LINK_FAILURES as exc:
            return (
                f"session {record.name} did not reply to "
                f"{payload.MESSAGE_TYPE}: {exc!r} — it may have received it; "
                "do not resend without the developer"
            )
        if resp.ok:
            return resp
        reason = parse_nack(resp).error
        if reason:
            rejection = f"{rejection}: {reason}"
        return rejection

    async def request(
        self, record: SessionRecord, payload: WireMessage, *, timeout_s: float
    ) -> Response:
        """Send ``payload`` to a session's broker and return its reply.

        Args:
            record: Registry record of the target session.
            payload: Message to send.
            timeout_s: Hard deadline for the whole exchange.

        Returns:
            The broker's reply, ACK or NACK.
        """
        return await client.send(
            Path(record.socket_path), payload, session_id=None, timeout_s=timeout_s
        )
