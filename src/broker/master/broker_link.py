"""Master-side probes for a session's broker process: socket liveness and
what a replacement broker needs to adopt a session it did not spawn.
"""

import asyncio
import contextlib
from pathlib import Path

from broker.config import AdoptedSession
from broker.master.registry import SessionRecord

SOCKET_PROBE_TIMEOUT_S = 2.0


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
