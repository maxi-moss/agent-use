"""The ONLY notifier. Session brokers never notify."""

import asyncio
from collections.abc import Awaitable, Callable

from broker.herdr import driver
from broker.master.viewmodel import Notice

NOTIFY_TIMEOUT_S = 10.0


async def notify_request(title: str, body: str) -> None:
    """Notify the developer that a session wants them, with the request sound."""
    await asyncio.to_thread(
        driver.notification_show,
        title,
        body=body,
        sound="request",
        timeout_s=NOTIFY_TIMEOUT_S,
    )


async def notify_done(title: str, body: str) -> None:
    """Notify the developer that a session finished, with the done sound."""
    await asyncio.to_thread(
        driver.notification_show,
        title,
        body=body,
        sound="done",
        timeout_s=NOTIFY_TIMEOUT_S,
    )


async def notify_or_notice(
    emit: Callable[[Notice], None],
    fn: Callable[[str, str], Awaitable[None]],
    title: str,
    body: str,
) -> None:
    """Send one notification, downgrading a notifier failure to a notice.

    Args:
        emit: Sink for the downgraded notice.
        fn: Notifier coroutine from this module.
        title: Notification title.
        body: Notification body, passed through verbatim.
    """
    try:
        await fn(title, body)
    except Exception as exc:
        # A dead notifier must not lose the escalation it announces.
        emit(Notice(f"notification failed: {exc}"))
