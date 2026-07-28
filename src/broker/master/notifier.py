"""The ONLY notifier. Session brokers never notify."""

import asyncio

from broker.herdr import driver

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
