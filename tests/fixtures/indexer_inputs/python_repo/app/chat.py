from typing import overload

from app.providers import factory
from app.settings import Settings


class ChatService:
    """Sends messages through the configured provider."""

    def __init__(self, settings: Settings) -> None:
        self.provider = factory.create_provider(settings)

    def send(self, message: str) -> str:
        return self.provider.complete(self._prepare(message))

    def _prepare(self, message: str) -> str:
        def strip(text: str) -> str:
            return text.strip()

        return strip(message)


@overload
def render(value: str) -> str: ...
@overload
def render(value: int) -> str: ...
def render(value: str | int) -> str:
    """Render a value."""
    return str(value)
