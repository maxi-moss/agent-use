from app.providers.anthropic import AnthropicProvider
from app.providers.base import Provider
from app.settings import Settings


def create_provider(settings: Settings) -> Provider:
    if settings.provider == "anthropic":
        return AnthropicProvider(settings.anthropic_api_key)
    raise ValueError(settings.provider)
