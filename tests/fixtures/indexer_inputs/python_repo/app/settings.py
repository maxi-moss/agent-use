from pydantic import BaseModel

DEFAULT_PROVIDER = "anthropic"


class Settings(BaseModel):
    """Runtime settings."""

    provider: str = DEFAULT_PROVIDER
    anthropic_api_key: str = ""
