from .base import Provider


class AnthropicProvider(Provider):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def complete(self, prompt: str) -> str:
        return self._call(prompt)

    def _call(self, prompt: str) -> str:
        return prompt
