"""OpenAI embedding client — the index's own stack.

Deliberately not factored with the session, permission, or master LLM stacks:
a shared helper is how one surface's model gets silently re-pinned to
another's.

Rules encoded here:
- AsyncOpenAI(max_retries=0) — the SDK default of 2 silently retries.
- Catch the ROOT openai.OpenAIError once; APITimeoutError subclasses
  APIConnectionError, so a discriminating chain mis-sorts timeouts.
- The key is read from the environment and passed explicitly; it is never
  written anywhere.
"""

import os
from typing import Protocol

import openai
from openai import AsyncOpenAI

from broker.config import EmbeddingConfig


class EmbeddingError(Exception):
    """Any embedding failure. One class, no retry, no fallback."""


class Embedder(Protocol):
    """The injected embedding seam: tests pass a fake, production binds OpenAI."""

    async def embed(self, texts: list[str], *, timeout_s: float) -> list[list[float]]:
        """Embed ``texts`` in order, one vector per text."""
        ...


class OpenAIEmbedder:
    def __init__(self, client: AsyncOpenAI, cfg: EmbeddingConfig) -> None:
        self._client = client
        self._cfg = cfg

    @classmethod
    def from_env(cls, cfg: EmbeddingConfig) -> "OpenAIEmbedder":
        """Build a client from ``OPENAI_API_KEY`` for the pinned model.

        Args:
            cfg: The model id pinned in ``broker.config``.

        Raises:
            EmbeddingError: The variable is unset or empty.
        """
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EmbeddingError("OPENAI_API_KEY is not set")
        return cls(AsyncOpenAI(api_key=key, max_retries=0), cfg)

    async def embed(self, texts: list[str], *, timeout_s: float) -> list[list[float]]:
        """Embed one batch with a named timeout.

        Args:
            texts: Non-empty strings; the API rejects empty input.
            timeout_s: Whole-request timeout.

        Returns:
            One vector per text, in input order.

        Raises:
            EmbeddingError: The SDK raised, or the response count is wrong.
        """
        if not texts:
            return []
        try:
            response = await self._client.embeddings.create(
                model=self._cfg.model_id,
                input=texts,
                encoding_format="float",
                timeout=timeout_s,
            )
        except openai.OpenAIError as exc:
            raise EmbeddingError(f"{type(exc).__name__}: {exc}") from exc
        if len(response.data) != len(texts):
            raise EmbeddingError(
                f"expected {len(texts)} embeddings, got {len(response.data)}"
            )
        return [
            item.embedding for item in sorted(response.data, key=lambda item: item.index)
        ]
